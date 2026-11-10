#!/usr/bin/env bash
# 01_locate.sh — Interactive optical satellite verifier & BBOX coordinate finder
set -euo pipefail
source "$(dirname "$0")/config.sh"

LOCATOR_HTML="$DATA_DIR/locator.html"

# Extract actual GCP footprints + full-scene scout PNG bounds via gdalinfo & python3
FOOTPRINTS_JSON="$(python3 - "$DATA_DIR" "$SCOUT_DIR" <<'PY'
import json, glob, os, subprocess, sys

data_dir, scout_dir = sys.argv[1], sys.argv[2]
scenes = []
for tif in sorted(glob.glob(os.path.join(data_dir, "*.tiff")) + glob.glob(os.path.join(data_dir, "*.tif"))):
    try:
        info = json.loads(subprocess.check_output(["gdalinfo", "-json", tif], stderr=subprocess.DEVNULL))
        gcps = info.get("gcps", {}).get("gcpList", [])
        if not gcps:
            continue
        lons = [g["x"] for g in gcps]
        lats = [g["y"] for g in gcps]
        w, s, e, n = min(lons), min(lats), max(lons), max(lats)
        stem = os.path.splitext(os.path.basename(tif))[0]
        scout_png = os.path.join(scout_dir, stem + ".png")
        scenes.append({
            "name": stem,
            "bounds": [[s, w], [n, e]],
            "bbox_str": f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}",
            "png": scout_png if os.path.exists(scout_png) else ""
        })
    except Exception:
        pass
print(json.dumps(scenes))
PY
)"

cat > "$LOCATOR_HTML" <<EOF
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>SIAM Coordinate Verifier & BBOX Finder</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { margin: 0; font-family: -apple-system, monospace; }
  #map { height: 100vh; width: 100vw; cursor: crosshair; }
  .panel {
    position: absolute; top: 12px; right: 12px; z-index: 1000;
    background: rgba(20,20,24,0.92); color: #eee; padding: 14px;
    border-radius: 8px; width: 360px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }
  .panel code { display: block; background: #111; padding: 8px; margin: 6px 0; color: #4af626; user-select: all; }
  .hint { color: #aaa; margin-top: 6px; line-height: 1.4; }
</style>
</head>
<body>
<div id="map"></div>
<div class="panel">
  <strong>Target ROI BBOX (Gold)</strong>
  <code id="bbox-out">ROI_BBOX="$ROI_BBOX"</code>
  <strong>Regional Footprint (Green)</strong>
  <code>TRAIN_BBOX="$TRAIN_BBOX"</code>
  <div id="cursor-coords">Cursor: --</div>
  <div style="margin-top:8px;">
    <label>SAR Scout Overlay Opacity: <input id="op" type="range" min="0" max="1" step="0.05" value="0.65"/></label>
  </div>
  <div class="hint">
    • <b>Shift + Drag</b> on the map to draw a new AOI rectangle and copy <code>BBOX="..."</code>.<br/>
    • <b>Cyan dashed boxes</b> show full Sentinel-1 TIFF footprints.
  </div>
</div>
<script>
const [w, s, e, n] = [$ROI_BBOX];
const [tw, ts, te, tn] = [$TRAIN_BBOX];
const scenes = $FOOTPRINTS_JSON;

const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {attribution: 'Esri Satellite'});
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {attribution: 'OSM'});
const labels = L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}');

const map = L.map('map', { layers: [sat, labels], boxZoom: false }).fitBounds([[Math.min(s, ts), Math.min(w, tw)], [Math.max(n, tn), Math.max(e, te)]]);
L.control.layers({"Optical Satellite": sat, "Street Map": osm}, {"Place Labels": labels}, {position: 'topleft'}).addTo(map);

let aoiRect = L.rectangle([[s, w], [n, e]], {color: '#ffcc00', weight: 2, fillOpacity: 0.15}).addTo(map);
if ('$TRAIN_BBOX' !== '$ROI_BBOX') {
  L.rectangle([[ts, tw], [tn, te]], {color: '#00ff88', weight: 2, dashArray: '4,4', fillOpacity: 0.05}).addTo(map);
}
const overlays = [];

scenes.forEach(sc => {
  L.rectangle(sc.bounds, {color: '#00e5ff', weight: 1.5, dashArray: '6,4', fill: false})
   .bindPopup('<b>' + sc.name + '</b><br>Full Bounds: ' + sc.bbox_str).addTo(map);
  if (sc.png) {
    overlays.push(L.imageOverlay('file://' + sc.png, sc.bounds, {opacity: 0.65}).addTo(map));
  }
});

document.getElementById('op').addEventListener('input', ev => {
  overlays.forEach(o => o.setOpacity(ev.target.value));
});

map.on('mousemove', ev => {
  document.getElementById('cursor-coords').textContent =
    'Cursor (lon, lat): ' + ev.latlng.lng.toFixed(5) + ', ' + ev.latlng.lat.toFixed(5);
});

// Shift + Drag to select a new BBOX
let startLatLng = null;
map.on('mousedown', ev => {
  if (!ev.originalEvent.shiftKey) return;
  map.dragging.disable();
  startLatLng = ev.latlng;
});
map.on('mousemove', ev => {
  if (!startLatLng) return;
  aoiRect.setBounds(L.latLngBounds(startLatLng, ev.latlng));
});
map.on('mouseup', ev => {
  if (!startLatLng) return;
  map.dragging.enable();
  const b = L.latLngBounds(startLatLng, ev.latlng);
  startLatLng = null;
  const str = 'BBOX="' + [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map(v => v.toFixed(4)).join(',') + '"';
  document.getElementById('bbox-out').textContent = str;
  navigator.clipboard?.writeText(str);
});
</script>
</body>
</html>
EOF

echo "==> Opening optical satellite verifier: $LOCATOR_HTML"
open_files "$LOCATOR_HTML"
