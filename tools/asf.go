// asf.go - Search NASA ASF Sentinel-1 GRD scenes and extract VV/VH GeoTIFFs.
package main

import (
	"archive/zip"
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"
)

const api = "https://api.daac.asf.alaska.edu/services/search/param"

var errEarthdataUnauthorized = errors.New("Earthdata authentication failed")

type result struct {
	Features []struct {
		Geometry struct {
			Type        string         `json:"type"`
			Coordinates [][][2]float64 `json:"coordinates"`
		} `json:"geometry"`
		Properties struct {
			URL             string      `json:"url"`
			FileName        string      `json:"fileName"`
			SceneName       string      `json:"sceneName"`
			StartTime       string      `json:"startTime"`
			FlightDirection string      `json:"flightDirection"`
			Polarization    string      `json:"polarization"`
			PathNumber      json.Number `json:"pathNumber"`
			Bytes           json.Number `json:"bytes"`
		} `json:"properties"`
	} `json:"features"`
}

type scene struct {
	name, url, date, dir, orbit, pols string
	ts                                time.Time
	bytes                             int64
	poly                              Polygon // lon/lat polygon
	accepted                          bool
	rejectReason                      string
}

func main() {
	log.SetFlags(0)
	progress := newProgressDisplay(os.Stderr)
	bbox := flag.String("bbox", "", "west,south,east,north")
	pointFlag := flag.String("point", "", "seed coordinate: lon,lat to discover scenes and allocate aligned BBOX")
	seedMargin := flag.Float64("seed-margin", 5.0, "minimum distance in km from seed point to scene edge")
	minOverlap := flag.Float64("min-overlap", 0.70, "minimum mutual overlap fraction with consensus polygon (0..1)")
	autoOrbit := flag.Bool("auto-orbit", true, "automatically select dominant orbit track when -point is used")
	saveBBox := flag.String("save-bbox", "", "optional path to save the calculated maximum aligned BBOX")
	from := flag.String("from", "", "start date YYYY-MM-DD (required)")
	to := flag.String("to", "", "end date YYYY-MM-DD (required)")
	out := flag.String("out", "/Users/nlane/Development/siam/data", "output directory for extracted TIFFs")
	direction := flag.String("direction", "", "ASCENDING or DESCENDING")
	orbit := flag.String("orbit", "", "relative orbit number")
	polFlag := flag.String("pol", "vv,vh", "polarizations to extract: vv, vh, or vv,vh")
	scenesFlag := flag.String("scenes", "", "comma-separated scene names, or '-' to read from stdin")
	listOnly := flag.Bool("ls", false, "list available scenes to stdout without downloading")
	workers := flag.Int("workers", 6, "concurrent downloads")
	limit := flag.Int("limit", 100, "maximum search results")
	flag.Parse()

	var seedPt *Point
	var err error
	if *pointFlag != "" {
		seedPt, err = parseSeedPoint(*pointFlag)
		if err != nil {
			log.Fatalf("point: %v", err)
		}
		progress.Printf("seed: lon=%.4f, lat=%.4f", seedPt.X, seedPt.Y)
	}

	if (*bbox == "" && seedPt == nil) || *from == "" || *to == "" || *workers < 1 || *limit < 1 {
		flag.Usage()
		log.Fatal("either -bbox or -point, and from, to, workers, and limit are required")
	}
	if _, err := time.Parse("2006-01-02", *from); err != nil {
		log.Fatalf("from: %v", err)
	}
	if _, err := time.Parse("2006-01-02", *to); err != nil {
		log.Fatalf("to: %v", err)
	}
	dir := strings.ToUpper(strings.TrimSpace(*direction))
	if dir != "" && dir != "ASCENDING" && dir != "DESCENDING" {
		log.Fatal("direction must be ASCENDING or DESCENDING")
	}
	pols, err := parsePols(*polFlag)
	if err != nil {
		log.Fatal(err)
	}
	selected, err := parseSelected(*scenesFlag)
	if err != nil {
		log.Fatalf("scenes: %v", err)
	}
	if !*listOnly {
		if err := loadEarthdataEnv(); err != nil {
			log.Fatalf("load Earthdata .env: %v", err)
		}
	}

	client := earthdataClient(*listOnly)
	progress.Printf("search: ASF")
	scenes, err := search(context.Background(), client, *bbox, seedPt, *from, *to, dir, *orbit, *limit, selected)
	if err != nil {
		log.Fatal(err)
	}
	progress.Printf("search: %d scenes", len(scenes))
	if len(scenes) == 0 {
		return
	}

	autoBBoxStr := ""
	scenes, autoBBoxStr = analyzeAndFilterScenes(scenes, seedPt, *orbit, *autoOrbit, *seedMargin, *minOverlap)

	if *saveBBox != "" && autoBBoxStr != "" {
		if err := os.WriteFile(*saveBBox, []byte(autoBBoxStr+"\n"), 0o644); err != nil {
			log.Fatalf("save-bbox: %v", err)
		}
		fmt.Fprintf(os.Stderr, "==> Saved BBOX to %s\n", *saveBBox)
	}

	if *listOnly {
		printCatalog(os.Stdout, scenes, autoBBoxStr)
		return
	}

	var toDownload []scene
	for _, s := range scenes {
		if s.accepted {
			toDownload = append(toDownload, s)
		}
	}
	if len(toDownload) == 0 {
		log.Fatal("no scenes accepted for download after footprint filtering")
	}
	scenes = toDownload

	rawDir := filepath.Join(*out, ".raw")
	for _, d := range []string{*out, rawDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			log.Fatal(err)
		}
	}

	type outcome struct {
		idx   int
		paths []string
		err   error
	}
	jobs := make(chan int, len(scenes))
	results := make(chan outcome, len(scenes))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var authFailureOnce sync.Once
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				if ctx.Err() != nil {
					results <- outcome{idx: idx, err: context.Canceled}
					continue
				}
				s := scenes[idx]
				paths, err := fetchScene(ctx, client, s, *out, rawDir, pols, progress)
				if err != nil {
					if errors.Is(err, errEarthdataUnauthorized) {
						authFailureOnce.Do(func() {
							cancel()
							progress.Printf("fail %s: %v", shortName(s.name), err)
							progress.Printf("stopping remaining downloads after Earthdata authentication failure")
						})
					} else if !errors.Is(err, context.Canceled) {
						progress.Printf("fail %s: %v", shortName(s.name), err)
					}
				}
				results <- outcome{idx: idx, paths: paths, err: err}
			}
		}()
	}
	for i := range scenes {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	close(results)
	_ = os.Remove(rawDir)

	ordered := make([][]string, len(scenes))
	failed := 0
	for r := range results {
		if r.err != nil {
			failed++
		} else {
			ordered[r.idx] = r.paths
		}
	}
	for _, paths := range ordered {
		for _, p := range paths {
			fmt.Println(p)
		}
	}
	progress.Printf("done: %d; failed: %d", len(scenes)-failed, failed)
	if failed > 0 {
		os.Exit(1)
	}
}

func search(ctx context.Context, c *http.Client, bbox string, seedPt *Point, from, to, dir, orbit string, limit int, sel map[string]bool) ([]scene, error) {
	var wkt string
	if seedPt != nil {
		wkt = fmt.Sprintf("point(%f %f)", seedPt.X, seedPt.Y)
	} else {
		v := strings.Split(bbox, ",")
		if len(v) != 4 {
			return nil, fmt.Errorf("bbox must be west,south,east,north")
		}
		var p [4]float64
		for i := range v {
			x, err := strconv.ParseFloat(strings.TrimSpace(v[i]), 64)
			if err != nil {
				return nil, fmt.Errorf("invalid bbox: %w", err)
			}
			p[i] = x
		}
		if p[0] < -180 || p[2] > 180 || p[1] < -90 || p[3] > 90 || p[0] >= p[2] || p[1] >= p[3] {
			return nil, fmt.Errorf("invalid bbox coordinates")
		}
		wkt = fmt.Sprintf("polygon((%f %f,%f %f,%f %f,%f %f,%f %f))", p[0], p[1], p[2], p[1], p[2], p[3], p[0], p[3], p[0], p[1])
	}
	u, _ := url.Parse(api)
	q := u.Query()
	q.Set("platform", "Sentinel-1A,Sentinel-1B,Sentinel-1C")
	q.Set("beamMode", "IW")
	q.Set("processingLevel", "GRD_HD")
	q.Set("polarization", "VV,VV+VH")
	q.Set("intersectsWith", wkt)
	q.Set("start", from)
	q.Set("end", to)
	q.Set("output", "geojson")
	q.Set("maxResults", strconv.Itoa(limit))
	if dir != "" {
		q.Set("flightDirection", dir)
	}
	if orbit != "" {
		q.Set("relativeOrbit", orbit)
	}
	u.RawQuery = q.Encode()

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("ASF search %s: %s", resp.Status, b)
	}
	var data result
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	seen, scenes := map[string]bool{}, []scene{}
	for _, f := range data.Features {
		pr := f.Properties
		if pr.URL == "" {
			continue
		}
		name := pr.SceneName
		if name == "" {
			name = strings.TrimSuffix(filepath.Base(pr.FileName), filepath.Ext(pr.FileName))
		}
		name = safe(name)
		if seen[name] || (len(sel) > 0 && !sel[name]) {
			continue
		}
		seen[name] = true
		ts, _ := time.Parse(time.RFC3339, pr.StartTime)
		date := pr.StartTime
		if len(date) >= 10 {
			date = date[:10]
		}
		sz, _ := pr.Bytes.Int64()

		var poly Polygon
		if f.Geometry.Type == "Polygon" && len(f.Geometry.Coordinates) > 0 {
			ring := f.Geometry.Coordinates[0]
			if len(ring) > 3 && ring[0] == ring[len(ring)-1] {
				ring = ring[:len(ring)-1]
			}
			for _, pt := range ring {
				poly = append(poly, Point{X: pt[0], Y: pt[1]})
			}
			poly = ensureCCW(poly)
		}

		scenes = append(scenes, scene{
			name: name, url: pr.URL, date: date, dir: pr.FlightDirection,
			orbit: pr.PathNumber.String(), pols: pr.Polarization, ts: ts, bytes: sz,
			poly: poly, accepted: true,
		})
	}
	sort.Slice(scenes, func(i, j int) bool { return scenes[i].ts.Before(scenes[j].ts) })
	return scenes, nil
}

func analyzeAndFilterScenes(scenes []scene, seedPt *Point, requestedOrbit string, autoOrbit bool, seedMarginKm, minOverlap float64) ([]scene, string) {
	if len(scenes) == 0 {
		return scenes, ""
	}

	targetOrbit := requestedOrbit
	if seedPt != nil && targetOrbit == "" && autoOrbit {
		trackCounts := map[string]int{}
		for _, s := range scenes {
			k := s.orbit + "_" + s.dir
			trackCounts[k]++
		}
		bestTrack, maxCount := "", 0
		for k, count := range trackCounts {
			if count > maxCount {
				maxCount = count
				bestTrack = k
			}
		}
		if bestTrack != "" {
			parts := strings.Split(bestTrack, "_")
			targetOrbit = parts[0]
			fmt.Fprintf(os.Stderr, "==> Auto-selected dominant track: orbit %s (%s) with %d scene(s)\n", parts[0], parts[1], maxCount)
		}
	}

	var refLon, refLat float64
	if seedPt != nil {
		refLon, refLat = seedPt.X, seedPt.Y
	} else if len(scenes[0].poly) > 0 {
		refLon, refLat = scenes[0].poly[0].X, scenes[0].poly[0].Y
	}
	zone, south := utmZone(refLon, refLat)

	type cand struct {
		idx    int
		utm    Polygon
		margin float64
	}
	var candidates []cand

	for i := range scenes {
		scenes[i].accepted = true

		if targetOrbit != "" && scenes[i].orbit != targetOrbit {
			scenes[i].accepted = false
			scenes[i].rejectReason = fmt.Sprintf("not on target orbit %s", targetOrbit)
			continue
		}

		if len(scenes[i].poly) < 3 {
			scenes[i].accepted = false
			scenes[i].rejectReason = "missing geometry polygon"
			continue
		}

		utmPoly := projectPolygonToUTM(scenes[i].poly, zone, south)

		if seedPt != nil {
			if !pointInPolygon(*seedPt, scenes[i].poly) {
				scenes[i].accepted = false
				scenes[i].rejectReason = "does not cover seed point"
				continue
			}

			margin := pointToPolygonDistKm(*seedPt, scenes[i].poly, zone, south)
			if margin < seedMarginKm {
				scenes[i].accepted = false
				scenes[i].rejectReason = fmt.Sprintf("seed too close to edge (%.1fkm < %.1fkm)", margin, seedMarginKm)
				continue
			}

			candidates = append(candidates, cand{idx: i, utm: utmPoly, margin: margin})
		} else {
			candidates = append(candidates, cand{idx: i, utm: utmPoly, margin: polygonArea(utmPoly)})
		}
	}

	if len(candidates) == 0 {
		return scenes, ""
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].margin > candidates[j].margin
	})

	pCore := candidates[0].utm
	for _, c := range candidates[1:] {
		clipped := clipPolygon(pCore, c.utm)
		clippedArea := polygonArea(clipped)
		coreArea := polygonArea(pCore)
		var ratio float64
		if coreArea > 0 {
			ratio = clippedArea / coreArea
		}

		if ratio < minOverlap {
			scenes[c.idx].accepted = false
			scenes[c.idx].rejectReason = fmt.Sprintf("low mutual overlap (%.1f%% < %.1f%%)", ratio*100, minOverlap*100)
		} else {
			pCore = clipped
		}
	}

	var seedUTM *Point
	if seedPt != nil {
		ux, uy := toUTM(seedPt.X, seedPt.Y, zone, south)
		seedUTM = &Point{X: ux, Y: uy}
	}

	minX, minY, maxX, maxY, ok := maxInscribedBox(pCore, seedUTM, 250, 250)
	if !ok {
		return scenes, ""
	}

	minLon, minLat := fromUTM(minX, minY, zone, south)
	maxLon, maxLat := fromUTM(maxX, maxY, zone, south)
	autoBBox := fmt.Sprintf("%.4f,%.4f,%.4f,%.4f", minLon, minLat, maxLon, maxLat)

	commonAreaKm2 := polygonArea(pCore) / 1e6
	boxAreaKm2 := ((maxX - minX) * (maxY - minY)) / 1e6

	accCount := 0
	for _, s := range scenes {
		if s.accepted {
			accCount++
		}
	}

	fmt.Fprintf(os.Stderr, "==> Footprint analysis: %d/%d accepted scene(s)\n", accCount, len(scenes))
	fmt.Fprintf(os.Stderr, "    Common overlap: %.1f km², Max inscribed box: %.1f km²\n", commonAreaKm2, boxAreaKm2)
	fmt.Fprintf(os.Stderr, "    BBOX: %s\n", autoBBox)

	return scenes, autoBBox
}

func printCatalog(w io.Writer, scenes []scene, autoBBox string) {
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "DATE\tORBIT\tDIR\tPOLS\tMB\tSTATUS\tSCENE")
	tracks := map[string]int{}
	for _, s := range scenes {
		status := "ACCEPTED"
		if !s.accepted {
			status = "REJECTED (" + s.rejectReason + ")"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%.0f\t%s\t%s\n",
			s.date, s.orbit, s.dir, s.pols, float64(s.bytes)/(1024*1024), status, s.name)
		if s.accepted {
			tracks[fmt.Sprintf("-orbit %s -direction %s", s.orbit, s.dir)]++
		}
	}
	_ = tw.Flush()
	if autoBBox != "" {
		fmt.Fprintf(os.Stderr, "\n==> Maximum aligned BBOX: %s\n", autoBBox)
	}
	for k, count := range tracks {
		fmt.Fprintf(os.Stderr, "track: %-32s (%d accepted scene(s))\n", k, count)
	}
}

func fetchScene(ctx context.Context, c *http.Client, s scene, outDir, rawDir string, pols []string, progress *progressDisplay) ([]string, error) {
	targets := make(map[string]string, len(pols))
	var missing []string
	var allPaths []string
	for _, pol := range pols {
		p := filepath.Join(outDir, fmt.Sprintf("%s_%s.tiff", s.name, strings.ToUpper(pol)))
		targets[pol] = p
		allPaths = append(allPaths, p)
		if !exists(p) {
			missing = append(missing, pol)
		}
	}
	if len(missing) == 0 {
		progress.Printf("skip %s", shortName(s.name))
		return allPaths, nil
	}

	zipPath := filepath.Join(rawDir, s.name+".zip")
	if !progress.interactive {
		progress.Printf("get %s", shortName(s.name))
	}
	if err := download(ctx, c, s.url, zipPath, s.name, progress); err != nil {
		return nil, err
	}
	defer os.Remove(zipPath)
	progress.Printf("extract %s (%s)", shortName(s.name), strings.ToUpper(strings.Join(missing, ",")))
	if err := extractBands(zipPath, targets, missing); err != nil {
		return nil, err
	}
	return allPaths, nil
}

func extractBands(zipPath string, targets map[string]string, pols []string) error {
	z, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer z.Close()
	for _, pol := range pols {
		tag := "-" + pol + "-"
		var band *zip.File
		for _, f := range z.File {
			n := strings.ToLower(filepath.Base(f.Name))
			if (strings.HasSuffix(n, ".tif") || strings.HasSuffix(n, ".tiff")) &&
				strings.Contains(n, tag) && !strings.Contains(f.Name, "preview") {
				if band != nil {
					return fmt.Errorf("multiple %s TIFFs in archive", strings.ToUpper(pol))
				}
				band = f
			}
		}
		if band == nil {
			return fmt.Errorf("no %s measurement TIFF in archive", strings.ToUpper(pol))
		}
		r, err := band.Open()
		if err != nil {
			return err
		}
		dst := targets[pol]
		tmp := dst + ".part"
		w, err := os.Create(tmp)
		if err != nil {
			_ = r.Close()
			return err
		}
		_, cp := io.Copy(w, r)
		_ = r.Close()
		cl := w.Close()
		if cp != nil || cl != nil {
			_ = os.Remove(tmp)
			if cp != nil {
				return cp
			}
			return cl
		}
		if err := os.Rename(tmp, dst); err != nil {
			return err
		}
	}
	return nil
}

func download(ctx context.Context, c *http.Client, src, dst, name string, progress *progressDisplay) error {
	tmp := dst + ".part"
	_ = os.Remove(tmp)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, src, nil)
	if err != nil {
		return err
	}
	if trusted(req.URL.Hostname()) {
		auth(req)
	}
	progress.start(name, -1)
	resp, err := c.Do(req)
	if err != nil {
		progress.finish(name)
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		progress.finish(name)
		if resp.StatusCode == http.StatusUnauthorized {
			return fmt.Errorf("%w: download %s: %s", errEarthdataUnauthorized, resp.Status, strings.TrimSpace(string(b)))
		}
		return fmt.Errorf("download %s: %s", resp.Status, b)
	}
	f, err := os.Create(tmp)
	if err != nil {
		progress.finish(name)
		return err
	}
	progress.setTotal(name, resp.ContentLength)
	_, cp := io.Copy(f, progressReader{reader: resp.Body, progress: progress, name: name})
	progress.finish(name)
	cl := f.Close()
	if cp != nil || cl != nil {
		_ = os.Remove(tmp)
		if cp != nil {
			return cp
		}
		return cl
	}
	return os.Rename(tmp, dst)
}

func parsePols(s string) ([]string, error) {
	var pols []string
	seen := map[string]bool{}
	for _, p := range strings.Split(strings.ToLower(s), ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if p != "vv" && p != "vh" {
			return nil, fmt.Errorf("invalid polarization %q (expected vv, vh, or vv,vh)", p)
		}
		if !seen[p] {
			seen[p] = true
			pols = append(pols, p)
		}
	}
	if len(pols) == 0 {
		return nil, fmt.Errorf("at least one polarization required")
	}
	return pols, nil
}

func parseSelected(s string) (map[string]bool, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	sel := map[string]bool{}
	if s == "-" {
		sc := bufio.NewScanner(os.Stdin)
		for sc.Scan() {
			fields := strings.Fields(sc.Text())
			if len(fields) > 0 && fields[0] != "DATE" && !strings.HasPrefix(fields[0], "#") {
				sel[safe(fields[len(fields)-1])] = true
			}
		}
		return sel, sc.Err()
	}
	for _, tok := range strings.Split(s, ",") {
		if tok = strings.TrimSpace(tok); tok != "" {
			sel[safe(tok)] = true
		}
	}
	return sel, nil
}

func earthdataClient(listOnly bool) *http.Client {
	if !listOnly && os.Getenv("EARTHDATA_TOKEN") == "" && (os.Getenv("EARTHDATA_USERNAME") == "" || os.Getenv("EARTHDATA_PASSWORD") == "") {
		log.Fatal("set EARTHDATA_TOKEN or EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
	}
	jar, _ := cookiejar.New(nil)
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConnsPerHost = 16
	return &http.Client{
		Transport: tr,
		Jar:       jar,
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			if trusted(r.URL.Hostname()) {
				auth(r)
			}
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}
}

// loadEarthdataEnv reads a local .env without overriding values already set by
// the caller. It checks the current directory and, for the repository binary,
// the repository root next to bin/asf.
func loadEarthdataEnv() error {
	candidates := []string{filepath.Join(".", ".env")}
	if executable, err := os.Executable(); err == nil {
		executableDir := filepath.Dir(executable)
		if filepath.Base(executableDir) == "bin" {
			candidates = append(candidates, filepath.Join(filepath.Dir(executableDir), ".env"))
		} else {
			candidates = append(candidates, filepath.Join(executableDir, ".env"))
		}
	}

	seen := make(map[string]bool)
	for _, candidate := range candidates {
		path, err := filepath.Abs(candidate)
		if err != nil {
			return err
		}
		if seen[path] {
			continue
		}
		seen[path] = true
		if err := loadEarthdataEnvFile(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func loadEarthdataEnvFile(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for lineNumber := 1; scanner.Scan(); lineNumber++ {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("%s:%d: expected KEY=value", path, lineNumber)
		}
		key = strings.TrimSpace(key)
		if key != "EARTHDATA_TOKEN" && key != "EARTHDATA_USERNAME" && key != "EARTHDATA_PASSWORD" {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		value, err := parseEnvValue(strings.TrimSpace(value))
		if err != nil {
			return fmt.Errorf("%s:%d: %w", path, lineNumber, err)
		}
		if err := os.Setenv(key, value); err != nil {
			return fmt.Errorf("%s:%d: set %s: %w", path, lineNumber, key, err)
		}
	}
	return scanner.Err()
}

func parseEnvValue(value string) (string, error) {
	if value == "" {
		return "", nil
	}
	if value[0] == '\'' || value[0] == '"' {
		quote := value[0]
		for i := 1; i < len(value); i++ {
			if quote == '"' && value[i] == '\\' {
				i++
				continue
			}
			if value[i] != quote {
				continue
			}
			rest := strings.TrimSpace(value[i+1:])
			if rest != "" && !strings.HasPrefix(rest, "#") {
				return "", fmt.Errorf("unexpected text after quoted value")
			}
			if quote == '"' {
				unquoted, err := strconv.Unquote(value[:i+1])
				if err != nil {
					return "", fmt.Errorf("invalid quoted value: %w", err)
				}
				return unquoted, nil
			}
			return value[1:i], nil
		}
		return "", fmt.Errorf("unterminated quoted value")
	}
	if comment := strings.Index(value, " #"); comment >= 0 {
		value = strings.TrimSpace(value[:comment])
	}
	return value, nil
}

func auth(r *http.Request) {
	if t := os.Getenv("EARTHDATA_TOKEN"); t != "" {
		r.Header.Set("Authorization", "Bearer "+t)
	} else if u := os.Getenv("EARTHDATA_USERNAME"); u != "" {
		r.SetBasicAuth(u, os.Getenv("EARTHDATA_PASSWORD"))
	}
}

func trusted(h string) bool {
	h = strings.ToLower(h)
	return h == "urs.earthdata.nasa.gov" || strings.HasSuffix(h, ".asf.alaska.edu")
}

func exists(p string) bool { s, e := os.Stat(p); return e == nil && s.Size() > 0 }

func safe(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || strings.ContainsRune("-_.", r) {
			return r
		}
		return '_'
	}, filepath.Base(s))
}

type downloadProgress struct {
	name            string
	received, total int64
}

type progressDisplay struct {
	mu          sync.Mutex
	out         io.Writer
	interactive bool
	active      map[string]*downloadProgress
	drawn       int
	lastDraw    time.Time
}

func newProgressDisplay(out io.Writer) *progressDisplay {
	info, err := os.Stderr.Stat()
	return &progressDisplay{
		out:         out,
		interactive: err == nil && info.Mode()&os.ModeCharDevice != 0,
		active:      make(map[string]*downloadProgress),
	}
}

func (p *progressDisplay) Printf(format string, args ...any) {
	line := strings.NewReplacer("\r", " ", "\n", " ").Replace(fmt.Sprintf(format, args...))
	p.mu.Lock()
	defer p.mu.Unlock()
	p.clearLocked()
	fmt.Fprintln(p.out, line)
	p.drawLocked()
}

func (p *progressDisplay) start(name string, total int64) {
	if !p.interactive {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.active[name] = &downloadProgress{name: name, total: total}
	p.drawLocked()
	p.lastDraw = time.Now()
}

func (p *progressDisplay) setTotal(name string, total int64) {
	if !p.interactive {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if item := p.active[name]; item != nil {
		item.total = total
		p.drawLocked()
		p.lastDraw = time.Now()
	}
}

func (p *progressDisplay) add(name string, n int64) {
	if !p.interactive || n <= 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if item := p.active[name]; item != nil {
		item.received += n
		if time.Since(p.lastDraw) >= 200*time.Millisecond {
			p.drawLocked()
			p.lastDraw = time.Now()
		}
	}
}

func (p *progressDisplay) finish(name string) {
	if !p.interactive {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.active, name)
	p.drawLocked()
	p.lastDraw = time.Now()
}

func (p *progressDisplay) clearLocked() {
	if p.interactive && p.drawn > 0 {
		fmt.Fprintf(p.out, "\033[%dA\r\033[J", p.drawn)
		p.drawn = 0
	}
}

func (p *progressDisplay) drawLocked() {
	if !p.interactive {
		return
	}
	p.clearLocked()
	names := make([]string, 0, len(p.active))
	for name := range p.active {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fmt.Fprintln(p.out, p.progressLine(p.active[name]))
	}
	p.drawn = len(names)
}

func (p *progressDisplay) progressLine(item *downloadProgress) string {
	name := shortName(item.name)
	if item.total <= 0 {
		return fmt.Sprintf("get %s %s MiB", name, mib(item.received))
	}
	pct := min(100, int(float64(item.received)*100/float64(item.total)))
	filled := pct * 20 / 100
	bar := strings.Repeat("#", filled) + strings.Repeat(".", 20-filled)
	return fmt.Sprintf("get %s [%s] %3d%% %s/%s MiB", name, bar, pct, mib(item.received), mib(item.total))
}

func mib(n int64) string { return fmt.Sprintf("%.0f", float64(n)/(1024*1024)) }

func shortName(name string) string {
	if len(name) <= 22 {
		return name
	}
	return name[:9] + "..." + name[len(name)-10:]
}

type progressReader struct {
	reader   io.Reader
	progress *progressDisplay
	name     string
}

func (r progressReader) Read(b []byte) (int, error) {
	n, err := r.reader.Read(b)
	if n > 0 {
		r.progress.add(r.name, int64(n))
	}
	return n, err
}
