// geom.go - Shared 2D computational geometry, UTM projection, and maximum inscribed rectangle allocation.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
)

type Point struct {
	X, Y float64 // Longitude/Latitude or UTM Easting/Northing
}

type Polygon []Point

// parseSeedPoint parses "lon,lat" or "lat,lon" string.
// If one coordinate has |val| > 90, it is automatically identified as longitude.
func parseSeedPoint(s string) (*Point, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "point(")
	s = strings.TrimPrefix(s, "POINT(")
	s = strings.TrimSuffix(s, ")")
	s = strings.ReplaceAll(s, " ", ",")

	var parts []string
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			parts = append(parts, p)
		}
	}
	if len(parts) != 2 {
		return nil, fmt.Errorf("seed coordinate must be 'lon,lat' (got %q)", s)
	}

	v1, err1 := strconv.ParseFloat(parts[0], 64)
	v2, err2 := strconv.ParseFloat(parts[1], 64)
	if err1 != nil || err2 != nil {
		return nil, fmt.Errorf("invalid coordinates in %q", s)
	}

	var lon, lat float64
	if math.Abs(v1) > 90 && math.Abs(v2) <= 90 {
		lon, lat = v1, v2
	} else if math.Abs(v2) > 90 && math.Abs(v1) <= 90 {
		lon, lat = v2, v1
	} else {
		// Standard GIS/WKT convention: lon, lat (X, Y)
		lon, lat = v1, v2
	}

	if lon < -180 || lon > 180 || lat < -90 || lat > 90 {
		return nil, fmt.Errorf("coordinates out of range: lon=%.4f, lat=%.4f", lon, lat)
	}

	return &Point{X: lon, Y: lat}, nil
}

func utmZone(lon, lat float64) (int, bool) {
	zone := min(60, max(1, int(math.Floor((lon+180)/6))+1))
	return zone, lat < 0
}

func utmGrid(b [4]float64, res float64) (string, float64, float64, float64, float64) {
	lonC, latC := 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])
	zone, south := utmZone(lonC, latC)
	epsg := 32600 + zone
	if south {
		epsg = 32700 + zone
	}
	x1, y1 := toUTM(b[0], b[1], zone, south)
	x2, y2 := toUTM(b[2], b[3], zone, south)
	return fmt.Sprintf("EPSG:%d", epsg),
		math.Floor(math.Min(x1, x2)/res) * res, math.Floor(math.Min(y1, y2)/res) * res,
		math.Ceil(math.Max(x1, x2)/res) * res, math.Ceil(math.Max(y1, y2)/res) * res
}

func toUTM(lon, lat float64, zone int, south bool) (float64, float64) {
	const a, f, k0 = 6378137.0, 1.0 / 298.257223563, 0.9996
	e2 := f * (2 - f)
	ep2 := e2 / (1 - e2)
	latR, lonR := lat*math.Pi/180, lon*math.Pi/180
	lon0R := float64((zone-1)*6-177) * math.Pi / 180
	s, c, t := math.Sin(latR), math.Cos(latR), math.Tan(latR)
	n := a / math.Sqrt(1-e2*s*s)
	T, C, A := t*t, ep2*c*c, c*(lonR-lon0R)
	e4, e6 := e2*e2, e2*e2*e2
	m := a * ((1-e2/4-3*e4/64-5*e6/256)*latR - (3*e2/8+3*e4/32+45*e6/1024)*math.Sin(2*latR) +
		(15*e4/256+45*e6/1024)*math.Sin(4*latR) - (35*e6/3072)*math.Sin(6*latR))
	x := k0*n*(A+(1-T+C)*math.Pow(A, 3)/6+(5-18*T+T*T+72*C-58*ep2)*math.Pow(A, 5)/120) + 500000
	y := k0 * (m + n*t*(A*A/2+(5-T+9*C+4*C*C)*math.Pow(A, 4)/24+(61-58*T+T*T+600*C-330*ep2)*math.Pow(A, 6)/720))
	if south {
		y += 10000000
	}
	return x, y
}

func fromUTM(x, y float64, zone int, south bool) (float64, float64) {
	const a, f, k0 = 6378137.0, 1.0 / 298.257223563, 0.9996
	e2 := f * (2 - f)
	e1 := (1.0 - math.Sqrt(1.0-e2)) / (1.0 + math.Sqrt(1.0-e2))
	ep2 := e2 / (1.0 - e2)
	x -= 500000.0
	if south {
		y -= 10000000.0
	}
	m := y / k0
	mu := m / (a * (1.0 - e2/4.0 - 3.0*e2*e2/64.0 - 5.0*e2*e2*e2/256.0))
	phi1 := mu + (3.0*e1/2.0-27.0*math.Pow(e1, 3)/32.0)*math.Sin(2.0*mu) +
		(21.0*e1*e1/16.0-55.0*math.Pow(e1, 4)/32.0)*math.Sin(4.0*mu) +
		(151.0*math.Pow(e1, 3)/96.0)*math.Sin(6.0*mu) +
		(1097.0*math.Pow(e1, 4)/512.0)*math.Sin(8.0*mu)

	c1 := ep2 * math.Cos(phi1) * math.Cos(phi1)
	t1 := math.Tan(phi1) * math.Tan(phi1)
	n1 := a / math.Sqrt(1.0-e2*math.Sin(phi1)*math.Sin(phi1))
	r1 := a * (1.0 - e2) / math.Pow(1.0-e2*math.Sin(phi1)*math.Sin(phi1), 1.5)
	d := x / (n1 * k0)

	lat := phi1 - (n1*math.Tan(phi1)/r1)*(d*d/2.0-(5.0+3.0*t1+10.0*c1-4.0*c1*c1-9.0*ep2)*math.Pow(d, 4)/24.0+
		(61.0+90.0*t1+298.0*c1+45.0*t1*t1-252.0*ep2-3.0*c1*c1)*math.Pow(d, 6)/720.0)
	lon0 := float64((zone-1)*6 - 177)
	lon := lon0*math.Pi/180.0 + (d-(1.0+2.0*t1+c1)*math.Pow(d, 3)/6.0+
		(5.0-2.0*c1+28.0*t1-3.0*c1*c1+8.0*ep2+24.0*t1*t1)*math.Pow(d, 5)/120.0)/math.Cos(phi1)

	return lon * 180.0 / math.Pi, lat * 180.0 / math.Pi
}

func polygonArea(p Polygon) float64 {
	n := len(p)
	if n < 3 {
		return 0
	}
	var a float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += p[i].X*p[j].Y - p[j].X*p[i].Y
	}
	return math.Abs(a) * 0.5
}

func ensureCCW(p Polygon) Polygon {
	n := len(p)
	if n < 3 {
		return p
	}
	var signedArea float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		signedArea += p[i].X*p[j].Y - p[j].X*p[i].Y
	}
	if signedArea < 0 {
		rev := make(Polygon, n)
		for i := 0; i < n; i++ {
			rev[i] = p[n-1-i]
		}
		return rev
	}
	return p
}

func pointInPolygon(pt Point, poly Polygon) bool {
	n := len(poly)
	if n < 3 {
		return false
	}
	inside := false
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		xi, yi := poly[i].X, poly[i].Y
		xj, yj := poly[j].X, poly[j].Y
		if ((yi > pt.Y) != (yj > pt.Y)) && (pt.X < (xj-xi)*(pt.Y-yi)/(yj-yi)+xi) {
			inside = !inside
		}
	}
	return inside
}

func pointToSegmentDist(p, a, b Point) float64 {
	dx := b.X - a.X
	dy := b.Y - a.Y
	l2 := dx*dx + dy*dy
	if l2 == 0 {
		return math.Hypot(p.X-a.X, p.Y-a.Y)
	}
	t := ((p.X-a.X)*dx + (p.Y-a.Y)*dy) / l2
	t = math.Max(0, math.Min(1, t))
	proj := Point{X: a.X + t*dx, Y: a.Y + t*dy}
	return math.Hypot(p.X-proj.X, p.Y-proj.Y)
}

func pointToPolygonDistKm(p Point, poly Polygon, zone int, south bool) float64 {
	if len(poly) < 2 {
		return 0
	}
	// Project to UTM to measure Euclidean distance in meters
	pUTM_X, pUTM_Y := toUTM(p.X, p.Y, zone, south)
	pUTM := Point{X: pUTM_X, Y: pUTM_Y}

	minMeters := math.MaxFloat64
	for i := 0; i < len(poly); i++ {
		j := (i + 1) % len(poly)
		aX, aY := toUTM(poly[i].X, poly[i].Y, zone, south)
		bX, bY := toUTM(poly[j].X, poly[j].Y, zone, south)
		d := pointToSegmentDist(pUTM, Point{X: aX, Y: aY}, Point{X: bX, Y: bY})
		if d < minMeters {
			minMeters = d
		}
	}
	return minMeters / 1000.0
}

// Sutherland-Hodgman polygon clipping (clip must be convex and CCW)
func clipPolygon(subject, clip Polygon) Polygon {
	output := subject
	clip = ensureCCW(clip)
	for i := 0; i < len(clip); i++ {
		cp1 := clip[i]
		cp2 := clip[(i+1)%len(clip)]
		input := output
		output = Polygon{}
		if len(input) == 0 {
			break
		}
		s := input[len(input)-1]
		for _, e := range input {
			if isInsideEdge(e, cp1, cp2) {
				if isInsideEdge(s, cp1, cp2) {
					output = append(output, e)
				} else {
					output = append(output, intersectionPoint(s, e, cp1, cp2))
					output = append(output, e)
				}
			} else if isInsideEdge(s, cp1, cp2) {
				output = append(output, intersectionPoint(s, e, cp1, cp2))
			}
			s = e
		}
	}
	return output
}

func isInsideEdge(p, cp1, cp2 Point) bool {
	return (cp2.X-cp1.X)*(p.Y-cp1.Y)-(cp2.Y-cp1.Y)*(p.X-cp1.X) >= -1e-9
}

func intersectionPoint(p1, p2, cp1, cp2 Point) Point {
	a1 := p2.Y - p1.Y
	b1 := p1.X - p2.X
	c1 := a1*p1.X + b1*p1.Y

	a2 := cp2.Y - cp1.Y
	b2 := cp1.X - cp2.X
	c2 := a2*cp1.X + b2*cp1.Y

	det := a1*b2 - a2*b1
	if math.Abs(det) < 1e-12 {
		return p1
	}
	return Point{
		X: (b2*c1 - b1*c2) / det,
		Y: (a1*c2 - a2*c1) / det,
	}
}

func projectPolygonToUTM(poly Polygon, zone int, south bool) Polygon {
	res := make(Polygon, len(poly))
	for i, pt := range poly {
		x, y := toUTM(pt.X, pt.Y, zone, south)
		res[i] = Point{X: x, Y: y}
	}
	return ensureCCW(res)
}

// maxInscribedBox finds the largest axis-aligned rectangle inside a UTM polygon.
// If seed is non-nil, ensures the box contains the seed point.
func maxInscribedBox(poly Polygon, seed *Point, nx, ny int) (minX, minY, maxX, maxY float64, ok bool) {
	if len(poly) < 3 || nx < 10 || ny < 10 {
		return 0, 0, 0, 0, false
	}
	pMinX, pMaxX := poly[0].X, poly[0].X
	pMinY, pMaxY := poly[0].Y, poly[0].Y
	for _, pt := range poly[1:] {
		pMinX = math.Min(pMinX, pt.X)
		pMaxX = math.Max(pMaxX, pt.X)
		pMinY = math.Min(pMinY, pt.Y)
		pMaxY = math.Max(pMaxY, pt.Y)
	}

	dx := (pMaxX - pMinX) / float64(nx)
	dy := (pMaxY - pMinY) / float64(ny)

	var seedCol, seedRow int = -1, -1
	if seed != nil {
		if !pointInPolygon(*seed, poly) {
			return 0, 0, 0, 0, false
		}
		seedCol = int((seed.X - pMinX) / dx)
		seedRow = int((seed.Y - pMinY) / dy)
		seedCol = max(0, min(nx-1, seedCol))
		seedRow = max(0, min(ny-1, seedRow))
	}

	grid := make([][]bool, ny)
	for r := 0; r < ny; r++ {
		grid[r] = make([]bool, nx)
		cy := pMinY + (float64(r)+0.5)*dy
		for c := 0; c < nx; c++ {
			cx := pMinX + (float64(c)+0.5)*dx
			if pointInPolygon(Point{X: cx, Y: cy}, poly) {
				grid[r][c] = true
			}
		}
	}

	heights := make([]int, nx)
	bestArea := 0
	var bestC1, bestR1, bestC2, bestR2 int

	for r := 0; r < ny; r++ {
		for c := 0; c < nx; c++ {
			if grid[r][c] {
				heights[c]++
			} else {
				heights[c] = 0
			}
		}

		stack := []int{}
		for c := 0; c <= nx; c++ {
			var h int
			if c < nx {
				h = heights[c]
			}
			for len(stack) > 0 && h < heights[stack[len(stack)-1]] {
				topIdx := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				curH := heights[topIdx]
				left := 0
				if len(stack) > 0 {
					left = stack[len(stack)-1] + 1
				}
				right := c - 1
				curW := right - left + 1
				r1 := r - curH + 1
				r2 := r

				if seed != nil {
					if seedCol < left || seedCol > right || seedRow < r1 || seedRow > r2 {
						continue
					}
				}

				area := curW * curH
				if area > bestArea {
					bestArea = area
					bestC1, bestR1, bestC2, bestR2 = left, r1, right, r2
				}
			}
			stack = append(stack, c)
		}
	}

	if bestArea == 0 {
		return 0, 0, 0, 0, false
	}

	resMinX := pMinX + float64(bestC1)*dx
	resMaxX := pMinX + float64(bestC2+1)*dx
	resMinY := pMinY + float64(bestR1)*dy
	resMaxY := pMinY + float64(bestR2+1)*dy

	// Refine boundary outwards towards polygon edges
	resMinX, resMinY, resMaxX, resMaxY = refineBox(poly, resMinX, resMinY, resMaxX, resMaxY, dx*0.2, dy*0.2)

	return resMinX, resMinY, resMaxX, resMaxY, true
}

func boxInPoly(poly Polygon, minX, minY, maxX, maxY float64) bool {
	return pointInPolygon(Point{X: minX, Y: minY}, poly) &&
		pointInPolygon(Point{X: maxX, Y: minY}, poly) &&
		pointInPolygon(Point{X: maxX, Y: maxY}, poly) &&
		pointInPolygon(Point{X: minX, Y: maxY}, poly)
}

func refineBox(poly Polygon, minX, minY, maxX, maxY, stepX, stepY float64) (float64, float64, float64, float64) {
	for i := 0; i < 50; i++ {
		expanded := false
		if boxInPoly(poly, minX-stepX, minY, maxX, maxY) {
			minX -= stepX
			expanded = true
		}
		if boxInPoly(poly, minX, minY, maxX+stepX, maxY) {
			maxX += stepX
			expanded = true
		}
		if boxInPoly(poly, minX, minY-stepY, maxX, maxY) {
			minY -= stepY
			expanded = true
		}
		if boxInPoly(poly, minX, minY, maxX, maxY+stepY) {
			maxY += stepY
			expanded = true
		}
		if !expanded {
			break
		}
	}
	return minX, minY, maxX, maxY
}

// extractTiffFootprint reads corner GCPs from a GeoTIFF using gdalinfo.
func extractTiffFootprint(path string) (Polygon, error) {
	cmd := exec.Command("gdalinfo", "-json", path)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("gdalinfo %s: %w", path, err)
	}

	var info struct {
		GCPs struct {
			GCPList []struct {
				Pixel float64 `json:"pixel"`
				Line  float64 `json:"line"`
				X     float64 `json:"x"`
				Y     float64 `json:"y"`
			} `json:"gcpList"`
		} `json:"gcps"`
	}

	if err := json.Unmarshal(out, &info); err != nil {
		return nil, fmt.Errorf("parse gdalinfo json %s: %w", path, err)
	}
	gcps := info.GCPs.GCPList
	if len(gcps) == 0 {
		return nil, fmt.Errorf("no GCPs found in %s", path)
	}

	minP, maxP := math.MaxFloat64, -math.MaxFloat64
	minL, maxL := math.MaxFloat64, -math.MaxFloat64
	for _, g := range gcps {
		minP = math.Min(minP, g.Pixel)
		maxP = math.Max(maxP, g.Pixel)
		minL = math.Min(minL, g.Line)
		maxL = math.Max(maxL, g.Line)
	}

	var tl, tr, br, bl Point
	foundTL, foundTR, foundBR, foundBL := false, false, false, false

	for _, g := range gcps {
		if g.Pixel == minP && g.Line == minL {
			tl = Point{X: g.X, Y: g.Y}
			foundTL = true
		}
		if g.Pixel == maxP && g.Line == minL {
			tr = Point{X: g.X, Y: g.Y}
			foundTR = true
		}
		if g.Pixel == maxP && g.Line == maxL {
			br = Point{X: g.X, Y: g.Y}
			foundBR = true
		}
		if g.Pixel == minP && g.Line == maxL {
			bl = Point{X: g.X, Y: g.Y}
			foundBL = true
		}
	}

	if !foundTL || !foundTR || !foundBR || !foundBL {
		return nil, fmt.Errorf("could not resolve all 4 corner GCPs in %s", path)
	}

	return ensureCCW(Polygon{tl, tr, br, bl}), nil
}
