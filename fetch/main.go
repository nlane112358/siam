// Search ASF for Sentinel-1 IW GRD scenes, extract the VV GeoTIFF, and make a PGM.
package main

import (
	"archive/zip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const api = "https://api.daac.asf.alaska.edu/services/search/param"

type result struct {
	Features []struct {
		Properties struct {
			URL       string `json:"url"`
			FileName  string `json:"fileName"`
			SceneName string `json:"sceneName"`
		} `json:"properties"`
	} `json:"features"`
}
type scene struct{ name, url string }

func main() {
	log.SetFlags(0)
	progress := newProgressDisplay(os.Stderr)
	bbox := flag.String("bbox", "", "west,south,east,north (required)")
	from := flag.String("from", "", "start date YYYY-MM-DD (required)")
	to := flag.String("to", "", "end date YYYY-MM-DD (required)")
	out := flag.String("out", "asf-data", "output directory")
	direction := flag.String("direction", "", "ASCENDING or DESCENDING")
	orbit := flag.String("orbit", "", "relative orbit number")
	workers := flag.Int("workers", 3, "concurrent downloads")
	limit := flag.Int("limit", 100, "maximum search results")
	flag.Parse()

	if *bbox == "" || *from == "" || *to == "" || *workers < 1 || *limit < 1 {
		flag.Usage()
		log.Fatal("bbox, dates, workers and limit are required and must be valid")
	}
	if _, err := time.Parse("2006-01-02", *from); err != nil {
		log.Fatalf("from: %v", err)
	}
	if _, err := time.Parse("2006-01-02", *to); err != nil {
		log.Fatalf("to: %v", err)
	}
	if strings.ToUpper(*direction) != "" && strings.ToUpper(*direction) != "ASCENDING" && strings.ToUpper(*direction) != "DESCENDING" {
		log.Fatal("direction must be ASCENDING or DESCENDING")
	}

	client := earthdataClient()
	progress.Printf("search: ASF")
	scenes, err := search(context.Background(), client, *bbox, *from, *to, *direction, *orbit, *limit)
	if err != nil {
		log.Fatal(err)
	}
	progress.Printf("search: %d scenes", len(scenes))
	if len(scenes) == 0 {
		progress.Printf("done: 0; failed: 0")
		return
	}
	for _, d := range []string{"tif", "pgm", "raw"} {
		if err := os.MkdirAll(filepath.Join(*out, d), 0o755); err != nil {
			log.Fatal(err)
		}
	}

	jobs, errs := make(chan scene), make(chan error, len(scenes))
	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for s := range jobs {
				if err := process(context.Background(), client, s, *out, progress); err != nil {
					progress.Printf("fail %s: %v", shortName(s.name), err)
					errs <- err
				}
			}
		}()
	}
	for _, s := range scenes {
		jobs <- s
	}
	close(jobs)
	wg.Wait()
	close(errs)
	failed := 0
	for range errs {
		failed++
	}
	progress.Printf("done: %d; failed: %d", len(scenes)-failed, failed)
	if failed > 0 {
		os.Exit(1)
	}
}

func search(ctx context.Context, c *http.Client, bbox, from, to, direction, orbit string, limit int) ([]scene, error) {
	v := strings.Split(bbox, ",")
	if len(v) != 4 {
		return nil, fmt.Errorf("bbox must be west,south,east,north")
	}
	p := [4]float64{}
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
	wkt := fmt.Sprintf("polygon((%f %f,%f %f,%f %f,%f %f,%f %f))", p[0], p[1], p[2], p[1], p[2], p[3], p[0], p[3], p[0], p[1])
	u, _ := url.Parse(api)
	q := u.Query()
	q.Set("platform", "Sentinel-1A,Sentinel-1B")
	q.Set("beamMode", "IW")
	q.Set("processingLevel", "GRD_HD")
	q.Set("polarization", "VV,VV+VH")
	q.Set("intersectsWith", wkt)
	q.Set("start", from)
	q.Set("end", to)
	q.Set("output", "geojson")
	q.Set("maxResults", strconv.Itoa(limit))
	if direction != "" {
		q.Set("flightDirection", strings.ToUpper(direction))
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
		if name == "" {
			name = strings.TrimSuffix(filepath.Base(pr.URL), filepath.Ext(pr.URL))
		}
		name = safe(name)
		if !seen[name] {
			seen[name] = true
			scenes = append(scenes, scene{name, pr.URL})
		}
	}
	return scenes, nil
}

func process(ctx context.Context, c *http.Client, s scene, out string, progress *progressDisplay) error {
	tifPath := filepath.Join(out, "tif", s.name+"_VV.tiff")
	pgmPath := filepath.Join(out, "pgm", s.name+"_VV.pgm")
	if exists(tifPath) && exists(pgmPath) {
		progress.Printf("skip %s", shortName(s.name))
		return nil
	}
	if !exists(tifPath) {
		zipPath := filepath.Join(out, "raw", s.name+".zip")
		if !progress.interactive {
			progress.Printf("get %s", shortName(s.name))
		}
		if err := download(ctx, c, s.url, zipPath, s.name, progress); err != nil {
			return err
		}
		if !progress.interactive {
			progress.Printf("got %s", shortName(s.name))
		}
		progress.Printf("extract %s", shortName(s.name))
		if err := extractVV(zipPath, tifPath); err != nil {
			return err
		}
		_ = os.Remove(zipPath)
	}
	if exists(pgmPath) {
		progress.Printf("skip convert %s (PGM exists)", shortName(s.name))
		return nil
	}
	tmp := pgmPath + ".part"
	progress.Printf("convert %s", shortName(s.name))
	cmd := exec.Command("gdal_translate", "-q", "-of", "PNM", "-ot", "UInt16", "-co", "MAXVAL=65535", tifPath, tmp)
	if b, err := cmd.CombinedOutput(); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("gdal_translate: %w: %s", err, b)
	}
	if err := os.Rename(tmp, pgmPath); err != nil {
		return err
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
	if cp != nil {
		_ = os.Remove(tmp)
		return cp
	}
	if cl != nil {
		_ = os.Remove(tmp)
		return cl
	}
	return os.Rename(tmp, dst)
}

func extractVV(src, dst string) error {
	z, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer z.Close()
	var band *zip.File
	for _, f := range z.File {
		n := strings.ToLower(filepath.Base(f.Name))
		if (strings.HasSuffix(n, ".tif") || strings.HasSuffix(n, ".tiff")) && strings.Contains(n, "-vv-") {
			if band != nil {
				return fmt.Errorf("multiple VV measurement TIFFs in archive")
			}
			band = f
		}
	}
	if band == nil {
		return fmt.Errorf("no VV measurement TIFF in SAFE archive")
	}
	r, err := band.Open()
	if err != nil {
		return err
	}
	defer r.Close()
	tmp := dst + ".part"
	w, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, cp := io.Copy(w, r)
	cl := w.Close()
	if cp != nil {
		_ = os.Remove(tmp)
		return cp
	}
	if cl != nil {
		_ = os.Remove(tmp)
		return cl
	}
	return os.Rename(tmp, dst)
}

func earthdataClient() *http.Client {
	if os.Getenv("EARTHDATA_TOKEN") == "" && (os.Getenv("EARTHDATA_USERNAME") == "" || os.Getenv("EARTHDATA_PASSWORD") == "") {
		log.Fatal("set EARTHDATA_TOKEN or EARTHDATA_USERNAME and EARTHDATA_PASSWORD")
	}
	jar, _ := cookiejar.New(nil)
	return &http.Client{Jar: jar, CheckRedirect: func(r *http.Request, via []*http.Request) error {
		if trusted(r.URL.Hostname()) {
			auth(r)
		}
		if len(via) >= 10 {
			return fmt.Errorf("too many redirects")
		}
		return nil
	}}
}
func auth(r *http.Request) {
	if t := os.Getenv("EARTHDATA_TOKEN"); t != "" {
		r.Header.Set("Authorization", "Bearer "+t)
	} else {
		r.SetBasicAuth(os.Getenv("EARTHDATA_USERNAME"), os.Getenv("EARTHDATA_PASSWORD"))
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
	name     string
	received int64
	total    int64
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
	line := fmt.Sprintf(format, args...)
	line = strings.ReplaceAll(line, "\r", " ")
	line = strings.ReplaceAll(line, "\n", " ")

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
	if !p.interactive || p.drawn == 0 {
		return
	}
	fmt.Fprintf(p.out, "\033[%dA\r\033[J", p.drawn)
	p.drawn = 0
}

func (p *progressDisplay) drawLocked() {
	if !p.interactive {
		return
	}
	if p.drawn > 0 {
		fmt.Fprintf(p.out, "\033[%dA\r\033[J", p.drawn)
	}
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
	pct := int(float64(item.received) * 100 / float64(item.total))
	if pct > 100 {
		pct = 100
	}
	const width = 20
	filled := pct * width / 100
	bar := strings.Repeat("#", filled) + strings.Repeat(".", width-filled)
	return fmt.Sprintf("get %s [%s] %3d%% %s/%s MiB", name, bar, pct, mib(item.received), mib(item.total))
}

func mib(n int64) string {
	return fmt.Sprintf("%.0f", float64(n)/(1024*1024))
}

func shortName(name string) string {
	const maxLength = 22
	if len(name) <= maxLength {
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
