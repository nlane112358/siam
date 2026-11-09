// sarimg.go - Co-register Sentinel-1 TIFFs, apply p2-p98 stretch, slice chips,
// emit PRE/POST pairs for siam, and convert between TIFF, PGM, PPM, and PNG.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

const patchSize = 32 // Matches siam::Config::PATCH_S

type config struct {
	out       string
	formats   map[string]bool
	bbox      *[4]float64
	res       float64
	minCov    float64
	chip      int
	bits      int
	stretch   string
	cmap      string
	thresh    float64
	upsample  string
	stride    int
	baseImg   string
	dropImg   string
	incImg    string
	changeImg string
	pairs     bool
	overwrite bool
	workers   int
}

type raster struct {
	w, h, ch, maxv int
	gray           []float64 // [0,1] for single-band
	rgb            []uint8   // 3*w*h for P3/P6 PPM
}

func main() {
	log.SetFlags(0)
	var cfg config
	toFlag := flag.String("to", "auto", "output format(s): pgm, png, or pgm,png (auto: pgm for TIFF when -pairs, else png)")
	bboxFlag := flag.String("bbox", "", "west,south,east,north: warp TIFFs to a shared UTM grid")
	flag.StringVar(&cfg.out, "o", "asf-data", "output directory, file path, or '-' for stdout")
	flag.Float64Var(&cfg.res, "res", 10.0, "target pixel resolution in meters when -bbox is set")
	flag.Float64Var(&cfg.minCov, "min-cov", 0.0, "minimum nonzero pixel fraction [0..1] (default 0.90 if -bbox/-chip)")
	flag.IntVar(&cfg.chip, "chip", 0, "slice into NxN pixel chips (e.g. 1024; 0 = full image)")
	flag.IntVar(&cfg.bits, "bits", 16, "PGM bit depth: 8 or 16")
	flag.StringVar(&cfg.stretch, "stretch", "auto", "nonzero p2-p98 stretch: auto (on for TIFF, off for PGM), on, off")
	flag.StringVar(&cfg.cmap, "cmap", "gray", "PNG colormap for single-band inputs: gray or inferno")
	flag.Float64Var(&cfg.thresh, "thresh", 0.0, "minimum normalized value [0..1] for colormap/overlay")
	flag.StringVar(&cfg.upsample, "upsample", "", "reference file or WxH to bilinearly upsample tile maps")
	flag.IntVar(&cfg.stride, "stride", 8, "siam stride for -upsample")
	flag.StringVar(&cfg.baseImg, "base", "", "base SAR image for directional/change overlay PNG")
	flag.StringVar(&cfg.changeImg, "change", "", "siam -o change.pgm to overlay on -base")
	flag.StringVar(&cfg.dropImg, "drop", "", "siam -D drop.pgm (cyan) to overlay on -base")
	flag.StringVar(&cfg.incImg, "inc", "", "siam -I increase.pgm (red-orange) to overlay on -base")
	flag.BoolVar(&cfg.pairs, "pairs", false, "emit chronological PRE POST PGM pairs to stdout")
	flag.BoolVar(&cfg.overwrite, "overwrite", false, "regenerate existing output files")
	flag.IntVar(&cfg.workers, "workers", runtime.NumCPU(), "concurrent workers")
	flag.Parse()

	if cfg.bits != 8 && cfg.bits != 16 {
		log.Fatal("bits must be 8 or 16")
	}
	if cfg.bbox, _ = parseBBox(*bboxFlag); *bboxFlag != "" && cfg.bbox == nil {
		log.Fatal("bbox must be valid west,south,east,north")
	}
	if cfg.minCov == 0.0 && (cfg.bbox != nil || cfg.chip > 0) {
		cfg.minCov = 0.90
	}

	if cfg.baseImg != "" {
		if err := renderOverlay(cfg); err != nil {
			log.Fatal(err)
		}
		return
	}

	inputs, err := collectInputs(flag.Args())
	if err != nil || len(inputs) == 0 {
		log.Fatalf("no valid input files found (%v)", err)
	}
	cfg.formats = resolveFormats(*toFlag, inputs, cfg.pairs)

	// Single stream or explicit single-file output
	if len(inputs) == 1 && (inputs[0] == "-" || strings.HasSuffix(strings.ToLower(cfg.out), ".png") ||
		strings.HasSuffix(strings.ToLower(cfg.out), ".pgm") || cfg.out == "-") {
		if _, err := processOne(inputs[0], cfg); err != nil {
			log.Fatal(err)
		}
		return
	}

	type res struct {
		pgms []string
		err  error
		src  string
	}
	jobs := make(chan string, len(inputs))
	results := make(chan res, len(inputs))
	var wg sync.WaitGroup
	for i := 0; i < max(1, cfg.workers); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for src := range jobs {
				pgms, err := processOne(src, cfg)
				results <- res{pgms: pgms, err: err, src: src}
			}
		}()
	}
	for _, in := range inputs {
		jobs <- in
	}
	close(jobs)
	wg.Wait()
	close(results)

	var allPGMs []string
	failed := 0
	for r := range results {
		if r.err != nil {
			failed++
			fmt.Fprintf(os.Stderr, "fail %s: %v\n", filepath.Base(r.src), r.err)
		} else {
			allPGMs = append(allPGMs, r.pgms...)
		}
	}
	if cfg.pairs {
		emitPairs(os.Stdout, allPGMs)
	}
	fmt.Fprintf(os.Stderr, "done: %d; failed: %d\n", len(inputs)-failed, failed)
	if failed > 0 {
		os.Exit(1)
	}
}

func processOne(src string, cfg config) ([]string, error) {
	ext := strings.ToLower(filepath.Ext(src))
	isTif := ext == ".tif" || ext == ".tiff"
	doStretch := cfg.stretch == "on" || (cfg.stretch == "auto" && isTif)
	stem := strings.TrimSuffix(filepath.Base(src), filepath.Ext(src))

	// Check if outputs already exist before loading/warping
	if src != "-" && !cfg.overwrite && cfg.chip == 0 {
		pgmOut, pngOut := outPaths(cfg, stem)
		pgmOK := !cfg.formats["pgm"] || exists(pgmOut)
		pngOK := !cfg.formats["png"] || exists(pngOut)
		if pgmOK && pngOK {
			fmt.Fprintf(os.Stderr, "skip %s\n", filepath.Base(src))
			if cfg.formats["pgm"] {
				return []string{pgmOut}, nil
			}
			return nil, nil
		}
	}

	ras, p2, p98, cov, err := loadRaster(src, isTif, doStretch, cfg)
	if err != nil {
		return nil, err
	}
	if cov < cfg.minCov {
		return nil, fmt.Errorf("coverage %.1f%% < %.1f%%", cov*100, cfg.minCov*100)
	}
	if cfg.upsample != "" {
		tw, th, err := targetDims(cfg.upsample)
		if err != nil {
			return nil, err
		}
		ras = upsampleBilinear(ras, tw, th, cfg.stride)
	}

	if cfg.chip > 0 {
		return writeChips(ras, stem, cfg)
	}

	pgmOut, pngOut := outPaths(cfg, stem)
	var pgms []string
	if cfg.formats["pgm"] {
		if err := savePGM(pgmOut, ras, cfg.bits); err != nil {
			return nil, err
		}
		pgms = append(pgms, pgmOut)
		if !cfg.pairs && pgmOut != "-" {
			fmt.Println(pgmOut)
		}
	}
	if cfg.formats["png"] {
		if err := savePNG(pngOut, ras, cfg.cmap, cfg.thresh); err != nil {
			return nil, err
		}
		if !cfg.pairs && !cfg.formats["pgm"] && pngOut != "-" {
			fmt.Println(pngOut)
		}
	}
	if doStretch {
		fmt.Fprintf(os.Stderr, "ok %s (%dx%d p2=%.1f p98=%.1f)\n", stem, ras.w, ras.h, p2, p98)
	} else if src != "-" {
		fmt.Fprintf(os.Stderr, "ok %s (%dx%d)\n", stem, ras.w, ras.h)
	}
	return pgms, nil
}

func loadRaster(src string, isTif, stretch bool, cfg config) (*raster, float64, float64, float64, error) {
	if !isTif {
		var r io.Reader = os.Stdin
		if src != "-" {
			f, err := os.Open(src)
			if err != nil {
				return nil, 0, 0, 0, err
			}
			defer f.Close()
			r = f
		}
		return readPNM(r, stretch)
	}

	tmpF, err := os.CreateTemp("", "sarimg-*.pgm")
	if err != nil {
		return nil, 0, 0, 0, err
	}
	tmpPgm := tmpF.Name()
	_ = tmpF.Close()
	defer os.Remove(tmpPgm)

	var cmd *exec.Cmd
	if cfg.bbox != nil {
		epsg, minX, minY, maxX, maxY := utmGrid(*cfg.bbox, cfg.res)
		cmd = exec.Command("gdalwarp", "-q", "-overwrite", "-tps",
			"-t_srs", epsg,
			"-te", fmt.Sprintf("%.3f", minX), fmt.Sprintf("%.3f", minY), fmt.Sprintf("%.3f", maxX), fmt.Sprintf("%.3f", maxY),
			"-tr", fmt.Sprintf("%.3f", cfg.res), fmt.Sprintf("%.3f", cfg.res),
			"-r", "bilinear", "-srcnodata", "0", "-dstnodata", "0",
			"-ot", "UInt16", "-of", "PNM", "-co", "MAXVAL=65535", src, tmpPgm)
	} else {
		cmd = exec.Command("gdal_translate", "-q", "-of", "PNM", "-ot", "UInt16", "-co", "MAXVAL=65535", src, tmpPgm)
	}
	if b, err := cmd.CombinedOutput(); err != nil {
		return nil, 0, 0, 0, fmt.Errorf("gdal: %w: %s", err, b)
	}
	f, err := os.Open(tmpPgm)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer f.Close()
	return readPNM(f, stretch)
}

func readPNM(r io.Reader, stretch bool) (*raster, float64, float64, float64, error) {
	br := bufio.NewReaderSize(r, 256*1024)
	magic, err := nextToken(br)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	wTok, _ := nextToken(br)
	hTok, _ := nextToken(br)
	mTok, err := nextToken(br)
	w, _ := strconv.Atoi(wTok)
	h, _ := strconv.Atoi(hTok)
	maxv, _ := strconv.Atoi(mTok)
	if err != nil || w <= 0 || h <= 0 || maxv <= 0 || maxv > 65535 {
		return nil, 0, 0, 0, fmt.Errorf("invalid PNM header (%s %dx%d %d)", magic, w, h, maxv)
	}

	n := w * h
	ras := &raster{w: w, h: h, maxv: maxv}
	if magic == "P6" || magic == "P3" {
		ras.ch = 3
		ras.rgb = make([]uint8, n*3)
		if magic == "P6" {
			if _, err := io.ReadFull(br, ras.rgb); err != nil {
				return nil, 0, 0, 0, err
			}
		} else {
			for i := 0; i < n*3; i++ {
				t, err := nextToken(br)
				if err != nil {
					return nil, 0, 0, 0, err
				}
				v, _ := strconv.Atoi(t)
				ras.rgb[i] = uint8(min(255, max(0, v*255/maxv)))
			}
		}
		return ras, 0, float64(maxv), 1.0, nil
	}
	if magic != "P5" && magic != "P2" {
		return nil, 0, 0, 0, fmt.Errorf("unsupported PNM magic %q", magic)
	}

	ras.ch = 1
	ras.gray = make([]float64, n)
	raw := make([]uint16, n)
	if magic == "P5" {
		if maxv < 256 {
			buf := make([]byte, n)
			if _, err := io.ReadFull(br, buf); err != nil {
				return nil, 0, 0, 0, err
			}
			for i, b := range buf {
				raw[i] = uint16(b)
			}
		} else {
			buf := make([]byte, n*2)
			if _, err := io.ReadFull(br, buf); err != nil {
				return nil, 0, 0, 0, err
			}
			for i := 0; i < n; i++ {
				raw[i] = binary.BigEndian.Uint16(buf[2*i:])
			}
		}
	} else {
		for i := 0; i < n; i++ {
			t, err := nextToken(br)
			if err != nil {
				return nil, 0, 0, 0, err
			}
			v, _ := strconv.Atoi(t)
			raw[i] = uint16(min(maxv, max(0, v)))
		}
	}

	var hist [65536]uint64
	var nz uint64
	for _, v := range raw {
		if v > 0 {
			hist[v]++
			nz++
		}
	}
	cov := float64(nz) / float64(n)
	if !stretch {
		inv := 1.0 / float64(maxv)
		for i, v := range raw {
			ras.gray[i] = float64(v) * inv
		}
		return ras, 0, float64(maxv), cov, nil
	}
	if nz == 0 {
		return nil, 0, 0, 0, fmt.Errorf("no nonzero pixels")
	}
	p2 := percentile(hist[:], nz, 2.0)
	p98 := percentile(hist[:], nz, 98.0)
	if p98 <= p2 {
		return nil, 0, 0, cov, fmt.Errorf("degenerate percentile range p2=%.2f p98=%.2f", p2, p98)
	}
	invR := 1.0 / (p98 - p2)
	for i, v := range raw {
		if v > 0 {
			ras.gray[i] = math.Max(0.0, math.Min(1.0, (float64(v)-p2)*invR))
		}
	}
	return ras, p2, p98, cov, nil
}

func writeChips(ras *raster, stem string, cfg config) ([]string, error) {
	cs := cfg.chip
	if ras.w < cs || ras.h < cs {
		return nil, fmt.Errorf("image %dx%d smaller than chip %d", ras.w, ras.h, cs)
	}
	var pgms []string
	for gy := 0; gy < ras.h/cs; gy++ {
		for gx := 0; gx < ras.w/cs; gx++ {
			sub := &raster{w: cs, h: cs, ch: 1, maxv: ras.maxv, gray: make([]float64, cs*cs)}
			nz := 0
			for y := 0; y < cs; y++ {
				row := ras.gray[(gy*cs+y)*ras.w+gx*cs:]
				for x := 0; x < cs; x++ {
					v := row[x]
					sub.gray[y*cs+x] = v
					if v > 0 {
						nz++
					}
				}
			}
			if float64(nz)/float64(cs*cs) < cfg.minCov {
				continue
			}
			cStem := fmt.Sprintf("%s_r%02d_c%02d", stem, gy, gx)
			pgmOut, pngOut := outPaths(cfg, cStem)
			if cfg.formats["pgm"] {
				if err := savePGM(pgmOut, sub, cfg.bits); err != nil {
					return nil, err
				}
				pgms = append(pgms, pgmOut)
			}
			if cfg.formats["png"] {
				if err := savePNG(pngOut, sub, cfg.cmap, cfg.thresh); err != nil {
					return nil, err
				}
			}
		}
	}
	return pgms, nil
}

func savePGM(dst string, ras *raster, bits int) error {
	if ras.ch != 1 {
		return fmt.Errorf("cannot write 3-channel PPM as PGM")
	}
	maxv := 255
	if bits == 16 {
		maxv = 65535
	}
	write := func(w io.Writer) error {
		bw := bufio.NewWriterSize(w, 256*1024)
		if _, err := fmt.Fprintf(bw, "P5\n%d %d\n%d\n", ras.w, ras.h, maxv); err != nil {
			return err
		}
		if bits == 8 {
			for _, v := range ras.gray {
				if err := bw.WriteByte(uint8(math.Max(0, math.Min(255, v*255.0+0.5)))); err != nil {
					return err
				}
			}
		} else {
			var b [2]byte
			for _, v := range ras.gray {
				binary.BigEndian.PutUint16(b[:], uint16(math.Max(0, math.Min(65535, v*65535.0+0.5))))
				if _, err := bw.Write(b[:]); err != nil {
					return err
				}
			}
		}
		return bw.Flush()
	}
	return atomicWrite(dst, write)
}

func savePNG(dst string, ras *raster, cmap string, thresh float64) error {
	var img image.Image
	if ras.ch == 3 {
		rgba := image.NewRGBA(image.Rect(0, 0, ras.w, ras.h))
		for i := 0; i < ras.w*ras.h; i++ {
			rgba.Pix[4*i+0] = ras.rgb[3*i+0]
			rgba.Pix[4*i+1] = ras.rgb[3*i+1]
			rgba.Pix[4*i+2] = ras.rgb[3*i+2]
			rgba.Pix[4*i+3] = 255
		}
		img = rgba
	} else if strings.ToLower(cmap) == "inferno" {
		rgba := image.NewRGBA(image.Rect(0, 0, ras.w, ras.h))
		for i, v := range ras.gray {
			if v < thresh {
				v = 0
			}
			c := inferno(v)
			rgba.Pix[4*i+0], rgba.Pix[4*i+1], rgba.Pix[4*i+2], rgba.Pix[4*i+3] = c.R, c.G, c.B, 255
		}
		img = rgba
	} else {
		g := image.NewGray(image.Rect(0, 0, ras.w, ras.h))
		for i, v := range ras.gray {
			if v < thresh {
				v = 0
			}
			g.Pix[i] = uint8(math.Max(0, math.Min(255, v*255.0+0.5)))
		}
		img = g
	}
	return atomicWrite(dst, func(w io.Writer) error { return png.Encode(w, img) })
}

func renderOverlay(cfg config) error {
	ext := strings.ToLower(filepath.Ext(cfg.baseImg))
	isTif := ext == ".tif" || ext == ".tiff"
	base, _, _, _, err := loadRaster(cfg.baseImg, isTif, isTif || cfg.stretch == "on", cfg)
	if err != nil {
		return err
	}
	loadMap := func(p string) ([]float64, error) {
		if p == "" {
			return nil, nil
		}
		r, _, _, _, err := loadRaster(p, false, false, cfg)
		if err != nil {
			return nil, err
		}
		if r.w != base.w || r.h != base.h {
			stride := cfg.stride
			if r.w > 1 && (base.w-patchSize)%(r.w-1) == 0 {
				stride = (base.w - patchSize) / (r.w - 1)
			}
			r = upsampleBilinear(r, base.w, base.h, stride)
		}
		return r.gray, nil
	}
	chMap, err := loadMap(cfg.changeImg)
	if err != nil {
		return err
	}
	drMap, err := loadMap(cfg.dropImg)
	if err != nil {
		return err
	}
	inMap, err := loadMap(cfg.incImg)
	if err != nil {
		return err
	}

	rgba := image.NewRGBA(image.Rect(0, 0, base.w, base.h))
	for i := 0; i < base.w*base.h; i++ {
		bg := base.gray[i] * 255.0
		or, og, ob, a := 0.0, 0.0, 0.0, 0.0
		if chMap != nil && chMap[i] >= cfg.thresh && chMap[i] > 0 {
			c := inferno(chMap[i])
			or, og, ob, a = float64(c.R), float64(c.G), float64(c.B), 0.75*math.Min(1.0, chMap[i]*1.8)
		}
		if drMap != nil && drMap[i] >= cfg.thresh && drMap[i] > 0 {
			or, og, ob, a = 0, 210, 255, 0.75*math.Min(1.0, drMap[i]*2.0)
		}
		if inMap != nil && inMap[i] >= cfg.thresh && inMap[i] > 0 {
			or, og, ob, a = 255, 80, 20, 0.75*math.Min(1.0, inMap[i]*2.0)
		}
		rgba.Pix[4*i+0] = uint8(bg*(1-a) + or*a + 0.5)
		rgba.Pix[4*i+1] = uint8(bg*(1-a) + og*a + 0.5)
		rgba.Pix[4*i+2] = uint8(bg*(1-a) + ob*a + 0.5)
		rgba.Pix[4*i+3] = 255
	}
	dst := cfg.out
	if !strings.HasSuffix(strings.ToLower(dst), ".png") && dst != "-" {
		dst = filepath.Join(dst, "overlay.png")
	}
	return atomicWrite(dst, func(w io.Writer) error { return png.Encode(w, rgba) })
}

func emitPairs(w io.Writer, pgms []string) {
	sort.Slice(pgms, func(i, j int) bool {
		return sceneSortKey(pgms[i]) < sceneSortKey(pgms[j])
	})
	groups := map[string][]string{}
	var keys []string
	for _, p := range pgms {
		k := pairGroupKey(p)
		if _, ok := groups[k]; !ok {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], p)
	}
	sort.Strings(keys)
	for _, k := range keys {
		list := groups[k]
		for i := 0; i+1 < len(list); i++ {
			fmt.Fprintf(w, "%s %s\n", list[i], list[i+1])
		}
	}
}

func sceneSortKey(path string) string {
	base := filepath.Base(path)
	for _, tok := range strings.Split(base, "_") {
		if len(tok) == 15 && tok[8] == 'T' {
			return tok + "_" + base
		}
	}
	return base
}

func pairGroupKey(path string) string {
	stem := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	parts := strings.Split(stem, "_")
	if len(parts) >= 3 && strings.HasPrefix(parts[len(parts)-2], "r") && strings.HasPrefix(parts[len(parts)-1], "c") {
		return strings.Join(parts[len(parts)-3:], "_")
	}
	return parts[len(parts)-1]
}

func upsampleBilinear(src *raster, outW, outH, stride int) *raster {
	if src.w == outW && src.h == outH {
		return src
	}
	gw, gh := src.w, src.h
	hp := 0.5 * float64(patchSize-1)
	invS := 1.0 / float64(stride)
	dst := &raster{w: outW, h: outH, ch: src.ch, maxv: 255}
	if src.ch == 1 {
		dst.gray = make([]float64, outW*outH)
	} else {
		dst.rgb = make([]uint8, outW*outH*3)
	}
	for y := 0; y < outH; y++ {
		fy := math.Max(0, math.Min(float64(gh-1), (float64(y)-hp)*invS))
		y0, y1 := int(fy), min(int(fy)+1, gh-1)
		wy := fy - float64(y0)
		for x := 0; x < outW; x++ {
			fx := math.Max(0, math.Min(float64(gw-1), (float64(x)-hp)*invS))
			x0, x1 := int(fx), min(int(fx)+1, gw-1)
			wx := fx - float64(x0)
			if src.ch == 1 {
				top := src.gray[y0*gw+x0] + wx*(src.gray[y0*gw+x1]-src.gray[y0*gw+x0])
				bot := src.gray[y1*gw+x0] + wx*(src.gray[y1*gw+x1]-src.gray[y1*gw+x0])
				dst.gray[y*outW+x] = top + wy*(bot-top)
			} else {
				for c := 0; c < 3; c++ {
					r00, r01 := float64(src.rgb[(y0*gw+x0)*3+c]), float64(src.rgb[(y0*gw+x1)*3+c])
					r10, r11 := float64(src.rgb[(y1*gw+x0)*3+c]), float64(src.rgb[(y1*gw+x1)*3+c])
					val := (r00 + wx*(r01-r00)) + wy*((r10+wx*(r11-r10))-(r00+wx*(r01-r00)))
					dst.rgb[(y*outW+x)*3+c] = uint8(math.Max(0, math.Min(255, val+0.5)))
				}
			}
		}
	}
	return dst
}

func percentile(hist []uint64, count uint64, p float64) float64 {
	pos := float64(count-1) * p / 100.0
	lo, hi := uint64(math.Floor(pos)), uint64(math.Ceil(pos))
	var cum uint64
	var loBin, hiBin int
	gotLo := false
	for bin, c := range hist {
		if c == 0 {
			continue
		}
		cum += c
		if !gotLo && cum >= lo+1 {
			loBin, gotLo = bin, true
		}
		if cum >= hi+1 {
			hiBin = bin
			break
		}
	}
	return float64(loBin) + float64(hiBin-loBin)*(pos-float64(lo))
}

func utmGrid(b [4]float64, res float64) (string, float64, float64, float64, float64) {
	lonC, latC := 0.5*(b[0]+b[2]), 0.5*(b[1]+b[3])
	zone := min(60, max(1, int(math.Floor((lonC+180)/6))+1))
	epsg := 32600 + zone
	if latC < 0 {
		epsg = 32700 + zone
	}
	x1, y1 := toUTM(b[0], b[1], zone, latC < 0)
	x2, y2 := toUTM(b[2], b[3], zone, latC < 0)
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

func inferno(t float64) color.RGBA {
	stops := [][3]float64{{0, 0, 4}, {87, 16, 110}, {188, 55, 84}, {249, 142, 9}, {252, 255, 164}}
	s := math.Max(0, math.Min(1, t)) * float64(len(stops)-1)
	i := min(len(stops)-2, int(s))
	f := s - float64(i)
	return color.RGBA{
		R: uint8(stops[i][0] + f*(stops[i+1][0]-stops[i][0]) + 0.5),
		G: uint8(stops[i][1] + f*(stops[i+1][1]-stops[i][1]) + 0.5),
		B: uint8(stops[i][2] + f*(stops[i+1][2]-stops[i][2]) + 0.5),
		A: 255,
	}
}

func collectInputs(args []string) ([]string, error) {
	if len(args) == 0 {
		info, err := os.Stdin.Stat()
		if err == nil && info.Mode()&os.ModeCharDevice == 0 {
			var list []string
			sc := bufio.NewScanner(os.Stdin)
			for sc.Scan() {
				if p := strings.TrimSpace(sc.Text()); p != "" {
					list = append(list, p)
				}
			}
			return list, sc.Err()
		}
		args = []string{"tif"}
	}
	var out []string
	for _, a := range args {
		if a == "-" {
			out = append(out, "-")
			continue
		}
		st, err := os.Stat(a)
		if err != nil {
			return nil, err
		}
		if !st.IsDir() {
			out = append(out, a)
			continue
		}
		entries, err := os.ReadDir(a)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			ext := strings.ToLower(filepath.Ext(e.Name()))
			if !e.IsDir() && (ext == ".tif" || ext == ".tiff" || ext == ".pgm" || ext == ".ppm") {
				out = append(out, filepath.Join(a, e.Name()))
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

func resolveFormats(toFlag string, inputs []string, pairs bool) map[string]bool {
	m := map[string]bool{}
	if toFlag != "auto" {
		for _, f := range strings.Split(strings.ToLower(toFlag), ",") {
			m[strings.TrimSpace(f)] = true
		}
		return m
	}
	if pairs {
		m["pgm"] = true
		return m
	}
	for _, in := range inputs {
		ext := strings.ToLower(filepath.Ext(in))
		if ext == ".tif" || ext == ".tiff" {
			m["pgm"], m["png"] = true, true
			return m
		}
	}
	m["png"] = true
	return m
}

func outPaths(cfg config, stem string) (string, string) {
	if cfg.out == "-" {
		return "-", "-"
	}
	low := strings.ToLower(cfg.out)
	if strings.HasSuffix(low, ".pgm") || strings.HasSuffix(low, ".png") {
		base := strings.TrimSuffix(cfg.out, filepath.Ext(cfg.out))
		return base + ".pgm", base + ".png"
	}
	if cfg.formats["pgm"] && cfg.formats["png"] {
		return filepath.Join(cfg.out, "pgm", stem+".pgm"), filepath.Join(cfg.out, "png", stem+".png")
	}
	return filepath.Join(cfg.out, stem+".pgm"), filepath.Join(cfg.out, stem+".png")
}

func targetDims(spec string) (int, int, error) {
	if p := strings.Split(strings.ToLower(spec), "x"); len(p) == 2 {
		w, e1 := strconv.Atoi(p[0])
		h, e2 := strconv.Atoi(p[1])
		if e1 == nil && e2 == nil && w > 0 && h > 0 {
			return w, h, nil
		}
	}
	f, err := os.Open(spec)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	br := bufio.NewReader(f)
	_, _ = nextToken(br)
	wt, _ := nextToken(br)
	ht, err := nextToken(br)
	w, _ := strconv.Atoi(wt)
	h, _ := strconv.Atoi(ht)
	return w, h, err
}

func parseBBox(s string) (*[4]float64, error) {
	if s == "" {
		return nil, nil
	}
	v := strings.Split(s, ",")
	if len(v) != 4 {
		return nil, fmt.Errorf("invalid bbox")
	}
	var p [4]float64
	for i := range v {
		x, err := strconv.ParseFloat(strings.TrimSpace(v[i]), 64)
		if err != nil {
			return nil, err
		}
		p[i] = x
	}
	if p[0] < -180 || p[2] > 180 || p[1] < -90 || p[3] > 90 || p[0] >= p[2] || p[1] >= p[3] {
		return nil, fmt.Errorf("invalid bbox bounds")
	}
	return &p, nil
}

func atomicWrite(dst string, fn func(io.Writer) error) error {
	if dst == "-" {
		return fn(os.Stdout)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	err = fn(f)
	cl := f.Close()
	if err != nil || cl != nil {
		_ = os.Remove(tmp)
		if err != nil {
			return err
		}
		return cl
	}
	return os.Rename(tmp, dst)
}

func nextToken(r *bufio.Reader) (string, error) {
	for {
		b, err := r.ReadByte()
		if err != nil {
			return "", err
		}
		if b == '#' {
			_, _ = r.ReadBytes('\n')
			continue
		}
		if b > ' ' {
			tok := []byte{b}
			for {
				nb, err := r.ReadByte()
				if err != nil {
					if err == io.EOF {
						return string(tok), nil
					}
					return "", err
				}
				if nb <= ' ' {
					return string(tok), nil
				}
				tok = append(tok, nb)
			}
		}
	}
}

func exists(p string) bool { s, e := os.Stat(p); return e == nil && s.Size() > 0 }
