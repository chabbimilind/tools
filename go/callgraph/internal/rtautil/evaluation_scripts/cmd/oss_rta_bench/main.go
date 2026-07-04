// oss_rta_bench runs the RTA (Rapid Type Analysis) flavor sweep against a
// single open-source Go project, mirroring the per-(flavor x workers) bench
// loop in rpcshield/graph/rtaeval.go:buildCGOnlySweep but without the
// monorepo-only WPA pipeline (Bazel go_path, Grail, IDL discovery, etc.).
//
// The binary is built once inside the monorepo and then invoked against
// cloned OSS source trees. packages.Config.Dir directs go/packages' driver
// to the dataset's own go.mod, so the dataset's transitive deps resolve
// against its module graph rather than the monorepo's.
//
// Driver: ../../run_oss_rta_sweep.sh loops over targets.txt.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/types"
	"log"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	rta "golang.org/x/tools/go/callgraph/internal/rtautil"
	"golang.org/x/tools/go/callgraph/prta_kumo_nonblocking"
	"golang.org/x/tools/go/callgraph/srta"
	"golang.org/x/tools/go/callgraph/srta_baseline"
	"golang.org/x/tools/go/callgraph/srta_kumo"
	"golang.org/x/tools/go/callgraph/srta_kumo_random"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// flavorEntry registers one of the 5 RTA flavor implementations.
// sequential flavors collapse the worker list to {1} per rtaeval.go's
// runSingleFlavor contract.
type flavorEntry struct {
	name       string
	sequential bool
	ctor       func() rta.RTA
}

var flavorRegistry = []flavorEntry{
	{"srta", true, func() rta.RTA { return srta.New() }},
	{"srta_baseline", true, func() rta.RTA { return srta_baseline.New() }},
	{"srta_kumo", true, func() rta.RTA { return srta_kumo.New() }},
	{"srta_kumo_random", true, func() rta.RTA { return srta_kumo_random.New() }},
	{"prta_kumo_nonblocking", false, func() rta.RTA { return prta_kumo_nonblocking.New() }},
}

func flavorByName(name string) (flavorEntry, bool) {
	for _, f := range flavorRegistry {
		if f.name == name {
			return f, true
		}
	}
	return flavorEntry{}, false
}

var stdout = log.New(os.Stdout, "", log.LstdFlags|log.Lmicroseconds)

// metrics accumulates all (flavor, workers) measurements across the run and
// is dumped at the end as a single JSON line, mirroring rpcshield's
// `wpa_metrics` log entry shape so the same parsers can ingest both.
var metrics = map[string]float64{}

// metricsStr holds string-valued metrics (e.g. "service") that cannot be
// represented as float64. Both maps are merged into the final JSON output.
var metricsStr = map[string]string{}

func recordMetric(key, flavor string, workers int, val float64) {
	metrics[fmt.Sprintf(key, flavor, workers)] = val
}

// emitMetricsJSON dumps the accumulated metrics map as one log line that
// matches rpcshield/cmd/rpcshield/wpamode/wpa.go:307:
//
//	Infow("wpa_metrics", zap.Any("metrics", values))
//
// Under zap's console encoder, that renders as a tab-separated line whose
// last field is {"metrics": {...}}. We emit the same JSON shape; downstream
// parsers key on the `oss_rta_metrics` message text and the JSON object.
//
// Floats are stringified to match the rpcshield convention (zap with the
// upstream metrics-as-strings wrapper produces "0.288" not 0.288).
func emitMetricsJSON() {
	asStrings := make(map[string]string, len(metrics)+len(metricsStr))
	maps.Copy(asStrings, metricsStr)
	for k, v := range metrics {
		asStrings[k] = strconv.FormatFloat(v, 'g', -1, 64)
	}
	payload, err := json.Marshal(map[string]any{"metrics": asStrings})
	if err != nil {
		Infof("failed to marshal metrics: %v", err)
		return
	}
	stdout.Printf("INFO  oss_rta_metrics\t%s", payload)
}

// Infof / Infow mirror zap's call sites in rpcshield. Downstream parsers grep
// on the message text, not the level prefix, so a stdlib log.Logger is enough.
func Infof(format string, a ...any) { stdout.Printf("INFO  "+format, a...) }
func Infow(msg string, kv ...any) {
	var sb strings.Builder
	sb.WriteString(msg)
	for i := 0; i+1 < len(kv); i += 2 {
		fmt.Fprintf(&sb, "\t%v=%v", kv[i], kv[i+1])
	}
	stdout.Printf("INFO  %s", sb.String())
}
func Fatalf(format string, a ...any) {
	stdout.Printf("FATAL "+format, a...)
	os.Exit(1)
}

func main() {
	var (
		datasetRoot = flag.String("dataset-root", "", "absolute path to the directory containing cloned OSS repos")
		targetName  = flag.String("target-name", "", "label for this target (used only in log messages)")
		targetDir   = flag.String("target-dir", "", "subdir of -dataset-root containing the project's go.mod (e.g., kubernetes, etcd/server)")
		targetPkg   = flag.String("target-pkg", ".", "package path relative to target-dir to analyze (e.g., ./cmd/kubelet)")
		flavorsCSV  = flag.String("flavors", "srta,srta_baseline,srta_kumo,srta_kumo_random,prta_kumo_nonblocking", "comma-separated RTA flavors")
		workersCSV  = flag.String("workers", "1,2,4,8,16,32,64", "comma-separated worker counts (sequential flavors collapse to 1)")
		buildCG     = flag.Bool("build-cg", true, "whether each Analyze call builds the call graph")
	)
	flag.Parse()

	if *datasetRoot == "" || *targetName == "" || *targetDir == "" {
		Fatalf("required flags: -dataset-root, -target-name, -target-dir (got -dataset-root=%q -target-name=%q -target-dir=%q)",
			*datasetRoot, *targetName, *targetDir)
	}

	flavors, err := parseFlavors(*flavorsCSV)
	if err != nil {
		Fatalf("parsing -flavors: %v", err)
	}
	workers, err := parseWorkers(*workersCSV)
	if err != nil {
		Fatalf("parsing -workers: %v", err)
	}

	Infof("target=%s dir=%s pkg=%s flavors=%s workers=%s",
		*targetName, *targetDir, *targetPkg, *flavorsCSV, *workersCSV)

	pkgByPath, prog, roots, err := loadProgram(*datasetRoot, *targetDir, *targetPkg)
	if err != nil {
		Fatalf("loading target: %v", err)
	}
	computeCorpusMetrics(pkgByPath, prog, *targetName)
	for _, fl := range flavors {
		ws := workers
		if fl.sequential {
			ws = []int{1}
		}
		for _, n := range ws {
			runOne(fl, roots, n, *buildCG)
		}
	}

	// rawGraphNumEdges/Nodes mirror the Uber wpa_metrics convention: the "raw"
	// call graph is the full RTA call graph before any pruning. All flavors
	// produce the same CG size, so we alias from the srta_1_workers result.
	if v, ok := metrics["rta_callgraph_edges_srta_1_workers"]; ok {
		metrics["rawGraphNumEdges"] = v
	}
	if v, ok := metrics["rta_callgraph_nodes_srta_1_workers"]; ok {
		metrics["rawGraphNumNodes"] = v
	}

	// Dump all metrics as one JSON line in the rpcshield wpa_metrics shape.
	emitMetricsJSON()

	// Completion marker: drives the resume check in run_oss_rta_sweep.sh and
	// matches rpcshield/rtabench.go:60 exactly.
	Infow("Finished CG construction-only evaluation")
}

func parseFlavors(s string) ([]flavorEntry, error) {
	names := splitCSV(s)
	if len(names) == 0 {
		return nil, fmt.Errorf("no flavors specified")
	}
	out := make([]flavorEntry, 0, len(names))
	for _, n := range names {
		fl, ok := flavorByName(n)
		if !ok {
			known := make([]string, 0, len(flavorRegistry))
			for _, f := range flavorRegistry {
				known = append(known, f.name)
			}
			return nil, fmt.Errorf("unknown flavor %q (known: %s)", n, strings.Join(known, ","))
		}
		out = append(out, fl)
	}
	return out, nil
}

func parseWorkers(s string) ([]int, error) {
	parts := splitCSV(s)
	if len(parts) == 0 {
		return nil, fmt.Errorf("no worker counts specified")
	}
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("worker count %q: %v", p, err)
		}
		if n < 1 {
			n = 1
		}
		out = append(out, n)
	}
	return out, nil
}

func splitCSV(s string) []string {
	out := []string{}
	for p := range strings.SplitSeq(s, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// loadProgram loads the target package's SSA program. Mirrors
// ~/prta_oss/kumo/rta_test.go::loadSSAProgram. cfg.Dir is the load-bearing
// setting: it tells go/packages' driver to run `go list` in that dir, so the
// target's own go.mod resolves transitive deps.
func loadProgram(datasetRoot, dir, pkg string) (map[string]*packages.Package, *ssa.Program, []*ssa.Function, error) {
	absRoot, err := filepath.Abs(datasetRoot)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("resolving dataset-root: %w", err)
	}
	loadDir := filepath.Join(absRoot, dir)
	if st, err := os.Stat(loadDir); err != nil || !st.IsDir() {
		return nil, nil, nil, fmt.Errorf("target dir not a directory: %s", loadDir)
	}

	Infof("Loading package %s from %s", pkg, loadDir)
	cfg := &packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedCompiledGoFiles |
			packages.NeedImports |
			packages.NeedDeps |
			packages.NeedTypes |
			packages.NeedSyntax |
			packages.NeedTypesInfo |
			packages.NeedTypesSizes,
		Dir:   loadDir,
		Tests: false,
	}
	loadStart := time.Now()
	pkgs, err := packages.Load(cfg, pkg)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("packages.Load: %w", err)
	}
	pkgErrCount := 0
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		pkgErrCount += len(p.Errors)
	})
	Infof("Loaded %d top-level packages in %v (%d package errors, non-fatal)",
		len(pkgs), time.Since(loadStart), pkgErrCount)

	ssaStart := time.Now()
	prog, ssaPkgs := ssautil.AllPackages(pkgs, ssa.InstantiateGenerics)
	prog.Build()
	Infof("SSA build complete in %v (%d ssa packages)", time.Since(ssaStart), len(ssaPkgs))

	// Build a path→packages.Package index so computeCorpusMetrics can look up
	// the AST/TypesInfo for each SSA package by import path.
	// Index by both p.PkgPath (the module-qualified path) and p.Types.Path()
	// (which is "main" for the program's main package in the types system).
	pkgByPath := make(map[string]*packages.Package)
	packages.Visit(pkgs, func(p *packages.Package) bool {
		pkgByPath[p.PkgPath] = p
		if p.Types != nil {
			pkgByPath[p.Types.Path()] = p
		}
		return true
	}, nil)

	roots := collectRoots(prog)
	if len(roots) == 0 {
		return nil, nil, nil, fmt.Errorf("no main/init roots found in loaded program (pkg errors: %d)", pkgErrCount)
	}
	mainCount := 0
	for _, fn := range roots {
		if fn.Name() == "main" {
			mainCount++
		}
	}
	if mainCount == 0 {
		Infof("warning: no main function found among %d roots (init-only)", len(roots))
	}
	Infof("Total roots (main + init functions): %d (mains: %d)", len(roots), mainCount)
	return pkgByPath, prog, roots, nil
}

// collectRoots returns one root per main function (from packages named "main")
// and per init function across all SSA packages.
func collectRoots(prog *ssa.Program) []*ssa.Function {
	var roots []*ssa.Function
	for _, p := range prog.AllPackages() {
		if p == nil || p.Pkg == nil {
			continue
		}
		if p.Pkg.Name() == "main" {
			if fn := p.Func("main"); fn != nil {
				roots = append(roots, fn)
			}
		}
		if fn := p.Func("init"); fn != nil {
			roots = append(roots, fn)
		}
	}
	return roots
}

// runOne executes one (flavor, numWorkers) iteration, mirroring
// rpcshield/graph/rtaeval.go:runSingleFlavor (lines 259-397) including the
// 100ms peak-heap ticker, getrusage CPU time, and the exact metric key shape.
func runOne(fl flavorEntry, roots []*ssa.Function, numWorkers int, buildCG bool) {
	runtime.GC()
	var memBefore runtime.MemStats
	runtime.ReadMemStats(&memBefore)
	baselineHeap := memBefore.HeapInuse

	var peakHeap atomic.Uint64
	peakHeap.Store(baselineHeap)
	stopTracker := make(chan struct{})
	trackerDone := make(chan struct{})
	go func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Fprintf(os.Stderr, "peak heap tracker panicked: %v\n", r)
			}
		}()
		defer close(trackerDone)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopTracker:
				return
			case <-ticker.C:
				var m runtime.MemStats
				runtime.ReadMemStats(&m)
				for {
					old := peakHeap.Load()
					if m.HeapInuse <= old {
						break
					}
					if peakHeap.CompareAndSwap(old, m.HeapInuse) {
						break
					}
				}
			}
		}
	}()

	var rusageBefore syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &rusageBefore)

	start := time.Now()
	Infof("Running RTA flavor: %s with %d workers", fl.name, numWorkers)

	impl := fl.ctor()
	res := impl.Analyze(roots, buildCG, rta.WithNumWorkers(numWorkers))

	duration := time.Since(start)

	var rusageAfter syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &rusageAfter)
	cpuTimeMs := float64(
		(rusageAfter.Utime.Sec-rusageBefore.Utime.Sec)*1000 +
			int64(rusageAfter.Utime.Usec-rusageBefore.Utime.Usec)/1000 +
			(rusageAfter.Stime.Sec-rusageBefore.Stime.Sec)*1000 +
			int64(rusageAfter.Stime.Usec-rusageBefore.Stime.Usec)/1000,
	)

	close(stopTracker)
	<-trackerDone
	var memAfter runtime.MemStats
	runtime.ReadMemStats(&memAfter)
	finalHeap := memAfter.HeapInuse
	for {
		old := peakHeap.Load()
		if finalHeap <= old {
			break
		}
		if peakHeap.CompareAndSwap(old, finalHeap) {
			break
		}
	}
	peakDelta := max(int64(peakHeap.Load())-int64(baselineHeap), 0)
	peakDeltaMB := float64(peakDelta) / (1024 * 1024)
	allocBytesDelta := memAfter.TotalAlloc - memBefore.TotalAlloc
	allocObjectsDelta := memAfter.Mallocs - memBefore.Mallocs
	allocBytesMB := float64(allocBytesDelta) / (1024 * 1024)

	// Metric key shape mirrors rtaeval.go lines 377-381.
	emitMetric("rta_duration_secs_%s_%d_workers", fl.name, numWorkers, duration.Seconds())
	emitMetric("rta_cpu_time_ms_%s_%d_workers", fl.name, numWorkers, cpuTimeMs)
	emitMetric("rta_peak_heap_delta_mb_%s_%d_workers", fl.name, numWorkers, peakDeltaMB)
	emitMetric("rta_alloc_bytes_total_mb_%s_%d_workers", fl.name, numWorkers, allocBytesMB)
	emitMetric("rta_alloc_objects_total_%s_%d_workers", fl.name, numWorkers, float64(allocObjectsDelta))

	// Summary line: byte-equivalent to rtaeval.go:383 so downstream parsers
	// keyed on this format keep working.
	Infof("RTA %s with %d workers took %v, cpu time: %.0f ms, peak heap delta: %.2f MB, total allocated: %.2f MB / %d objects",
		fl.name, numWorkers, duration, cpuTimeMs, peakDeltaMB, allocBytesMB, allocObjectsDelta)

	if res != nil && buildCG {
		if cg := res.GetCallGraph(); cg != nil {
			edgeCount := 0
			for _, node := range cg.Nodes {
				edgeCount += len(node.Out)
			}
			emitMetric("rta_callgraph_nodes_%s_%d_workers", fl.name, numWorkers, float64(len(cg.Nodes)))
			emitMetric("rta_callgraph_edges_%s_%d_workers", fl.name, numWorkers, float64(edgeCount))
		}
	}
	if res != nil {
		emitMetric("rta_reachable_funcs_%s_%d_workers", fl.name, numWorkers, float64(len(res.GetReachable())))
		collectOptionalMetrics(res, fl.name, numWorkers)
	}
}

// emitMetric stores one (flavor, workers) metric in the in-memory map; all
// metrics are dumped as one JSON line at the end by emitMetricsJSON, matching
// rpcshield's wpa_metrics output. We keep the per-metric Infof out of the log
// because the JSON dump is the canonical source.
func emitMetric(key, flavor string, workers int, val float64) {
	recordMetric(key, flavor, workers, val)
}

// collectOptionalMetrics replicates the optional-counter type assertions in
// rpcshield/graph/rtaeval.go:collectRTAMetrics.
func collectOptionalMetrics(res rta.Result, flavor string, numWorkers int) {
	if c, ok := res.(rta.ImplementsCounter); ok {
		emitMetric("rta_implements_calls_%s_%d_workers", flavor, numWorkers, float64(c.ImplementsCallCount()))
	}
	if c, ok := res.(rta.TypeCounter); ok {
		emitMetric("rta_concrete_types_%s_%d_workers", flavor, numWorkers, float64(c.NumConcreteTypes()))
		emitMetric("rta_interface_types_%s_%d_workers", flavor, numWorkers, float64(c.NumInterfaceTypes()))
	}
	if c, ok := res.(rta.ImplementsSuccessFailCounter); ok {
		emitMetric("rta_implements_success_%s_%d_workers", flavor, numWorkers, float64(c.ImplementsSuccessCount()))
		emitMetric("rta_implements_fail_%s_%d_workers", flavor, numWorkers, float64(c.ImplementsFailCount()))
	}
	if c, ok := res.(rta.ImplementsSourceCounter); ok {
		emitMetric("rta_checks_from_interfaces_%s_%d_workers", flavor, numWorkers, float64(c.ChecksFromInterfaces()))
		emitMetric("rta_checks_from_implementations_%s_%d_workers", flavor, numWorkers, float64(c.ChecksFromImplementations()))
	}
	if c, ok := res.(rta.ImplementsDirectionCounter); ok {
		emitMetric("rta_fails_from_interfaces_%s_%d_workers", flavor, numWorkers, float64(c.FailsFromInterfaces()))
		emitMetric("rta_success_from_interfaces_%s_%d_workers", flavor, numWorkers, float64(c.SuccessFromInterfaces()))
		emitMetric("rta_fails_from_implementations_%s_%d_workers", flavor, numWorkers, float64(c.FailsFromImplementations()))
		emitMetric("rta_success_from_implementations_%s_%d_workers", flavor, numWorkers, float64(c.SuccessFromImplementations()))
	}
	if c, ok := res.(rta.CallSiteCounter); ok {
		emitMetric("rta_static_call_sites_%s_%d_workers", flavor, numWorkers, float64(c.StaticCallSites()))
		emitMetric("rta_indirect_call_sites_%s_%d_workers", flavor, numWorkers, float64(c.IndirectCallSites()))
		emitMetric("rta_invoke_call_sites_%s_%d_workers", flavor, numWorkers, float64(c.InvokeCallSites()))
		emitMetric("rta_total_call_sites_%s_%d_workers", flavor, numWorkers, float64(c.TotalCallSites()))
	}
	if c, ok := res.(rta.IndexStats); ok {
		mc := c.MCBucketSizePercentiles()
		mi := c.MIBucketSizePercentiles()
		for _, p := range []struct {
			name string
			val  int
		}{
			{"p50", mc.P50}, {"p90", mc.P90}, {"p95", mc.P95}, {"p99", mc.P99}, {"p100", mc.P100},
		} {
			emitMetric("rta_mc_bucket_"+p.name+"_%s_%d_workers", flavor, numWorkers, float64(p.val))
		}
		for _, p := range []struct {
			name string
			val  int
		}{
			{"p50", mi.P50}, {"p90", mi.P90}, {"p95", mi.P95}, {"p99", mi.P99}, {"p100", mi.P100},
		} {
			emitMetric("rta_mi_bucket_"+p.name+"_%s_%d_workers", flavor, numWorkers, float64(p.val))
		}
		bucketLabels := [8]string{"1", "2", "4", "8", "16", "32", "64", "gt64"}
		buckets := c.InterfaceMethodCountBuckets()
		for i, label := range bucketLabels {
			emitMetric("rta_frac_iface_methods_le"+label+"_%s_%d_workers", flavor, numWorkers, buckets[i])
		}
	}
	if c, ok := res.(rta.MethodStats); ok {
		mpc := c.MethodsPerConcretePercentiles()
		mpi := c.MethodsPerInterfacePercentiles()
		for _, p := range []struct {
			name       string
			cval, ival int
		}{
			{"p50", mpc.P50, mpi.P50}, {"p95", mpc.P95, mpi.P95}, {"p99", mpc.P99, mpi.P99}, {"p100", mpc.P100, mpi.P100},
		} {
			emitMetric("rta_methods_per_concrete_"+p.name+"_%s_%d_workers", flavor, numWorkers, float64(p.cval))
			emitMetric("rta_methods_per_interface_"+p.name+"_%s_%d_workers", flavor, numWorkers, float64(p.ival))
		}
		concBucketLabels := [8]string{"1", "2", "4", "8", "16", "32", "64", "gt64"}
		concBuckets := c.ConcreteMethodCountBuckets()
		for i, label := range concBucketLabels {
			emitMetric("rta_frac_concrete_methods_le"+label+"_%s_%d_workers", flavor, numWorkers, concBuckets[i])
		}
	}
	if c, ok := res.(rta.InvokeLookupCounter); ok {
		emitMetric("rta_invoke_lookups_from_visit_invoke_%s_%d_workers", flavor, numWorkers, float64(c.InvokeLookupsFromVisitInvoke()))
		emitMetric("rta_invoke_lookups_from_add_runtime_type_%s_%d_workers", flavor, numWorkers, float64(c.InvokeLookupsFromAddRuntimeType()))
	}
}

// computeCorpusMetrics mirrors rpcshield/graph/graph.go:computeCodebaseMetrics
// exactly — no stdlib filter, no path-based filtering. In the rpcshield pipeline,
// b.pkgs is already bounded to non-stdlib packages by the Bazel go_path loader;
// for OSS, pkgByPath contains whatever packages.Visit found.
func computeCorpusMetrics(pkgByPath map[string]*packages.Package, ssaProg *ssa.Program, targetName string) {
	numPackages := 0
	numFunctions := 0
	numFiles := 0
	numLinesOfCode := 0
	numConcreteTypes := 0
	numInterfaces := 0

	seenPkgs := make(map[*packages.Package]bool)
	seenFiles := make(map[string]bool)
	seenTypes := make(map[types.Type]bool)

	for _, pkg := range pkgByPath {
		if pkg == nil || seenPkgs[pkg] {
			continue
		}
		seenPkgs[pkg] = true
		numPackages++

		for _, file := range pkg.Syntax {
			if file == nil {
				continue
			}
			pos := pkg.Fset.Position(file.Pos())
			if pos.Filename != "" && !seenFiles[pos.Filename] {
				seenFiles[pos.Filename] = true
				numFiles++
				endPos := pkg.Fset.Position(file.End())
				numLinesOfCode += endPos.Line
			}
			for _, decl := range file.Decls {
				if _, ok := decl.(*ast.FuncDecl); ok {
					numFunctions++
				}
			}
		}

		if pkg.TypesInfo != nil {
			for _, obj := range pkg.TypesInfo.Defs {
				if obj == nil {
					continue
				}
				typeName, ok := obj.(*types.TypeName)
				if !ok {
					continue
				}
				typ := typeName.Type()
				if seenTypes[typ] {
					continue
				}
				seenTypes[typ] = true
				if _, isIface := typ.Underlying().(*types.Interface); isIface {
					numInterfaces++
				} else {
					numConcreteTypes++
				}
			}
		}
	}

	ssaFuncCount := 0
	if ssaProg != nil {
		for fn := range ssautil.AllFunctions(ssaProg) {
			if fn != nil {
				ssaFuncCount++
			}
		}
	}

	metrics["num_packages"] = float64(numPackages)
	metrics["num_functions"] = float64(numFunctions)
	metrics["num_files"] = float64(numFiles)
	metrics["num_lines_of_code"] = float64(numLinesOfCode)
	metrics["num_concrete_types"] = float64(numConcreteTypes)
	metrics["num_interfaces"] = float64(numInterfaces)
	metrics["num_ssa_functions"] = float64(ssaFuncCount)
	metricsStr["service"] = targetName
}
