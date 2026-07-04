package rtatest

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/tools/go/callgraph"
	rta "golang.org/x/tools/go/callgraph/internal/rtautil"
	rtapkg "golang.org/x/tools/go/callgraph/rta"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// tidbPkg is the package analyzed for correctness testing, matching the
// "tidb" row in evaluation_scripts/targets.csv.
const tidbPkg = "./cmd/tidb-server"

// errDatasetMissing distinguishes "tidb hasn't been cloned yet" from a
// genuine load failure, so callers can t.Skip instead of failing.
var errDatasetMissing = errors.New("tidb dataset not found")

// tidbDir resolves the tidb checkout used as the correctness-test fixture.
// RTA_TIDB_DIR overrides; otherwise it defaults to the datasets/ directory
// evaluation_scripts/clone_datasets.sh populates next to this package.
func tidbDir() string {
	if d := os.Getenv("RTA_TIDB_DIR"); d != "" {
		return d
	}
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "evaluation_scripts", "datasets", "tidb")
}

// loadTiDB loads the tidb-server SSA program once per test binary, mirroring
// evaluation_scripts/cmd/oss_rta_bench/main.go:loadProgram.
var loadTiDB = sync.OnceValues(func() (*ssa.Program, error) {
	dir, err := filepath.Abs(tidbDir())
	if err != nil {
		return nil, fmt.Errorf("resolving tidb dir: %w", err)
	}
	if st, statErr := os.Stat(dir); statErr != nil || !st.IsDir() {
		return nil, fmt.Errorf("%w: %s (run evaluation_scripts/clone_datasets.sh, or set RTA_TIDB_DIR)", errDatasetMissing, dir)
	}

	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedDeps | packages.NeedTypes |
			packages.NeedSyntax | packages.NeedTypesInfo | packages.NeedTypesSizes,
		Dir:   dir,
		Tests: false,
	}
	pkgs, err := packages.Load(cfg, tidbPkg)
	if err != nil {
		return nil, fmt.Errorf("packages.Load: %w", err)
	}

	prog, _ := ssautil.AllPackages(pkgs, ssa.InstantiateGenerics)
	prog.Build()
	return prog, nil
})

// mustLoadTiDB returns the tidb SSA program, or skips the test if the
// dataset hasn't been cloned yet.
func mustLoadTiDB(t *testing.T) *ssa.Program {
	t.Helper()
	prog, err := loadTiDB()
	if errors.Is(err, errDatasetMissing) {
		t.Skip(err)
	}
	require.NoError(t, err, "loading tidb SSA program")
	return prog
}

// graphStats holds summary statistics for a callgraph.
type graphStats struct {
	Nodes     int
	Edges     int
	GraphHash [32]byte
}

// computeStats counts nodes and edges and computes a neighbor-based hash
// (XOR of per-node SHA256) for graph equivalence checking.
func computeStats(g *callgraph.Graph) graphStats {
	var s graphStats
	s.Nodes = len(g.Nodes)
	for fn, node := range g.Nodes {
		s.Edges += len(node.Out)
		if fn == nil {
			continue
		}
		callees := make([]string, 0, len(node.Out))
		for _, edge := range node.Out {
			if edge.Callee != nil && edge.Callee.Func != nil {
				callees = append(callees, edge.Callee.Func.String())
			}
		}
		sort.Strings(callees)
		h := sha256.New()
		h.Write([]byte(fn.String()))
		for _, c := range callees {
			h.Write([]byte(c))
		}
		nodeHash := h.Sum(nil)
		for i := range s.GraphHash {
			s.GraphHash[i] ^= nodeHash[i]
		}
	}
	return s
}

// roots extracts main and init entry-point functions from the SSA program.
func roots(prog *ssa.Program) []*ssa.Function {
	var entry []*ssa.Function
	for f, ok := range ssautil.AllFunctions(prog) {
		if !ok || (f.Signature != nil && f.Signature.Recv() != nil) {
			continue
		}
		if f.Name() == "main" || f.Name() == "init" {
			entry = append(entry, f)
		}
	}
	return entry
}

// LoadProgramAndRoots returns the tidb SSA program and root functions.
// Use this in per-flavor tests to call Analyze directly and exercise
// flavor-specific Result methods.
func LoadProgramAndRoots(t *testing.T) (*ssa.Program, []*ssa.Function) {
	t.Helper()
	prog := mustLoadTiDB(t)
	entry := roots(prog)
	require.NotEmpty(t, entry, "no root functions found")
	return prog, entry
}

// AssertMatchesStdlib runs the given RTA flavor on tidb-server and asserts
// that the resulting callgraph has the same number of nodes, edges, and
// neighbor hash as the built-in golang.org/x/tools/go/callgraph/rta.
func AssertMatchesStdlib(t *testing.T, flavor rta.RTA, opts ...rta.AnalyzeOption) {
	t.Helper()

	prog := mustLoadTiDB(t)
	entry := roots(prog)
	require.NotEmpty(t, entry, "no root functions found")

	// Reference: built-in RTA.
	refResult := rtapkg.Analyze(entry, true)
	require.NotNil(t, refResult, "stdlib RTA returned nil")
	refStats := computeStats(refResult.CallGraph)

	// Flavor under test.
	flavorResult := flavor.Analyze(entry, true, opts...)
	require.NotNil(t, flavorResult, "flavor Analyze returned nil")
	flavorCG := flavorResult.GetCallGraph()
	require.NotNil(t, flavorCG, "flavor callgraph is nil")
	flavorStats := computeStats(flavorCG)

	require.Equal(t, refStats.Nodes, flavorStats.Nodes, "node count mismatch")
	require.Equal(t, refStats.Edges, flavorStats.Edges, "edge count mismatch")
	require.Equal(t, refStats.GraphHash, flavorStats.GraphHash, "graph hash mismatch")

	// Exercise remaining Result interface methods for coverage.
	reachable := flavorResult.GetReachable()
	require.NotNil(t, reachable, "GetReachable returned nil")
	require.NotEmpty(t, reachable, "GetReachable returned empty map")

	runtimeTypes := flavorResult.GetRuntimeTypes()
	require.Greater(t, runtimeTypes.Len(), 0, "GetRuntimeTypes returned empty map")

	materialized := flavorResult.Materialize()
	require.NotNil(t, materialized, "Materialize returned nil")
	require.NotNil(t, materialized.CallGraph, "Materialized CallGraph is nil")
	require.NotNil(t, materialized.Reachable, "Materialized Reachable is nil")
	require.Greater(t, materialized.RuntimeTypes.Len(), 0, "Materialized RuntimeTypes is empty")
}
