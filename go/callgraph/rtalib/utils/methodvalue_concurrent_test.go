package utils

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/tools/go/ssa"
)

// buildSSAManyEmbedded builds an SSA program with n type pairs of the form
//
//	type InnerK struct{}
//	func (InnerK) M() {}
//	type OuterK struct{ InnerK }
//
// Each (*OuterK).M selection has Index() length > 1, which forces
// prog.MethodValue down the wrapper-synthesis path — exactly the path
// that contends on prog.methodsMu under load.
func buildSSAManyEmbedded(tb testing.TB, n int) (*ssa.Program, *types.Package) {
	tb.Helper()
	var sb strings.Builder
	sb.WriteString("package pkg\n")
	for i := 0; i < n; i++ {
		fmt.Fprintf(&sb, "type Inner%d struct{}\n", i)
		fmt.Fprintf(&sb, "func (Inner%d) M() {}\n", i)
		fmt.Fprintf(&sb, "type Outer%d struct{ Inner%d }\n", i, i)
	}
	src := sb.String()

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", src, 0)
	require.NoError(tb, err)

	var conf types.Config
	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	pkg, err := conf.Check("test/pkg", fset, []*ast.File{file}, info)
	require.NoError(tb, err)

	prog := ssa.NewProgram(fset, ssa.InstantiateGenerics)
	ssaPkg := prog.CreatePackage(pkg, []*ast.File{file}, info, true)
	ssaPkg.Build()
	return prog, pkg
}

// promotedSelections returns the *types.Selection for OuterK.M for each
// of the n Outer types. Each selection has len(Index()) > 1, meaning
// prog.MethodValue must synthesize a promotion wrapper on first call.
func promotedSelections(tb testing.TB, pkg *types.Package, n int) []*types.Selection {
	tb.Helper()
	sels := make([]*types.Selection, n)
	for i := 0; i < n; i++ {
		outerType := pkg.Scope().Lookup(fmt.Sprintf("Outer%d", i)).Type()
		require.NotNil(tb, outerType, "Outer%d not found", i)
		sel, ok := types.LookupSelection(outerType, false, pkg, "M")
		require.Truef(tb, ok, "Outer%d.M not found", i)
		require.Greaterf(tb, len(sel.Index()), 1,
			"Outer%d.M should be promoted (Index length > 1) so prog.MethodValue "+
				"takes the wrapper-synthesis path", i)
		sels[i] = &sel
	}
	return sels
}

// TestMethodValue_ConcurrentCorrectness exercises prog.MethodValue from
// many goroutines, on many distinct promoted-method selections that each
// require wrapper synthesis. It asserts the functional invariants of
// MethodValue under concurrency:
//
//  1. every call returns a non-nil *ssa.Function with the right name,
//     even when several workers race to be the first to synthesize the
//     wrapper for a given type;
//  2. all callers observe the SAME *ssa.Function for the same selection
//     (idempotency across workers and across repeated calls);
//  3. wrapper synthesis for distinct receiver types T is independent —
//     no worker reads a partially-built function from another type.
//
// These invariants must hold both with and without the per-methodSet
// sharding patch in golang.org/x/tools/go/ssa: the test asserts
// behavior, not lock topology. It serves as the correctness baseline
// that the upcoming patch must preserve.
//
// Workload is sized to finish in well under a second on a quiet box,
// and to stay comfortably under the test target's timeout even when
// CI runs --runs_per_test=300 copies of this binary on a shared host —
// 8 workers x 5 iters x 20 promoted selections still races the wrapper
// synthesis path between workers (each type's first call races among
// 8 concurrent observers) without saturating the machine.
func TestMethodValue_ConcurrentCorrectness(t *testing.T) {
	const (
		numTypes = 20
		workers  = 8
		iters    = 5
	)
	prog, pkg := buildSSAManyEmbedded(t, numTypes)
	sels := promotedSelections(t, pkg, numTypes)

	// Per-worker observed *ssa.Function for each type. We compare across
	// workers at the end to assert all workers agree.
	observed := make([][]*ssa.Function, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		w := w
		observed[w] = make([]*ssa.Function, numTypes)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for it := 0; it < iters; it++ {
				for i, sel := range sels {
					fn := prog.MethodValue(sel)
					if fn == nil {
						t.Errorf("worker %d iter %d type %d: prog.MethodValue returned nil", w, it, i)
						continue
					}
					if fn.Name() != "M" {
						t.Errorf("worker %d iter %d type %d: name %q, want %q",
							w, it, i, fn.Name(), "M")
					}
					if fn.Signature.Recv() == nil {
						t.Errorf("worker %d iter %d type %d: wrapper missing receiver", w, it, i)
					}
					prev := observed[w][i]
					if prev == nil {
						observed[w][i] = fn
					} else if prev != fn {
						t.Errorf("worker %d type %d: MethodValue returned different *Function "+
							"across iterations (%p then %p) — wrapper not idempotently cached",
							w, i, prev, fn)
					}
				}
			}
		}()
	}
	wg.Wait()

	// All workers must agree on the canonical *Function for each type.
	for i := 0; i < numTypes; i++ {
		canon := observed[0][i]
		require.NotNil(t, canon, "worker 0 never observed type %d", i)
		for w := 1; w < workers; w++ {
			require.Samef(t, canon, observed[w][i],
				"worker %d disagrees with worker 0 on type %d's *Function — "+
					"concurrent MethodValue is not returning a single canonical wrapper",
				w, i)
		}
	}
}

// BenchmarkMethodValue_ConcurrentWrapperSynthesis measures the throughput
// of parallel prog.MethodValue calls when each call performs a FRESH
// wrapper synthesis (not a cache hit). Wrapper synthesis is the
// expensive operation that the per-methodSet sharding patch is designed
// to parallelize.
//
// Each benchmark iteration calls MethodValue on a unique selection so
// vanilla SSA serializes every iteration on prog.methodsMu, while
// patched SSA serializes only briefly on prog.methodsMu (the mset
// lookup) and runs the actual createWrapper work under a per-T mset.mu
// that doesn't collide across distinct types.
//
// Expected reading:
//   - vanilla SSA: throughput per goroutine is flat or regresses with
//     GOMAXPROCS — every synthesis stalls behind every other one;
//   - patched SSA: throughput per goroutine stays roughly flat as
//     GOMAXPROCS grows, so aggregate wall time scales near-linearly with
//     parallelism.
//
// Run with -benchtime=Nx (e.g. -benchtime=2000x) so b.N is small enough
// to keep the SSA-program build cost reasonable while still exercising
// the contention. Each fresh synthesis costs a few μs of real work, so
// any noise from cache-hit overhead is amortized away.
func BenchmarkMethodValue_ConcurrentWrapperSynthesis(b *testing.B) {
	procs := runtime.GOMAXPROCS(0)
	// Build at least b.N unique types so every iteration is a fresh
	// synthesis. Floor at procs*32 to avoid degenerate tiny programs
	// when -benchtime=1x is used for sanity checks.
	numTypes := b.N
	if min := procs * 32; numTypes < min {
		numTypes = min
	}

	prog, pkg := buildSSAManyEmbedded(b, numTypes)
	sels := promotedSelections(b, pkg, numTypes)

	b.ReportAllocs()
	b.ResetTimer()

	// Shared monotonic counter ensures every goroutine picks a unique
	// selection (modulo numTypes) — no two goroutines synthesize the
	// SAME wrapper, so both vanilla and patched SSA exercise their full
	// parallel-synthesis capacity.
	var ctr atomic.Int64
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := int(ctr.Add(1)-1) % numTypes
			_ = prog.MethodValue(sels[i])
		}
	})
}
