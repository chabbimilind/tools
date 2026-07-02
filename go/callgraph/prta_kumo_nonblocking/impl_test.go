package prta_kumo_nonblocking

import (
	"go/types"
	"sync"
	"sync/atomic"
	"testing"

	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/rtalib/rtatest"
	"golang.org/x/tools/go/callgraph/rtalib/utils"
	"golang.org/x/tools/go/types/typeutil"
)

func TestCorrectness(t *testing.T) {
	t.Run("RarestMethodByConcreteType", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4))
	})
	t.Run("RarestMethodByInterfaceType", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4), rtalib.WithMethodSelectionStrategy(rtalib.RarestMethodByInterfaceType))
	})
	t.Run("RandomMethodStrategy", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4), rtalib.WithMethodSelectionStrategy(rtalib.RandomMethodStrategy))
	})
}

func TestResultMethods(t *testing.T) {
	_, entry := rtatest.LoadProgramAndRoots(t)
	result := New().Analyze(entry, true, rtalib.WithNumWorkers(4)).(*Result)

	_ = result.MCBucketSizePercentiles()
	_ = result.MIBucketSizePercentiles()
	_ = result.InterfaceMethodCountBuckets()
}

//
// Two invariants together make the design correct:
//
//  1. typeSet / concretePtrSet / interfacePtrSet use Go map[K]struct{}
//     keyed by pointer == (NOT types.Identical). Two distinct pointers
//     that are types.Identical produce two separate entries. This is
//     what these tests pin down directly.
//
//  2. typeSet is only ever stored with keys that have already been
//     canonicalized upstream by concreteTypes / interfaceTypes -- both
//     utils.TypeMap, which use types.Identical via LoadOrStore. So in
//     the real algorithm, only the canonical pointer for an
//     Identical-equivalence class ever reaches a Set call.
//     "TestUpstreamCanonicalization" pins that down.
//
// Together: typeSet behaves "as if" Identical-keyed, at ~90× less memory.

// TestTypeSetPointerIdentity: typeSet dedups by pointer ==, not Identical.
// Two distinct *types.Pointer pointers that compare types.Identical produce
// TWO entries.
func TestTypeSetPointerIdentity(t *testing.T) {
	// types.NewPointer allocates a fresh *types.Pointer on each call, so
	// the two pointers below are distinct but Identical (both *bool).
	// (We can't use NewInterfaceType(nil, nil) because the stdlib interns
	// the empty interface to a single sentinel.)
	p1 := types.NewPointer(types.Typ[types.Bool])
	p2 := types.NewPointer(types.Typ[types.Bool])

	if p1 == p2 {
		t.Fatal("test setup: NewPointer returned same pointer twice")
	}
	if !types.Identical(p1, p2) {
		t.Fatal("test setup: two *bool pointers should be types.Identical")
	}

	var s typeSet
	s.Set(p1)
	s.Set(p1) // same pointer -> idempotent
	if got := s.Len(); got != 1 {
		t.Errorf("after Set(p1) twice, Len=%d want 1 (pointer-identity dedup)", got)
	}
	s.Set(p2) // Identical but different pointer -> new entry
	if got := s.Len(); got != 2 {
		t.Errorf("after Set(p2), Len=%d want 2 (Identical but distinct pointers count separately)", got)
	}

	seen := map[types.Type]int{}
	s.Iterate(func(k types.Type) { seen[k]++ })
	if len(seen) != 2 || seen[p1] != 1 || seen[p2] != 1 {
		t.Errorf("Iterate visits = %v, want {p1:1, p2:1}", seen)
	}
}

// TestUpstreamCanonicalization: utils.TypeMap.LoadOrStore (Identical-keyed)
// collapses different-pointer-but-Identical inputs to a single canonical
// stored value. This is what makes typeSet's pointer-identity correct in
// the real algorithm: every Set into cinfo.implements / iinfo.implementations
// uses iinfo.I or cinfo.C from the WINNING info, never a loser's input.
func TestUpstreamCanonicalization(t *testing.T) {
	p1 := types.NewPointer(types.Typ[types.Bool])
	p2 := types.NewPointer(types.Typ[types.Bool])
	if p1 == p2 || !types.Identical(p1, p2) {
		t.Fatal("test setup invariant broken")
	}

	var m utils.TypeMap
	m.SetHasher(typeutil.MakeHasher())

	// Two "threads" race to install their own cinfo. The first thread
	// stores cinfoA with cinfoA.C = p1; the second sees the existing
	// entry and discards cinfoB.
	cinfoA := &concreteTypeInfo{C: p1}
	cinfoB := &concreteTypeInfo{C: p2}

	got1, loaded1 := m.LoadOrStore(p1, cinfoA)
	got2, loaded2 := m.LoadOrStore(p2, cinfoB)

	if loaded1 {
		t.Error("first LoadOrStore: loaded=true, want false")
	}
	if !loaded2 {
		t.Error("second LoadOrStore (Identical key, different pointer): loaded=false, want true")
	}
	if got1.(*concreteTypeInfo) != cinfoA {
		t.Errorf("first LoadOrStore: returned %p, want %p", got1, cinfoA)
	}
	if got2.(*concreteTypeInfo) != cinfoA {
		t.Errorf("second LoadOrStore: returned %p, want %p (canonical)", got2, cinfoA)
	}

	// Downstream invariant: any Set into an iinfo.implementations uses
	// the canonical cinfo.C (= p1 here), not the loser's p2. So a typeSet
	// receiving Set calls from both threads' code paths sees only one
	// pointer (p1) for this Identical-class -- no duplicate entry.
	var ts typeSet
	canonicalC := got2.(*concreteTypeInfo).C // both threads would use this
	ts.Set(canonicalC)
	ts.Set(canonicalC) // second thread's redundant Set is a no-op
	if got := ts.Len(); got != 1 {
		t.Errorf("typeSet.Len=%d want 1 (both threads insert canonical pointer)", got)
	}
}

// TestTypeSetConcurrentSafe stresses concurrent Set + Iterate. The pooled
// snapshot under RWMutex must not corrupt state or observe partial writes.
// Intended to be exercised with -race.
func TestTypeSetConcurrentSafe(t *testing.T) {
	const Workers = 16
	const N = 1024

	keys := make([]types.Type, N)
	for i := range keys {
		// types.NewPointer allocates a fresh *types.Pointer per call, so
		// all N pointers are distinct (and pointer-identity dedup means
		// the final Len should be exactly N).
		keys[i] = types.NewPointer(types.Typ[types.Bool])
	}

	var s typeSet
	var wg sync.WaitGroup

	// Writers: each inserts all N keys, starting at a different offset
	// so writers concurrently hit overlapping subsets.
	for w := 0; w < Workers; w++ {
		wg.Add(1)
		go func(off int) {
			defer wg.Done()
			for i := 0; i < N; i++ {
				s.Set(keys[(i+off)%N])
			}
		}(w)
	}
	// Iterators: snapshot under RLock and walk; must never see a nil key
	// or more than N entries.
	for w := 0; w < Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				n := 0
				s.Iterate(func(k types.Type) {
					if k == nil {
						t.Error("nil key in typeSet snapshot")
					}
					n++
				})
				if n > N {
					t.Errorf("Iterate saw %d entries, > N=%d", n, N)
				}
			}
		}()
	}
	wg.Wait()

	if got := s.Len(); got != N {
		t.Errorf("after all writers done, Len=%d want %d", got, N)
	}
	seen := map[types.Type]int{}
	s.Iterate(func(k types.Type) { seen[k]++ })
	for _, k := range keys {
		if seen[k] != 1 {
			t.Errorf("key %p visited %d times, want 1", k, seen[k])
		}
	}
}

// TestConcretePtrSetPointerIdentity: pointer-identity dedup for the inner
// MC bucket type.
func TestConcretePtrSetPointerIdentity(t *testing.T) {
	a := &concreteTypeInfo{}
	b := &concreteTypeInfo{}

	var s concretePtrSet
	s.Store(a)
	s.Store(a)
	if got := s.Size(); got != 1 {
		t.Errorf("Size after duplicate Store(a): got %d want 1", got)
	}
	s.Store(b)
	if got := s.Size(); got != 2 {
		t.Errorf("Size after Store(b): got %d want 2", got)
	}

	seen := map[*concreteTypeInfo]int{}
	s.Range(func(c *concreteTypeInfo) bool {
		seen[c]++
		return true
	})
	if seen[a] != 1 || seen[b] != 1 || len(seen) != 2 {
		t.Errorf("Range visits = %v, want {a:1, b:1}", seen)
	}

	// Early termination: returning false stops iteration.
	var visited []*concreteTypeInfo
	s.Range(func(c *concreteTypeInfo) bool {
		visited = append(visited, c)
		return false
	})
	if len(visited) != 1 {
		t.Errorf("Range with f returning false visited %d items, want 1", len(visited))
	}
}

// TestInterfacePtrSetPointerIdentity: mirror of TestConcretePtrSet for MI.
func TestInterfacePtrSetPointerIdentity(t *testing.T) {
	a := &interfaceTypeInfo{}
	b := &interfaceTypeInfo{}

	var s interfacePtrSet
	s.Store(a)
	s.Store(a)
	if got := s.Size(); got != 1 {
		t.Errorf("Size after duplicate Store(a): got %d want 1", got)
	}
	s.Store(b)

	seen := map[*interfaceTypeInfo]int{}
	s.Range(func(i *interfaceTypeInfo) bool { seen[i]++; return true })
	if seen[a] != 1 || seen[b] != 1 || len(seen) != 2 {
		t.Errorf("Range visits = %v", seen)
	}
}

// TestPtrSetConcurrentSafe stresses concurrent Store + Range on
// concretePtrSet (the same code path is shared with interfacePtrSet via
// generics). Intended to be exercised with -race.
func TestPtrSetConcurrentSafe(t *testing.T) {
	const Workers = 16
	const N = 1024

	keys := make([]*concreteTypeInfo, N)
	for i := range keys {
		keys[i] = &concreteTypeInfo{}
	}

	var s concretePtrSet
	var wg sync.WaitGroup

	for w := 0; w < Workers; w++ {
		wg.Add(1)
		go func(off int) {
			defer wg.Done()
			for i := 0; i < N; i++ {
				s.Store(keys[(i+off)%N])
			}
		}(w)
	}
	for w := 0; w < Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				n := 0
				s.Range(func(c *concreteTypeInfo) bool {
					if c == nil {
						t.Error("nil entry in concretePtrSet snapshot")
					}
					n++
					return true
				})
				if n > N {
					t.Errorf("Range saw %d entries, > N=%d", n, N)
				}
			}
		}()
	}
	wg.Wait()

	if got := s.Size(); got != N {
		t.Errorf("after all writers done, Size=%d want %d", got, N)
	}
}

// TestMethodSitesMapLoadOrComputeOnce: under concurrent LoadOrCompute on
// the same key, newEntry runs exactly once and all callers receive the same
// entry pointer. Pins down the double-checked-locking pattern used to
// replace xsync.Map.LoadOrCompute for sitesByMethod.
func TestMethodSitesMapLoadOrComputeOnce(t *testing.T) {
	const Workers = 64
	var m methodSitesMap

	var newCount atomic.Int32
	var wg sync.WaitGroup
	results := make([]*methodSitesEntry, Workers)
	start := make(chan struct{})

	for w := 0; w < Workers; w++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start // all workers race for the same key
			results[i] = m.LoadOrCompute("foo", func() *methodSitesEntry {
				newCount.Add(1)
				return &methodSitesEntry{}
			})
		}(w)
	}
	close(start)
	wg.Wait()

	if n := newCount.Load(); n != 1 {
		t.Errorf("newEntry called %d times, want 1 (double-checked locking should serialize creation)", n)
	}
	for i, r := range results {
		if r != results[0] {
			t.Errorf("worker %d got entry %p, want %p (same canonical)", i, r, results[0])
		}
	}

	// A subsequent LoadOrCompute with a different newEntry must NOT call it.
	got := m.LoadOrCompute("foo", func() *methodSitesEntry {
		t.Error("newEntry called for already-cached key")
		return nil
	})
	if got != results[0] {
		t.Errorf("cached LoadOrCompute returned %p, want %p", got, results[0])
	}
}
