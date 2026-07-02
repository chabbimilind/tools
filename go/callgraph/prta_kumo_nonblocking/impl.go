// Code is heavily borrowed from Go's RTA package. We have modified it make
// it parallel.

package prta_kumo_nonblocking

import (
	"fmt"
	"go/types"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/puzpuzpuz/xsync/v4"
	"golang.org/x/tools/go/callgraph"
	rtapkg "golang.org/x/tools/go/callgraph/rta"
	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/rtalib/utils"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types/typeutil"
)

// Result implements the rta.Result interface for parallel RTA.
// It uses xsync.Map internally for concurrent access during analysis.
type Result struct {
	rtalib.BaseMetrics
	callGraph    *utils.ConcurrentCallGraph
	reachable    *xsync.Map[*ssa.Function, bool]
	runtimeTypes utils.TypeMap

	// Lazy index stats. The inner buckets are pointer-keyed sets backed by
	// a plain Go map + sync.Mutex (concretePtrSet / interfacePtrSet) rather
	// than xsync.Map[X, bool] -- per-method-name the latter cost ~4.5 KB
	// even when empty, which dominated allocation on services with many
	// method names.
	mcMap                  *xsync.Map[string, *concretePtrSet]
	miMap                  *xsync.Map[string, *interfacePtrSet]
	interfaceTypesMap      *utils.TypeMap
	mcBucketOnce           sync.Once
	mcBucketPercentiles    rtalib.Percentiles
	miBucketOnce           sync.Once
	miBucketPercentiles    rtalib.Percentiles
	ifaceMethodBucketsOnce sync.Once
	ifaceMethodBuckets     [8]float64
}

// GetCallGraph returns the discovered callgraph.
func (r *Result) GetCallGraph() *callgraph.Graph {
	if r.callGraph == nil {
		return nil
	}
	return r.callGraph.GetGraph()
}

// sitesSnapshotPool reuses []ssa.CallInstruction scratch buffers used to
// snapshot entry.sites under entry.mu so AddCallGraphEdgeFast can run
// outside the lock. addRuntimeType (and visitAddrTakenFunc) hit this
// path millions of times on large services — pooling cuts those small
// allocations to ~0 amortized, lowering the per-second allocation rate
// that drives peak heap above the live working set.
//
// We pool *[]T, not []T: putting a 3-word slice header into sync.Pool's
// interface{} would box-allocate ~24 B per Put (staticcheck SA6002),
// which on millions of Puts becomes meaningful. A pointer fits in the
// interface's data word directly with no boxing alloc.
var sitesSnapshotPool = sync.Pool{
	New: func() any {
		s := make([]ssa.CallInstruction, 0, 32)
		return &s
	},
}

var funcsSnapshotPool = sync.Pool{
	New: func() any {
		s := make([]*ssa.Function, 0, 32)
		return &s
	},
}

// GetReachable returns the set of reachable functions.
// This converts from the internal xsync.Map to a regular map.
func (r *Result) GetReachable() map[*ssa.Function]struct{ AddrTaken bool } {
	reachableMap := make(map[*ssa.Function]struct{ AddrTaken bool })
	r.reachable.Range(func(fn *ssa.Function, addrTaken bool) bool {
		reachableMap[fn] = struct{ AddrTaken bool }{AddrTaken: addrTaken}
		return true
	})
	return reachableMap
}

// GetRuntimeTypes returns the set of runtime types.
func (r *Result) GetRuntimeTypes() typeutil.Map {
	// Convert utils.TypeMap to typeutil.Map
	var tm typeutil.Map
	tm.SetHasher(typeutil.MakeHasher())
	r.runtimeTypes.Iterate(func(key types.Type, value any) {
		tm.Set(key, value)
	})
	return tm
}

// Materialize converts the result to the standard rta.Result.
// For parallel implementation, this performs the conversion from xsync.Map to regular map.
func (r *Result) Materialize() *rtapkg.Result {
	return &rtapkg.Result{
		CallGraph:    r.GetCallGraph(),
		Reachable:    r.GetReachable(),
		RuntimeTypes: r.GetRuntimeTypes(),
	}
}

// MCBucketSizePercentiles returns percentiles for MC bucket sizes (lazy computed).
func (r *Result) MCBucketSizePercentiles() rtalib.Percentiles {
	r.mcBucketOnce.Do(func() {
		var sizes []int
		r.mcMap.Range(func(_ string, ctypeSet *concretePtrSet) bool {
			sizes = append(sizes, ctypeSet.Size())
			return true
		})
		r.mcBucketPercentiles = rtalib.ComputePercentiles(sizes)
	})
	return r.mcBucketPercentiles
}

// MIBucketSizePercentiles returns percentiles for MI bucket sizes (lazy computed).
func (r *Result) MIBucketSizePercentiles() rtalib.Percentiles {
	r.miBucketOnce.Do(func() {
		var sizes []int
		r.miMap.Range(func(_ string, itypeSet *interfacePtrSet) bool {
			sizes = append(sizes, itypeSet.Size())
			return true
		})
		r.miBucketPercentiles = rtalib.ComputePercentiles(sizes)
	})
	return r.miBucketPercentiles
}

// InterfaceMethodCountBuckets returns the fraction of interfaces in each
// method-count bucket: [1, 2, 4, 8, 16, 32, 64, >64].
func (r *Result) InterfaceMethodCountBuckets() [8]float64 {
	r.ifaceMethodBucketsOnce.Do(func() {
		var counts [8]int
		total := 0
		r.interfaceTypesMap.Iterate(func(_ types.Type, v any) {
			iinfo := v.(*interfaceTypeInfo)
			total++
			n := iinfo.mset.Len()
			switch {
			case n <= 1:
				counts[0]++
			case n <= 2:
				counts[1]++
			case n <= 4:
				counts[2]++
			case n <= 8:
				counts[3]++
			case n <= 16:
				counts[4]++
			case n <= 32:
				counts[5]++
			case n <= 64:
				counts[6]++
			default:
				counts[7]++
			}
		})
		if total > 0 {
			for i, c := range counts {
				r.ifaceMethodBuckets[i] = float64(c) / float64(total)
			}
		}
	})
	return r.ifaceMethodBuckets
}

type workItem interface {
	doWork(r *rta, workerID int)
}

type processFunctionWorkItem struct {
	fn *ssa.Function
}

func (w *processFunctionWorkItem) doWork(r *rta, workerID int) {
	r.visitFunc(w.fn, workerID)
}

// Working state of the RTA algorithm.
type rta struct {
	result *Result

	prog *ssa.Program

	reflectValueCall *ssa.Function // (*reflect.Value).Call, iff part of prog

	// pool drives the parallel RTA phase: per-worker chaselev deques with
	// random-victim work-stealing and sleeper-counted wake. See
	// utils.WorkerPool.
	pool *utils.WorkerPool[workItem]

	// sigDataBySig holds the address-taken-funcs × dyncall-sites
	// matrix grouped by call signature. Each signature owns one
	// *sigData with its own mutex; both visitAddrTakenFunc and
	// visitDynCall acquire that mutex briefly to publish-and-snapshot
	// atomically. This eliminates the
	// xsync.Map.Store/Range visibility race that the old separate
	// addrTakenFuncsBySig + dynCallSites pair was vulnerable to,
	// where both paths could see each other's freshly-stored entry and
	// each emit the same (site, f) edge — the source of every
	// `rta_duplicated_edges_*` count > 0 in the prior implementation.
	// Eliminating that source removes the need for the post-RTA dedup
	// pass entirely (~3s of CPU on u4b-token alone), without
	// introducing a separate single-worker fast path.
	sigDataBySig utils.TypeMap

	// invoke sites are grouped per (interface, method) by
	// interfaceTypeInfo.sitesByMethod; no flat per-interface list is kept.

	// The following two maps together define the subset of the
	// m:n "implements" relation needed by the algorithm.

	// concreteTypes maps each concrete type to information about it.
	// Keys are types.Type, values are *concreteTypeInfo.
	// Only concrete types used as MakeInterface operands are included.
	concreteTypes utils.TypeMap

	// interfaceTypes maps each interface type to information about it.
	// Keys are *types.Interface, values are *interfaceTypeInfo.
	// Only interfaces used in "invoke"-mode CallInstructions are included.
	interfaceTypes utils.TypeMap

	// Kumo optimization: method-based indexing to avoid O(n*m) iterations
	// MC maps method names to sets of concrete types that have that method.
	// Inner buckets are *concretePtrSet (sync.Mutex + Go map[*concreteTypeInfo]struct{})
	// rather than xsync.Map; the latter cost ~4.5 KB even when empty, and
	// per-method-name there are tens of thousands of buckets.
	MC *xsync.Map[string, *concretePtrSet]

	// MI maps method names to sets of interface types that require that method.
	// Same inner-bucket rationale as MC.
	MI *xsync.Map[string, *interfacePtrSet]

	// methodCounts tracks how many interfaces use each method (for RarestMethodByInterfaceType)
	methodCounts *xsync.Map[string, *atomic.Int32]

	strategy rtalib.MethodSelectionStrategy

	// Method value cache, to avoid lock contention
	methodValueCache *xsync.Map[utils.MethodKey, *ssa.Function]

	// methodSetCache replaces prog.MethodSets (typeutil.MethodSetCache) for the
	// three call sites in this flavor that need a full *types.MethodSet:
	// interfaces(), implementations(), addRuntimeType(). The upstream cache
	// holds a single sync.Mutex around the entire compute, which dominates
	// scaling above ~16 workers; this xsync.Map is sharded and the cold compute
	// runs outside any global lock (LoadOrCompute serializes only same-key races).
	methodSetCache *xsync.Map[types.Type, *types.MethodSet]

	rtalib.WorkerCounters
}

type concreteTypeInfo struct {
	C      types.Type
	mset   *types.MethodSet
	fprint uint64 // fingerprint of method set

	// implements holds the set of interfaces C is known to implement.
	// Keys are iinfo.I (canonical *types.Interface per Identical class),
	// so pointer-identity dedup is sound. Replaces utils.TypeMap which
	// allocates a ~4.5 KB xsync.Map on first Set.
	implements typeSet

	// initMu is held by the goroutine that won the concreteTypes
	// LoadOrStore race while it populates `implements`. Consumers that
	// need a fully-initialized cinfo wait by Lock/Unlock. Replaces a
	// `chan struct{}` (whose backing hchan was ~96 B per cinfo) with a
	// single inline sync.Mutex (8 B).
	initMu sync.Mutex
}

type interfaceTypeInfo struct {
	I      *types.Interface
	mset   *types.MethodSet
	fprint uint64

	// implementations holds the set of concrete types known to implement I.
	// Keys are cinfo.C (canonical types.Type per Identical class).
	implementations typeSet

	// sitesByMethod groups invoke-site instructions at I by method
	// name. Plain Go map + sync.RWMutex; nil/empty until the first
	// visitInvoke at I. Was *xsync.Map allocated eagerly at iinfo
	// creation (~4.5 KB per iinfo, dominant on services with many
	// interfaces).
	sitesByMethod methodSitesMap

	// Init barrier; see concreteTypeInfo.initMu.
	initMu sync.Mutex
}

// methodSitesEntry tracks invoke sites at (I, method.Name()) plus the
// pre-resolved concrete-method functions for each impl of I.
//
// mu serializes mutations to sites/cmethods across workers. Lock
// contention per entry is bounded: in steady state only addRuntimeType
// (per new C implementing I) and visitInvoke (per new site at this
// method on I) touch a given entry, and the body inside the lock is
// small.
//
// cmethods dedup uses a dynamic strategy: linear scan while cmethods is
// small (below cmethodsLinearThreshold), then switch to cmethodSet
// (Go map) for O(1) lookup once the slice exceeds the threshold. This
// preserves O(n) total insertion cost for hot interfaces with many
// impls while avoiding the ~48 B+ map allocation per entry in the
// common case (most entries have <10 cmethods). With ~1M
// methodSitesEntry instances on large services, the saved map
// allocations add up.
//
// Two paths can append the same cmethod concurrently:
//  1. visitInvoke "first-time" snapshot of iinfo.implementations may
//     include a concrete C whose addRuntimeType is racing to also
//     append cmethod_C_m.
//  2. visitInvoke first-time may overlap with addRuntimeType for an
//     earlier C — the lock serializes them, but the snapshot is
//     unsynchronized.
//
// addUniqueCmethod ensures each cmethod appears in cmethods exactly
// once, so visitInvoke "seen-method" fast path doesn't emit duplicate
// edges that would later need to be removed by the dedup pass.
type methodSitesEntry struct {
	method   *types.Func
	mu       sync.Mutex
	sites    []ssa.CallInstruction
	cmethods []*ssa.Function
	// cmethodSet is nil while cmethods is small (< cmethodsLinearThreshold);
	// populated lazily once cmethods grows beyond that point.
	cmethodSet map[*ssa.Function]struct{}
	// initialized is set true once visitInvoke "first-time" has built
	// cmethods from the initial snapshot of iinfo.implementations.
	initialized bool
}

// cmethodsLinearThreshold picks the crossover between linear-scan and
// map-based dedup for methodSitesEntry.cmethods. Below the threshold,
// scan is faster than a map probe (no hash, fewer cache misses) and
// avoids the per-entry map allocation. The threshold sits at the size
// where map lookup begins to beat linear scan on amortized cost; the
// exact value isn't sensitive.
const cmethodsLinearThreshold = 16

// addUniqueCmethod appends fn to entry.cmethods iff not already present.
// Returns true if newly inserted. Caller must hold entry.mu.
//
// Linear scan while cmethods is small; switches to a map once the slice
// exceeds cmethodsLinearThreshold. The map is allocated only when needed,
// so entries that stay small never pay the map cost.
func (entry *methodSitesEntry) addUniqueCmethod(fn *ssa.Function) bool {
	if entry.cmethodSet != nil {
		if _, dup := entry.cmethodSet[fn]; dup {
			return false
		}
		entry.cmethodSet[fn] = struct{}{}
		entry.cmethods = append(entry.cmethods, fn)
		return true
	}
	for _, c := range entry.cmethods {
		if c == fn {
			return false
		}
	}
	entry.cmethods = append(entry.cmethods, fn)
	if len(entry.cmethods) >= cmethodsLinearThreshold {
		entry.cmethodSet = make(map[*ssa.Function]struct{}, len(entry.cmethods)*2)
		for _, c := range entry.cmethods {
			entry.cmethodSet[c] = struct{}{}
		}
	}
	return true
}

// sigData holds the address-taken-funcs and dyncall-sites observed so
// far for one call signature. The single mutex guards both sets so
// that visitAddrTakenFunc and visitDynCall can publish their own side
// and snapshot the other side atomically, eliminating the cross-race
// that would otherwise produce duplicate (site, f) edges in the
// callgraph.
type sigData struct {
	mu    sync.Mutex
	funcs map[*ssa.Function]struct{}
	sites map[ssa.CallInstruction]struct{}
}

func newSigData() *sigData {
	return &sigData{
		funcs: make(map[*ssa.Function]struct{}),
		sites: make(map[ssa.CallInstruction]struct{}),
	}
}

// addReachable marks a function as potentially callable at run-time,
// and ensures that it gets processed.
//
// workerID is the caller's worker ID (so newly-discovered work lands in the
// caller's local queue with zero contention), or -1 if called from outside
// the worker pool (initial roots, processNodesInBatches submissions).
func (r *rta) addReachable(f *ssa.Function, addrTaken bool, workerID int) {
	existing, loaded := r.result.reachable.LoadOrStore(f, addrTaken)
	if loaded && addrTaken && !existing {
		// Need to update existing entry to set addr taken to true
		r.result.reachable.Store(f, true)
	}
	if !loaded {
		// First time seeing f.  Add it to the worklist.
		r.addFunctionToWorklist(f, workerID)
	}
}

// addToWorklist enqueues item via the worker pool. workerID >= 0 routes the
// item to the caller's own deque (single-writer fast path); workerID < 0
// means "called from outside the worker pool" — the item lands in the pool's
// mainbox, which workers steal from.
//
// The worker pool calls workWg.Add internally on each Push and Done after
// the item's doWork returns.
func (r *rta) addToWorklist(item workItem, workerID int) {
	r.pool.Push(item, workerID)
}

func (r *rta) addFunctionToWorklist(f *ssa.Function, workerID int) {
	r.addToWorklist(&processFunctionWorkItem{fn: f}, workerID)
}

// addEdge adds the specified call graph edge, and marks it reachable.
// addrTaken indicates whether to mark the callee as "address-taken".
// site is nil for calls made via reflection.
//
// Edges are appended directly to e.Callee.In and e.Caller.Out via per-node
// sync.Mutex inside ConcurrentCallGraph.AddCallGraphEdge. An alternative
// design — routing edges through a per-(worker, partition) bucket matrix
// followed by a post-RTA merge phase — was tried and rejected because the
// merge phase turned out to cost more CPU than the per-node mutex
// contention it was meant to eliminate, due to cache-cold writes to the
// ~150+ MB of node.In/Out slice headers in large callgraphs.
func (r *rta) addEdge(caller *ssa.Function, site ssa.CallInstruction, callee *ssa.Function, addrTaken bool, workerID int) {
	r.addReachable(callee, addrTaken, workerID)

	if g := r.result.callGraph; g != nil {
		if caller == nil {
			panic(site)
		}

		// Fast variant: obtain the per-node locks alongside the node
		// pointers in a single xsync.Map.Load each, then pass them
		// straight to AddCallGraphEdgeFast. This skips the two
		// xsync.Map.Load operations the basic AddCallGraphEdge would
		// otherwise perform per call to rediscover the locks — on
		// services with tens of millions of static/dynamic edges that
		// reclaims several seconds of CPU.
		from, fromLock := g.CreateNodeWithLock(caller)
		to, toLock := g.CreateNodeWithLock(callee)
		g.AddCallGraphEdgeFast(from, fromLock, site, to, toLock)
	}
}

// ---------- addrTakenFuncs × dynCallSites ----------

// visitAddrTakenFunc is called each time we encounter an address-taken function f.
//
// Per-signature publish-and-snapshot under one mutex: we acquire
// sd.mu, mark f as published, and (if first time) snapshot the
// sites currently published for the same signature. Only after
// releasing the mutex do we attach edges to those snapshotted sites.
// Combined with the symmetric protocol in visitDynCall, this
// structurally prevents the duplicate-edge race in the previous
// xsync.Map-based design — every (site, f) pair is attached by
// exactly one of the two paths, namely whichever one acquires its
// mutex AFTER the other has already published (its snapshot then
// includes the earlier publication, the earlier snapshot didn't see
// the later publication). No races, no missed edges, no need for a
// post-RTA dedup pass.
func (r *rta) visitAddrTakenFunc(f *ssa.Function, workerID int) {
	S := f.Signature

	val, _ := r.sigDataBySig.LoadOrStore(S, newSigData())
	sd := val.(*sigData)

	var (
		sitesSnapshot []ssa.CallInstruction
		sitesPtr      *[]ssa.CallInstruction
		firstTime     bool
	)
	sd.mu.Lock()
	if _, seen := sd.funcs[f]; !seen {
		firstTime = true
		sd.funcs[f] = struct{}{}
		if len(sd.sites) > 0 {
			sitesPtr = sitesSnapshotPool.Get().(*[]ssa.CallInstruction)
			sitesSnapshot = (*sitesPtr)[:0]
			for site := range sd.sites {
				sitesSnapshot = append(sitesSnapshot, site)
			}
		}
	}
	sd.mu.Unlock()

	for _, site := range sitesSnapshot {
		r.addEdge(site.Parent(), site, f, true, workerID)
	}
	if sitesPtr != nil {
		*sitesPtr = sitesSnapshot[:0]
		sitesSnapshotPool.Put(sitesPtr)
	}

	// If the program includes (*reflect.Value).Call,
	// add a dynamic call edge from it to any address-taken
	// function, regardless of signature.
	//
	// This isn't perfect.
	// - The actual call comes from an internal function
	//   called reflect.call, but we can't rely on that here.
	// - reflect.Value.CallSlice behaves similarly,
	//   but we don't bother to create callgraph edges from
	//   it as well as it wouldn't fundamentally change the
	//   reachability but it would add a bunch more edges.
	// - We assume that if reflect.Value.Call is among
	//   the dependencies of the application, it is itself
	//   reachable. (It would be more accurate to defer
	//   all the addEdges below until r.V.Call itself
	//   becomes reachable.)
	// - Fake call graph edges are added from r.V.Call to
	//   each address-taken function, but not to every
	//   method reachable through a materialized rtype,
	//   which is a little inconsistent. Still, the
	//   reachable set includes both kinds, which is what
	//   matters for e.g. deadcode detection.)
	if firstTime && r.reflectValueCall != nil {
		var site ssa.CallInstruction = nil // can't find actual call site
		r.addEdge(r.reflectValueCall, site, f, true, workerID)
	}
}

// visitDynCall is called each time we encounter a dynamic "call"-mode call.
//
// Symmetric counterpart to visitAddrTakenFunc — see that function's
// docstring for the race-free publish-and-snapshot protocol.
func (r *rta) visitDynCall(site ssa.CallInstruction, workerID int) {
	S := site.Common().Signature()

	val, _ := r.sigDataBySig.LoadOrStore(S, newSigData())
	sd := val.(*sigData)

	// A site belongs to exactly one ssa.Function and each function's
	// instructions are visited once by visitFunc, so visitDynCall(site)
	// is called exactly once per site. The `_, seen` check is therefore
	// always false in practice, but kept for defensive symmetry with
	// visitAddrTakenFunc: should a future caller invoke us repeatedly,
	// we will not double-emit edges.
	var (
		funcsSnapshot []*ssa.Function
		funcsPtr      *[]*ssa.Function
	)
	sd.mu.Lock()
	if _, seen := sd.sites[site]; !seen {
		sd.sites[site] = struct{}{}
		if len(sd.funcs) > 0 {
			funcsPtr = funcsSnapshotPool.Get().(*[]*ssa.Function)
			funcsSnapshot = (*funcsPtr)[:0]
			for f := range sd.funcs {
				funcsSnapshot = append(funcsSnapshot, f)
			}
		}
	}
	sd.mu.Unlock()

	for _, f := range funcsSnapshot {
		r.addEdge(site.Parent(), site, f, true, workerID)
	}
	if funcsPtr != nil {
		*funcsPtr = funcsSnapshot[:0]
		funcsSnapshotPool.Put(funcsPtr)
	}
}

// ---------- concrete types × invoke sites ----------

// addInvokeEdge is called for each new pair (site, C) in the matrix.
func (r *rta) addInvokeEdge(site ssa.CallInstruction, C types.Type, workerID int) {
	// Ascertain the concrete method of C to be called.
	imethod := site.Common().Method
	cmethod := utils.LookupMethod(r.prog, r.methodValueCache, C, imethod.Pkg(), imethod.Name())
	r.addEdge(site.Parent(), site, cmethod, true, workerID)
}

// visitInvoke is called each time the algorithm encounters an
// "invoke"-mode call site. Invoke sites are grouped by (I, method)
// under iinfo.sitesByMethod. The first site at (I, methodName) builds
// a cmethods cache (one *ssa.Function per impl C, after LookupMethod
// + addReachable). Subsequent sites for the same method skip both
// LookupMethod (already in cache) and addReachable (already done);
// they only attach a callgraph edge per cached cmethod. On services
// with hot interfaces (many sites sharing few methods, e.g. thriftrw
// Writer with 80K+ sites across ~20 methods), this collapses an inner
// loop that used to be O(impls) per site into the same loop body
// without LookupMethod or addReachable overhead — typically a 2x
// wall-clock improvement on 1 worker, and reduces methodValueCache /
// reachable map contention at higher worker counts.
func (r *rta) visitInvoke(site ssa.CallInstruction, workerID int) {
	I := site.Common().Value.Type().Underlying().(*types.Interface)
	imethod := site.Common().Method

	// Ensure iinfo exists and its initial implementations are populated.
	iinfo := r.implementations(I, workerID)
	iinfo.initMu.Lock()
	iinfo.initMu.Unlock()

	// Get (or create) the per-method entry.
	methodName := imethod.Name()
	entry := iinfo.sitesByMethod.LoadOrCompute(methodName, func() *methodSitesEntry {
		return &methodSitesEntry{method: imethod}
	})

	entry.mu.Lock()
	entry.sites = append(entry.sites, site)

	if !entry.initialized {
		// First site for (I, methodName) — build cmethods cache from
		// the current set of impls of I. Pre-size cmethods to the
		// current impls count to skip the early slice doublings; this
		// is the parallel-version analog of pre-sizing
		// iinfo.implementations in srta_kumo. addRuntimeType may later
		// extend cmethods for new impls discovered after this point.
		entry.initialized = true
		implCount := iinfo.implementations.Len()
		if implCount > 0 && cap(entry.cmethods) < implCount {
			entry.cmethods = make([]*ssa.Function, 0, implCount)
		}
		// One utils.LookupMethod call per impl in the Iterate callback
		// below. Counted once via TypeMap.Len(); per-worker counter
		// avoids any atomic Inc inside the callback. Subsequent visits
		// at this (I, methodName) reuse entry.cmethods and do zero
		// lookups, so the increment lives only on this first-time path.
		r.InvokeLookupsFromVisitInvokeCounts.Add(workerID, implCount)
		iinfo.implementations.Iterate(func(C types.Type) {
			cmethod := utils.LookupMethod(r.prog, r.methodValueCache, C, imethod.Pkg(), methodName)
			if cmethod == nil {
				return
			}
			if !entry.addUniqueCmethod(cmethod) {
				return
			}
			r.addReachable(cmethod, true, workerID)
		})
	}

	// Attach edges from this site to every cached cmethod. (For the
	// first-time path above, addReachable was just done; for
	// subsequent sites, cmethods covers all impls reachable so far —
	// addRuntimeType extends cmethods when new impls of I arrive.)
	cmethodsSnapshot := entry.cmethods
	entry.mu.Unlock()

	// No LookupMethod call happens past this point on either path
	// (first-time already counted above; subsequent reuses cmethods
	// directly). Just attach edges; no counter increment.

	if g := r.result.callGraph; g != nil && len(cmethodsSnapshot) > 0 {
		from, fromLock := g.CreateNodeWithLock(site.Parent())
		for _, cmethod := range cmethodsSnapshot {
			to, toLock := g.CreateNodeWithLock(cmethod)
			g.AddCallGraphEdgeFast(from, fromLock, site, to, toLock)
		}
	}
}

// ---------- main algorithm ----------

// visitFunc processes function f.
func (r *rta) visitFunc(f *ssa.Function, workerID int) {
	var space [32]*ssa.Value // preallocate space for common case

	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			rands := instr.Operands(space[:0])

			switch instr := instr.(type) {
			case ssa.CallInstruction:
				call := instr.Common()
				if call.IsInvoke() {
					r.InvokeCallSiteCounts.Inc(workerID)
					r.visitInvoke(instr, workerID)
				} else if g := call.StaticCallee(); g != nil {
					r.StaticCallSiteCounts.Inc(workerID)
					r.addEdge(f, instr, g, false, workerID)
				} else if _, ok := call.Value.(*ssa.Builtin); !ok {
					r.IndirectCallSiteCounts.Inc(workerID)
					r.visitDynCall(instr, workerID)
				}

				// Ignore the call-position operand when
				// looking for address-taken Functions.
				// Hack: assume this is rands[0].
				rands = rands[1:]

			case *ssa.MakeInterface:
				// Converting a value of type T to an
				// interface materializes its runtime
				// type, allowing any of its exported
				// methods to be called though reflection.
				r.addRuntimeType(instr.X.Type(), false, workerID)
			}

			// Process all address-taken functions.
			for _, op := range rands {
				if g, ok := (*op).(*ssa.Function); ok {
					r.visitAddrTakenFunc(g, workerID)
				}
			}
		}
	}
}

// Analyze performs Rapid Type Analysis, starting at the specified root
// functions.  It returns nil if no roots were specified.
//
// The root functions must be one or more entrypoints (main and init
// functions) of a complete SSA program, with function bodies for all
// dependencies, constructed with the [ssa.InstantiateGenerics] mode
// flag.
//
// If buildCallGraph is true, Result.CallGraph will contain a call
// graph; otherwise, only the other fields (reachable functions) are
// populated.

// RTA is an RTA implementation with method indexing and lock-free callgraph.
type RTA struct{}

// New creates a new RTA instance.
func New() *RTA {
	return &RTA{}
}

func (a *RTA) Analyze(roots []*ssa.Function, buildCallGraph bool, opts ...rtalib.AnalyzeOption) rtalib.Result {

	if len(roots) == 0 {
		return nil
	}

	cfg := rtalib.DefaultAnalyzeConfig(opts...)
	numWorkers := cfg.NumWorkers
	if numWorkers <= 0 {
		numWorkers = 1
	}

	r := &rta{
		result: &Result{
			reachable: xsync.NewMap[*ssa.Function, bool](),
		},
		prog:             roots[0].Prog,
		pool:             utils.NewWorkerPool[workItem](numWorkers),
		MC:               xsync.NewMap[string, *concretePtrSet](),
		MI:               xsync.NewMap[string, *interfacePtrSet](),
		methodCounts:     xsync.NewMap[string, *atomic.Int32](),
		strategy:         cfg.MethodSelectionStrategy,
		methodValueCache: xsync.NewMap[utils.MethodKey, *ssa.Function](),
		methodSetCache:   xsync.NewMap[types.Type, *types.MethodSet](),
		WorkerCounters:   rtalib.NewWorkerCounters(numWorkers),
	}

	if buildCallGraph {
		r.result.callGraph = utils.NewConcurrentCallGraph(roots[0])
	}

	// Grab ssa.Function for (*reflect.Value).Call,
	// if "reflect" is among the dependencies.
	if reflectPkg := r.prog.ImportedPackage("reflect"); reflectPkg != nil {
		reflectValue := reflectPkg.Members["Value"].(*ssa.Type)
		r.reflectValueCall = utils.LookupMethod(r.prog, r.methodValueCache, reflectValue.Object().Type(), reflectPkg.Pkg, "Call")
	}

	hasher := typeutil.MakeHasher()
	r.result.runtimeTypes.SetHasher(hasher)
	r.sigDataBySig.SetHasher(hasher)
	r.concreteTypes.SetHasher(hasher)
	r.interfaceTypes.SetHasher(hasher)

	r.pool.Start(func(item workItem, workerID int) {
		item.doWork(r, workerID)
	})

	// Add the initial work via the pool's mainbox; workers steal from it.
	for _, root := range roots {
		r.addReachable(root, false, -1)
	}

	analysisStart := time.Now()

	// Capture baseline heap for delta computation
	var baselineMem runtime.MemStats
	runtime.ReadMemStats(&baselineMem)
	baselineHeapInuse := baselineMem.HeapInuse

	// Wait for RTA-related work to complete
	r.pool.Wait()

	// Edges are attached directly to node.In / node.Out via per-node mutex
	// inside g.AddCallGraphEdge during the parallel phase; no merge needed.
	//
	// No dedup phase: every (caller, site, callee) triple is emitted by
	// exactly one path now. The dyncall × addrTakenFunc race that used
	// to produce duplicates is structurally prevented by sigData's
	// per-signature mutex + atomic publish-and-snapshot (see
	// visitAddrTakenFunc / visitDynCall). Invoke edges are deduped at
	// the cmethod level by methodSitesEntry.cmethodSet. Static and
	// reflect-call edges are emitted once per visitFunc / first-time
	// addReachable gate. We keep the PreDedup/PostDedup metric fields
	// for API compatibility but report equal values: post = pre, and
	// DuplicatedEdgeCount is always 0.
	preDedupDuration := time.Since(analysisStart)
	var preDedupMem runtime.MemStats
	runtime.ReadMemStats(&preDedupMem)

	postDedupDuration := preDedupDuration
	postDedupMem := preDedupMem

	// Signal workers to shut down
	r.pool.Stop()

	// Sum per-worker counters
	r.WorkerCounters.MergeInto(&r.result.BaseMetrics)

	// Type counts
	r.result.NumConcreteTypesVal = r.concreteTypes.Len()
	r.result.NumInterfaceTypesVal = r.interfaceTypes.Len()

	// Dedup timing
	r.result.PreDedupDurationMsVal = float64(preDedupDuration.Milliseconds())
	r.result.PostDedupDurationMsVal = float64(postDedupDuration.Milliseconds())
	r.result.PreDedupHeapDeltaMBVal = float64(max(int64(preDedupMem.HeapInuse)-int64(baselineHeapInuse), 0)) / (1024 * 1024)
	r.result.PostDedupHeapDeltaMBVal = float64(max(int64(postDedupMem.HeapInuse)-int64(baselineHeapInuse), 0)) / (1024 * 1024)

	// Store raw data for lazy index stats
	r.result.mcMap = r.MC
	r.result.miMap = r.MI
	r.result.interfaceTypesMap = &r.interfaceTypes

	return r.result
}

// interfaces() and implementations()
//
// In both of these methods, if we are encoutering the concrete type/interface
// for the first time, we broadcast the existence of a
// concreteTypeInfo/interfaceTypeInfo (via a LoadOrStore) and only then
// populate `implements`/`implementations` fields (via Range). This order
// ensures that no work item gets dropped; either a consumer of `interfaces()`
// will end up adding it, or a consumer of `implementations()` will add it.

// methodSet returns the *types.MethodSet for T using a per-RTA xsync.Map cache,
// avoiding the global mutex inside prog.MethodSets (typeutil.MethodSetCache).
// The cold compute (types.NewMethodSet) is concurrent-safe for fully-checked
// types, which is what RTA operates on.
func (r *rta) methodSet(T types.Type) *types.MethodSet {
	mset, _ := r.methodSetCache.LoadOrCompute(T, func() (*types.MethodSet, bool) {
		return types.NewMethodSet(T), false
	})
	return mset
}

// interfaces(C) returns all currently known interfaces implemented by C.
func (r *rta) interfaces(C types.Type, workerID int) *concreteTypeInfo {
	switch C.(type) {
	case *types.Tuple,
		*types.Array,
		*types.Slice,
		*types.Chan,
		*types.Signature,
		*types.Map:
		// These types have no methods. Return a fully-initialized stub
		// (no init barrier needed: implements is empty and stays empty).
		return &concreteTypeInfo{C: C}
	}

	// Create an info for C the first time we see it.
	var cinfo *concreteTypeInfo
	if v := r.concreteTypes.At(C); v != nil {
		cinfo = v.(*concreteTypeInfo)
	} else {
		mset := r.methodSet(C)
		cinfo = &concreteTypeInfo{
			C:      C,
			mset:   mset,
			fprint: utils.Fingerprint(mset),
		}
		cinfo.initMu.Lock() // released after `implements` is populated
		if oldcinfo, loaded := r.concreteTypes.LoadOrStore(C, cinfo); !loaded {
			// Kumo optimization: Index this concrete type under each of its methods
			for i := 0; i < mset.Len(); i++ {
				methodName := mset.At(i).Obj().Name()
				ctypeSet, _ := r.MC.LoadOrCompute(methodName, func() (*concretePtrSet, bool) {
					return &concretePtrSet{}, false
				})
				ctypeSet.Store(cinfo)
			}

			// Kumo optimization: Find candidate interfaces by looking up methods
			// instead of iterating all interfaces
			candidates := make(map[*interfaceTypeInfo]bool)
			for i := 0; i < mset.Len(); i++ {
				methodName := mset.At(i).Obj().Name()
				if itypeSet, ok := r.MI.Load(methodName); ok {
					itypeSet.Range(func(iinfo *interfaceTypeInfo) bool {
						candidates[iinfo] = true
						return true
					})
				}
			}

			// Ascertain set of interfaces C implements
			// and update the 'implements' relation.
			for iinfo := range candidates {
				r.ChecksFromInterfacesCounts.Inc(workerID)
				if I := types.Unalias(iinfo.I).(*types.Interface); r.implements(cinfo, iinfo, workerID) {
					iinfo.implementations.Set(C)
					cinfo.implements.Set(I)
					r.SuccessFromInterfacesCounts.Inc(workerID)
				} else {
					r.FailsFromInterfacesCounts.Inc(workerID)
				}
			}

			cinfo.initMu.Unlock()
		} else {
			// Lost the race; our stub cinfo (still init-locked) is GC'd.
			// initMu was held by us only, so no leak.
			return oldcinfo.(*concreteTypeInfo)
		}
	}
	return cinfo
}

// implementations(I) returns all currently known concrete types that implement I.
func (r *rta) implementations(I *types.Interface, workerID int) *interfaceTypeInfo {
	// Create an info for I the first time we see it.
	var iinfo *interfaceTypeInfo
	if v := r.interfaceTypes.At(I); v != nil {
		iinfo = v.(*interfaceTypeInfo)
	} else {
		mset := r.methodSet(I)
		iinfo = &interfaceTypeInfo{
			I:      I,
			mset:   mset,
			fprint: utils.Fingerprint(mset),
		}
		iinfo.initMu.Lock() // released after `implementations` is populated
		if oldiinfo, loaded := r.interfaceTypes.LoadOrStore(I, iinfo); !loaded {
			// Kumo optimization: MI indexing with configurable method selection strategy.
			var selectedMethod string
			switch r.strategy {
			case rtalib.RandomMethodStrategy:
				if mset.Len() > 0 {
					selectedMethod = mset.At(r.pool.WorkerRand(workerID).Intn(mset.Len())).Obj().Name()
				}
			case rtalib.RarestMethodByInterfaceType:
				var rarestCount int32 = 1<<31 - 1
				for i := 0; i < mset.Len(); i++ {
					methodName := mset.At(i).Obj().Name()
					counter, _ := r.methodCounts.LoadOrCompute(methodName, func() (*atomic.Int32, bool) {
						return new(atomic.Int32), false
					})
					ifaceCount := counter.Add(1)
					if ifaceCount < rarestCount {
						rarestCount = ifaceCount
						selectedMethod = methodName
					}
				}
			default: // RarestMethodByConcreteType
				if mset.Len() == 1 {
					selectedMethod = mset.At(0).Obj().Name()
				} else {
					var bestMCSize int = 1<<31 - 1
					for i := 0; i < mset.Len(); i++ {
						methodName := mset.At(i).Obj().Name()
						mcSize := 0
						if ctypeSet, ok := r.MC.Load(methodName); ok {
							mcSize = ctypeSet.Size()
						}
						if mcSize < bestMCSize {
							bestMCSize = mcSize
							selectedMethod = methodName
						}
					}
				}
			}

			// Index interface under selected method in MI
			if selectedMethod != "" {
				itypeSet, _ := r.MI.LoadOrCompute(selectedMethod, func() (*interfacePtrSet, bool) {
					return &interfacePtrSet{}, false
				})
				itypeSet.Store(iinfo)
			}

			// Kumo optimization: Find method with fewest concrete types
			// and only check those candidates
			var bestMethod string
			var bestSize int = 1<<31 - 1 // max int
			noCandidates := false

			for i := 0; i < mset.Len(); i++ {
				methodName := mset.At(i).Obj().Name()
				if ctypeSet, ok := r.MC.Load(methodName); ok {
					size := ctypeSet.Size()
					if size < bestSize {
						bestSize = size
						bestMethod = methodName
					}
				} else {
					// No concrete type has this method, so no concrete type
					// can implement this interface. Skip the scan entirely.
					noCandidates = true
					break
				}
				if bestSize <= mset.Len()-(i+1) {
					break
				}
			}

			// Ascertain set of concrete types that implement I
			// and update the 'implements' relation.
			if bestMethod != "" && !noCandidates {
				if ctypeSet, ok := r.MC.Load(bestMethod); ok {
					ctypeSet.Range(func(cinfo *concreteTypeInfo) bool {
						r.ChecksFromImplementationsCounts.Inc(workerID)
						if r.implements(cinfo, iinfo, workerID) {
							cinfo.implements.Set(I)
							iinfo.implementations.Set(cinfo.C)
							r.SuccessFromImplementationsCounts.Inc(workerID)
						} else {
							r.FailsFromImplementationsCounts.Inc(workerID)
						}
						return true
					})
				}
			}

			iinfo.initMu.Unlock()
		} else {
			return oldiinfo.(*interfaceTypeInfo)
		}
	}
	return iinfo
}

// addRuntimeType is called for each concrete type that can be the
// dynamic type of some interface or reflect.Value.
// Adapted from needMethods in go/ssa/builder.go
func (r *rta) addRuntimeType(T types.Type, skip bool, workerID int) {
	// Never record aliases.
	T = types.Unalias(T)

	if prev, loaded := r.result.runtimeTypes.LoadOrStore(T, skip); loaded {
		// Type already exists, update if we need to change skip from true to false
		// `skip` only moves from true to false, not the other way around.
		if !skip && prev.(bool) {
			r.result.runtimeTypes.Set(T, skip)
		}
		return
	}
	// Type was newly stored, continue with processing

	mset := r.methodSet(T)

	if _, ok := T.Underlying().(*types.Interface); !ok {
		// T is a new concrete type.
		for i, n := 0, mset.Len(); i < n; i++ {
			sel := mset.At(i)
			m := sel.Obj()

			if m.Exported() {
				// Exported methods are always potentially callable via reflection.
				r.addReachable(utils.MethodValueFast(r.prog, sel), true, workerID)
			}
		}

		// Add callgraph edges for each existing dynamic "invoke"-mode
		// call via every interface T implements. Iterate
		// iinfo.sitesByMethod (per-method groups). For each (I, method)
		// seen so far, do one LookupMethod + addReachable for T and then
		// attach edges from every site in the group. Also extend
		// entry.cmethods so subsequent visitInvoke calls at this
		// (I, method) can skip LookupMethod entirely.
		cinfo := r.interfaces(T, workerID)

		// Wait for initialization to complete.
		cinfo.initMu.Lock()
		cinfo.initMu.Unlock()

		cinfo.implements.Iterate(func(k types.Type) {
			I := k.(*types.Interface)
			iinfo, _ := r.interfaceTypes.At(I).(*interfaceTypeInfo)
			if iinfo == nil {
				return
			}
			g := r.result.callGraph
			// Snapshot under sitesByMethod RLock so concurrent
			// visitInvoke writers aren't blocked while we iterate.
			// Reuse pooled scratch (per-P) to avoid the snapshot alloc.
			scratch := sitesEntriesScratchPool.Get().(*sitesEntriesScratch)
			iinfo.sitesByMethod.snapshotInto(scratch)
			// One utils.LookupMethod call per scratch.entries iteration
			// (regardless of how many sites the entry holds). Counted
			// once via len(); per-worker counter avoids cross-worker
			// atomic contention.
			r.InvokeLookupsFromAddRuntimeTypeCounts.Add(workerID, len(scratch.entries))
			for i, entry := range scratch.entries {
				name := scratch.names[i]
				cmethod := utils.LookupMethod(r.prog, r.methodValueCache, T, entry.method.Pkg(), name)
				if cmethod == nil {
					continue
				}
				entry.mu.Lock()
				if !entry.addUniqueCmethod(cmethod) {
					// Another path already appended cmethod for T
					// (e.g. a concurrent visitInvoke first-time
					// snapshot included T). That path is responsible
					// for attaching edges from every site that
					// existed at the time it appended; new sites
					// arriving after pick up cmethod from
					// entry.cmethods via visitInvoke's seen-method
					// snapshot.
					entry.mu.Unlock()
					continue
				}
				// Reuse a pooled scratch slice instead of allocating a
				// fresh backing array per (T, I, m). The snapshot is
				// only needed until the AddCallGraphEdgeFast loop ends.
				sitesPtr := sitesSnapshotPool.Get().(*[]ssa.CallInstruction)
				sitesSnapshot := (*sitesPtr)[:0]
				sitesSnapshot = append(sitesSnapshot, entry.sites...)
				entry.mu.Unlock()

				r.addReachable(cmethod, true, workerID)

				if g != nil && len(sitesSnapshot) > 0 {
					to, toLock := g.CreateNodeWithLock(cmethod)
					for _, site := range sitesSnapshot {
						from, fromLock := g.CreateNodeWithLock(site.Parent())
						g.AddCallGraphEdgeFast(from, fromLock, site, to, toLock)
					}
				}
				*sitesPtr = sitesSnapshot[:0]
				sitesSnapshotPool.Put(sitesPtr)
			}
			sitesEntriesScratchPool.Put(scratch)
		})
	}

	// Precondition: T is not a method signature (*Signature with Recv()!=nil).
	// Recursive case: skip => don't call makeMethods(T).
	// Each package maintains its own set of types it has visited.

	var n *types.Named
	switch T := types.Unalias(T).(type) {
	case *types.Named:
		n = T
	case *types.Pointer:
		n, _ = types.Unalias(T.Elem()).(*types.Named)
	}
	if n != nil {
		owner := n.Obj().Pkg()
		if owner == nil {
			return // built-in error type
		}
	}

	// Recursion over signatures of each exported method.
	for i := 0; i < mset.Len(); i++ {
		if mset.At(i).Obj().Exported() {
			sig := mset.At(i).Type().(*types.Signature)
			r.addRuntimeType(sig.Params(), true, workerID)  // skip the Tuple itself
			r.addRuntimeType(sig.Results(), true, workerID) // skip the Tuple itself
		}
	}

	switch t := T.(type) {
	case *types.Alias:
		panic("unreachable")

	case *types.Basic:
		// nop

	case *types.Interface:
		// nop---handled by recursion over method set.

	case *types.Pointer:
		r.addRuntimeType(t.Elem(), false, workerID)

	case *types.Slice:
		r.addRuntimeType(t.Elem(), false, workerID)

	case *types.Chan:
		r.addRuntimeType(t.Elem(), false, workerID)

	case *types.Map:
		r.addRuntimeType(t.Key(), false, workerID)
		r.addRuntimeType(t.Elem(), false, workerID)

	case *types.Signature:
		if t.Recv() != nil {
			panic(fmt.Sprintf("Signature %s has Recv %s", t, t.Recv()))
		}
		r.addRuntimeType(t.Params(), true, workerID)  // skip the Tuple itself
		r.addRuntimeType(t.Results(), true, workerID) // skip the Tuple itself

	case *types.Named:
		// A pointer-to-named type can be derived from a named
		// type via reflection.  It may have methods too.
		r.addRuntimeType(types.NewPointer(T), false, workerID)

		// Consider 'type T struct{S}' where S has methods.
		// Reflection provides no way to get from T to struct{S},
		// only to S, so the method set of struct{S} is unwanted,
		// so set 'skip' flag during recursion.
		r.addRuntimeType(t.Underlying(), true, workerID)

	case *types.Array:
		r.addRuntimeType(t.Elem(), false, workerID)

	case *types.Struct:
		for i, n := 0, t.NumFields(); i < n; i++ {
			r.addRuntimeType(t.Field(i).Type(), false, workerID)
		}

	case *types.Tuple:
		for i, n := 0, t.Len(); i < n; i++ {
			r.addRuntimeType(t.At(i).Type(), false, workerID)
		}

	default:
		panic(T)
	}
}

// implements reports whether types.Implements(cinfo.C, iinfo.I),
// but more efficiently.
func (r *rta) implements(cinfo *concreteTypeInfo, iinfo *interfaceTypeInfo, workerID int) (got bool) {
	r.ImplementsCallCounts.Inc(workerID)
	result := iinfo.fprint & ^cinfo.fprint == 0 && types.Implements(cinfo.C, iinfo.I)
	if result {
		r.ImplementsSuccessCounts.Inc(workerID)
	} else {
		r.ImplementsFailCounts.Inc(workerID)
	}
	return result
}

// Per-snapshot scratch pools. Iterate/Range/snapshot need a snapshot of keys
// so writers aren't blocked behind a callback. Allocating that snapshot on
// every call costs significant GC pressure on services with many Iterate
// callers (one per cinfo addRuntimeType, one per visitInvoke first-time).
// sync.Pool keeps at most one buffer per P (essentially per-worker for our
// pool size), so each worker amortizes the snapshot alloc to ~0.
//
// Pool *[]T not []T: putting a 3-word slice header into sync.Pool's
// interface{} boxes ~24 B per Put; the pointer variant fits in the
// interface's data word directly.
var (
	typesSnapshotPool = sync.Pool{
		New: func() any { s := make([]types.Type, 0, 32); return &s },
	}
	cinfosSnapshotPool = sync.Pool{
		New: func() any { s := make([]*concreteTypeInfo, 0, 32); return &s },
	}
	iinfosSnapshotPool = sync.Pool{
		New: func() any { s := make([]*interfaceTypeInfo, 0, 32); return &s },
	}
	sitesEntriesScratchPool = sync.Pool{
		New: func() any {
			return &sitesEntriesScratch{
				names:   make([]string, 0, 8),
				entries: make([]*methodSitesEntry, 0, 8),
			}
		},
	}
)

// setCore is the shared body of the three concurrent set types below
// (typeSet, concretePtrSet, interfacePtrSet). Go's generics use GCShape
// stenciling, so all three pointer-shaped K parameters produce at most two
// machine-code stencils (interface shape vs pointer shape) -- no per-K
// duplication and no measurable overhead vs hand-written versions.
//
// add/len_/snapshot are unexported so the public wrappers (Set/Store/Add,
// Len/Size, Iterate/Range) can attach the matching sync.Pool for snapshot
// reuse. Embedding setCore[K] in the wrapper types promotes these to method
// receivers without any extra indirection.
type setCore[K comparable] struct {
	mu  sync.RWMutex
	set map[K]struct{}
}

// add inserts key into the set. Idempotent.
func (s *setCore[K]) add(key K) {
	s.mu.Lock()
	if s.set == nil {
		s.set = make(map[K]struct{})
	}
	s.set[key] = struct{}{}
	s.mu.Unlock()
}

// len_ returns the current size.
func (s *setCore[K]) len_() int {
	s.mu.RLock()
	n := len(s.set)
	s.mu.RUnlock()
	return n
}

// snapshot fills buf with the current contents (truncating if needed) and
// returns the resulting slice. Held under RLock so concurrent add()s wait,
// but the caller iterates the snapshot outside the lock so the callback
// can take any duration without blocking writers.
func (s *setCore[K]) snapshot(buf []K) []K {
	buf = buf[:0]
	s.mu.RLock()
	for k := range s.set {
		buf = append(buf, k)
	}
	s.mu.RUnlock()
	return buf
}

// typeSet is a concurrent-safe set of types.Type using pointer identity.
//
// Pointer identity is sufficient (despite types.Type not being canonicalized
// globally) because the only producers of keys stored here are:
//   - iinfo.implementations[cinfo.C]  -- cinfo.C is the canonical types.Type
//     for that Identical-equivalence class (the value that won concreteTypes
//     LoadOrStore). All other Identical inputs return the same cinfo and
//     never reach the Set call.
//   - cinfo.implements[iinfo.I]       -- analogous: iinfo.I is canonical.
type typeSet struct {
	setCore[types.Type]
}

// Set adds key. Idempotent.
func (s *typeSet) Set(key types.Type) { s.add(key) }

// Len returns the current size.
func (s *typeSet) Len() int { return s.len_() }

// Iterate calls f on each member. Snapshots into a pooled scratch buffer so
// f runs without holding the lock and the snapshot allocation is amortized.
func (s *typeSet) Iterate(f func(key types.Type)) {
	ptr := typesSnapshotPool.Get().(*[]types.Type)
	snap := s.snapshot(*ptr)
	for _, k := range snap {
		f(k)
	}
	*ptr = snap[:0]
	typesSnapshotPool.Put(ptr)
}

// concretePtrSet is the inner element of MC: the set of concrete types whose
// method set contains a particular method name. Pointer-keyed (*concreteTypeInfo
// is canonical via concreteTypes LoadOrStore). One per method name in the
// program (~20k for large services); replaces a ~4.5 KB xsync.Map per name.
type concretePtrSet struct {
	setCore[*concreteTypeInfo]
}

func (s *concretePtrSet) Store(c *concreteTypeInfo) { s.add(c) }
func (s *concretePtrSet) Size() int                 { return s.len_() }

// Range invokes f on each member. Snapshots into a pooled buffer so f runs
// outside the lock; this matters because the callback in implementations()
// does CPU-bound work (types.Implements + cross-direction updates) per
// candidate, and holding the bucket lock across that would serialize all
// workers creating new iinfos that picked this method as bestMethod.
func (s *concretePtrSet) Range(f func(*concreteTypeInfo) bool) {
	ptr := cinfosSnapshotPool.Get().(*[]*concreteTypeInfo)
	snap := s.snapshot(*ptr)
	for _, c := range snap {
		if !f(c) {
			break
		}
	}
	*ptr = snap[:0]
	cinfosSnapshotPool.Put(ptr)
}

// interfacePtrSet is the symmetric MI element.
type interfacePtrSet struct {
	setCore[*interfaceTypeInfo]
}

func (s *interfacePtrSet) Store(i *interfaceTypeInfo) { s.add(i) }
func (s *interfacePtrSet) Size() int                  { return s.len_() }

func (s *interfacePtrSet) Range(f func(*interfaceTypeInfo) bool) {
	ptr := iinfosSnapshotPool.Get().(*[]*interfaceTypeInfo)
	snap := s.snapshot(*ptr)
	for _, i := range snap {
		if !f(i) {
			break
		}
	}
	*ptr = snap[:0]
	iinfosSnapshotPool.Put(ptr)
}

// sitesEntriesScratch holds parallel name/entry buffers reused across
// methodSitesMap.snapshotInto calls.
type sitesEntriesScratch struct {
	names   []string
	entries []*methodSitesEntry
}

// methodSitesMap holds invoke sites at an interface grouped by method name.
// One per interfaceTypeInfo (~100k on large services); the prior
// *xsync.Map[string, *methodSitesEntry] was ~4.5 KB even when empty. The Go
// map is nil/0-byte until the first site is added, and ~48 B + per-entry
// overhead after that.
//
// Kept distinct from pooledSet because it's a string-keyed map (not a set)
// and exposes LoadOrCompute / snapshotInto (parallel slices), not Range.
type methodSitesMap struct {
	mu sync.RWMutex
	m  map[string]*methodSitesEntry
}

func (m *methodSitesMap) Load(name string) (*methodSitesEntry, bool) {
	m.mu.RLock()
	e, ok := m.m[name]
	m.mu.RUnlock()
	return e, ok
}

func (m *methodSitesMap) LoadOrCompute(name string, newEntry func() *methodSitesEntry) *methodSitesEntry {
	m.mu.RLock()
	if e, ok := m.m[name]; ok {
		m.mu.RUnlock()
		return e
	}
	m.mu.RUnlock()
	m.mu.Lock()
	if e, ok := m.m[name]; ok {
		m.mu.Unlock()
		return e
	}
	e := newEntry()
	if m.m == nil {
		m.m = make(map[string]*methodSitesEntry)
	}
	m.m[name] = e
	m.mu.Unlock()
	return e
}

// snapshotInto fills the caller's scratch (drawn from sitesEntriesScratchPool)
// with parallel name/entry slices captured under the read lock.
func (m *methodSitesMap) snapshotInto(scratch *sitesEntriesScratch) {
	scratch.names = scratch.names[:0]
	scratch.entries = scratch.entries[:0]
	m.mu.RLock()
	for name, entry := range m.m {
		scratch.names = append(scratch.names, name)
		scratch.entries = append(scratch.entries, entry)
	}
	m.mu.RUnlock()
}
