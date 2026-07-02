// Code is heavily borrowed from Go's RTA package. We have modified it make
// it parallel.

package prta_kumo

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
	callGraph    *callgraph.Graph
	reachable    *xsync.Map[*ssa.Function, bool]
	runtimeTypes utils.TypeMap

	// Lazy index stats
	mcMap                  *xsync.Map[string, *xsync.Map[*concreteTypeInfo, bool]]
	miMap                  *xsync.Map[string, *xsync.Map[*interfaceTypeInfo, bool]]
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
	return r.callGraph
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

// MCBucketSizePercentiles returns percentiles for MC bucket sizes (lazy computed).
func (r *Result) MCBucketSizePercentiles() rtalib.Percentiles {
	r.mcBucketOnce.Do(func() {
		var sizes []int
		r.mcMap.Range(func(_ string, ctypeSet *xsync.Map[*concreteTypeInfo, bool]) bool {
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
		r.miMap.Range(func(_ string, itypeSet *xsync.Map[*interfaceTypeInfo, bool]) bool {
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

// Materialize converts the result to the standard rta.Result.
// For parallel implementation, this performs the conversion from xsync.Map to regular map.
func (r *Result) Materialize() *rtapkg.Result {
	return &rtapkg.Result{
		CallGraph:    r.GetCallGraph(),
		Reachable:    r.GetReachable(),
		RuntimeTypes: r.GetRuntimeTypes(),
	}
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

type deduplicateEdgesWorkItem struct {
	nodes     []*callgraph.Node
	semaphore chan []*callgraph.Node
}

func (w *deduplicateEdgesWorkItem) doWork(r *rta, workerID int) {
	for _, node := range w.nodes {
		if len(node.In) > 1 {
			before := len(node.In)
			node.In = utils.DedupEdges(node.In)
			r.DuplicatedEdgeCounts.Add(workerID, before-len(node.In))
		}
		if len(node.Out) > 1 {
			before := len(node.Out)
			node.Out = utils.DedupEdges(node.Out)
			r.DuplicatedEdgeCounts.Add(workerID, before-len(node.Out))
		}
	}
	// Return slice to semaphore for recycling
	w.semaphore <- w.nodes[:0]
}

// A Result holds the results of Rapid Type Analysis, which includes the
// set of reachable functions/methods, runtime types, and the call graph.
// Working state of the RTA algorithm.
type rta struct {
	result *Result

	prog *ssa.Program

	reflectValueCall *ssa.Function // (*reflect.Value).Call, iff part of prog

	// pool drives the parallel RTA phase: per-worker chaselev deques with
	// random-victim work-stealing and sleeper-counted wake. See
	// utils.WorkerPool.
	pool *utils.WorkerPool[workItem]

	// addrTakenFuncsBySig contains all address-taken *Functions, grouped by signature.
	// Keys are *types.Signature, values are *xsync.Map[*ssa.Function, bool] sets.
	addrTakenFuncsBySig utils.TypeMap

	// dynCallSites contains all dynamic "call"-mode call sites, grouped by signature.
	// Keys are *types.Signature, values are *callSites.
	dynCallSites utils.TypeMap

	// invokeSites contains all "invoke"-mode call sites, grouped by interface.
	// Keys are *types.Interface (never *types.Named),
	// Values are *callSites.
	invokeSites utils.TypeMap

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
	// MC maps method names to sets of concrete types that have that method
	// Key: string (method name), Value: *xsync.Map[*concreteTypeInfo, bool]
	MC *xsync.Map[string, *xsync.Map[*concreteTypeInfo, bool]]

	// MI maps method names to sets of interface types that require that method
	// Key: string (method name), Value: *xsync.Map[*interfaceTypeInfo, bool]
	MI *xsync.Map[string, *xsync.Map[*interfaceTypeInfo, bool]]

	// methodCounts tracks how many interfaces use each method (for RarestMethodByInterfaceType)
	methodCounts *xsync.Map[string, *atomic.Int32]

	strategy rtalib.MethodSelectionStrategy

	callgraphLk sync.Mutex

	// Method value cache, to avoid lock contention
	methodValueCache *xsync.Map[utils.MethodKey, *ssa.Function]

	rtalib.WorkerCounters
}

type concreteTypeInfo struct {
	C      types.Type
	mset   *types.MethodSet
	fprint uint64 // fingerprint of method set

	implements utils.TypeMap

	// We use a channel to signal when initialization is done because `concreteTypeInfo` is made visible in the `concreteTypes` map before we populate the `implements` field. This allows consumers that only wish to know the presence of the `concreteTypeInfo` to progress, while those that need to access `implements` must block on the `initDone` channel.
	initDone chan struct{} // closed when initialization is complete
}

type interfaceTypeInfo struct {
	I      *types.Interface
	mset   *types.MethodSet
	fprint uint64

	implementations utils.TypeMap

	// See comment in `concreteTypeInfo`
	initDone chan struct{} // closed when initialization is complete
}

type callSites struct {
	sites *xsync.Map[ssa.CallInstruction, struct{}]
}

func newCallSites() *callSites {
	return &callSites{sites: xsync.NewMap[ssa.CallInstruction, struct{}]()}
}

// addReachable marks a function as potentially callable at run-time,
// and ensures that it gets processed.
//
// workerID is the caller's worker ID (so newly-discovered work lands in the
// caller's local deque with no contention), or -1 if called from outside
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

// addToWorklist enqueues item. workerID >= 0 routes the item to the caller's
// own deque (single-writer fast path). workerID < 0 means "called from
// outside the worker pool" — the item lands in mainbox, which the main
// goroutine owns and workers steal from.
//
// New work items are only added while processing an existing work item (or
// during initial dispatch). The worker calls workWg.Add(1) for each new
// work item it adds and only then calls workWg.Done() after processing its
// own work item, so the main thread does not exit Wait while there is still
// work in flight.
func (r *rta) addToWorklist(item workItem, workerID int) {
	r.pool.Push(item, workerID)
}

func (r *rta) addFunctionToWorklist(f *ssa.Function, workerID int) {
	r.addToWorklist(&processFunctionWorkItem{fn: f}, workerID)
}

// addEdge adds the specified call graph edge, and marks it reachable.
// addrTaken indicates whether to mark the callee as "address-taken".
// site is nil for calls made via reflection.
func (r *rta) addEdge(caller *ssa.Function, site ssa.CallInstruction, callee *ssa.Function, addrTaken bool, workerID int) {
	r.addReachable(callee, addrTaken, workerID)

	if g := r.result.callGraph; g != nil {
		if caller == nil {
			panic(site)
		}

		// TODO(elton): Remove this lock after callgraph type is made thread-safe
		r.callgraphLk.Lock()

		// g.CreateNode takes care of node de-duplication
		from := g.CreateNode(caller)
		to := g.CreateNode(callee)
		callgraph.AddEdge(from, site, to)

		r.callgraphLk.Unlock()
	}
}

// ---------- addrTakenFuncs × dynCallSites ----------

// visitAddrTakenFunc is called each time we encounter an address-taken function f.
func (r *rta) visitAddrTakenFunc(f *ssa.Function, workerID int) {
	// Create two-level map (Signature -> Function -> bool).
	S := f.Signature

	val, _ := r.addrTakenFuncsBySig.LoadOrStore(S, xsync.NewMap[*ssa.Function, bool]())
	funcs := val.(*xsync.Map[*ssa.Function, bool])

	if _, loaded := funcs.LoadOrStore(f, true); !loaded {
		// First time seeing f.

		// If we've seen any dyncalls of this type, mark it reachable,
		// and add call graph edges.
		val, _ := r.dynCallSites.LoadOrStore(S, newCallSites())
		s := val.(*callSites)
		s.sites.Range(func(site ssa.CallInstruction, _ struct{}) bool {
			r.addEdge(site.Parent(), site, f, true, workerID)
			return true
		})

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
		if r.reflectValueCall != nil {
			var site ssa.CallInstruction = nil // can't find actual call site
			r.addEdge(r.reflectValueCall, site, f, true, workerID)
		}
	}
}

// visitDynCall is called each time we encounter a dynamic "call"-mode call.
func (r *rta) visitDynCall(site ssa.CallInstruction, workerID int) {
	S := site.Common().Signature()

	// Record the call site.
	// Note: A function is never added more than once to the worklist. A callsite is added only by visiting each instruction in the function. So, it can also be added to s.sites only once.
	val, _ := r.dynCallSites.LoadOrStore(S, newCallSites())
	s := val.(*callSites)
	s.sites.Store(site, struct{}{})

	// For each function of signature S that we know is address-taken,
	// add an edge and mark it reachable.
	val, _ = r.addrTakenFuncsBySig.LoadOrStore(S, xsync.NewMap[*ssa.Function, bool]())
	funcs := val.(*xsync.Map[*ssa.Function, bool])
	funcs.Range(func(g *ssa.Function, _ bool) bool {
		r.addEdge(site.Parent(), site, g, true, workerID)
		return true
	})
}

// ---------- concrete types × invoke sites ----------

// addInvokeEdge is called for each new pair (site, C) in the matrix.
func (r *rta) addInvokeEdge(site ssa.CallInstruction, C types.Type, workerID int) {
	// Ascertain the concrete method of C to be called.
	imethod := site.Common().Method
	cmethod := utils.LookupMethod(r.prog, r.methodValueCache, C, imethod.Pkg(), imethod.Name())
	r.addEdge(site.Parent(), site, cmethod, true, workerID)
}

// visitInvoke is called each time the algorithm encounters an "invoke"-mode call.
func (r *rta) visitInvoke(site ssa.CallInstruction, workerID int) {
	I := site.Common().Value.Type().Underlying().(*types.Interface)

	// Record the invoke site.
	val, _ := r.invokeSites.LoadOrStore(I, newCallSites())
	s := val.(*callSites)

	s.sites.Store(site, struct{}{})

	// Add callgraph edge for each existing
	// address-taken concrete type implementing I.
	iinfo := r.implementations(I, workerID)

	// Wait for initialization to complete
	<-iinfo.initDone

	// One add per visitInvoke using TypeMap.Len(); avoids a per-impl
	// atomic increment inside the Iterate callback.
	r.InvokeLookupsFromVisitInvokeCounts.Add(workerID, iinfo.implementations.Len())
	iinfo.implementations.Iterate(func(C types.Type, v any) {
		r.addInvokeEdge(site, C, workerID)
	})
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

// RTA is an RTA implementation with method-based indexing optimization.
type RTA struct{}

// New creates a new RTA instance.
func New() *RTA {
	return &RTA{}
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
		MC:               xsync.NewMap[string, *xsync.Map[*concreteTypeInfo, bool]](),
		MI:               xsync.NewMap[string, *xsync.Map[*interfaceTypeInfo, bool]](),
		methodCounts:     xsync.NewMap[string, *atomic.Int32](),
		strategy:         cfg.MethodSelectionStrategy,
		methodValueCache: xsync.NewMap[utils.MethodKey, *ssa.Function](),
		WorkerCounters:   rtalib.NewWorkerCounters(numWorkers),
	}

	if buildCallGraph {
		r.result.callGraph = callgraph.New(roots[0])
	}

	// Grab ssa.Function for (*reflect.Value).Call,
	// if "reflect" is among the dependencies.
	if reflectPkg := r.prog.ImportedPackage("reflect"); reflectPkg != nil {
		reflectValue := reflectPkg.Members["Value"].(*ssa.Type)
		r.reflectValueCall = utils.LookupMethod(r.prog, r.methodValueCache, reflectValue.Object().Type(), reflectPkg.Pkg, "Call")
	}

	hasher := typeutil.MakeHasher()
	r.result.runtimeTypes.SetHasher(hasher)
	r.addrTakenFuncsBySig.SetHasher(hasher)
	r.dynCallSites.SetHasher(hasher)
	r.invokeSites.SetHasher(hasher)
	r.concreteTypes.SetHasher(hasher)
	r.interfaceTypes.SetHasher(hasher)

	r.pool.Start(func(item workItem, workerID int) {
		item.doWork(r, workerID)
	})

	// Add the initial work to mainbox; workers will steal from it.
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

	// Record pre-dedup timing and memory
	preDedupDuration := time.Since(analysisStart)
	var preDedupMem runtime.MemStats
	runtime.ReadMemStats(&preDedupMem)

	// De-duplicate edges, using the worker pool with batching
	r.processNodesInBatches(numWorkers)

	// Record post-dedup timing and memory
	postDedupDuration := time.Since(analysisStart)
	var postDedupMem runtime.MemStats
	runtime.ReadMemStats(&postDedupMem)

	// Signal workers to shut down
	r.pool.Stop()

	// Sum per-worker counters
	r.WorkerCounters.MergeInto(&r.result.BaseMetrics)
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

// processNodesInBatches processes deduplication work items in batches to avoid
// doubling memory usage by limiting the number of work items queued at once.
// It uses a batch size of 2*numWorkers.
// A buffered channel acts as both a semaphore and a slice pool.
func (r *rta) processNodesInBatches(numWorkers int) {
	batchSize := numWorkers * 2

	// Create semaphore channel that holds slices for recycling
	semaphore := make(chan []*callgraph.Node, batchSize)

	// Pre-populate with empty slices
	for i := 0; i < batchSize; i++ {
		semaphore <- make([]*callgraph.Node, 0, batchSize)
	}

	// Get a slice from the semaphore (blocks if none available)
	batch := <-semaphore

	// Process nodes directly from the map without creating an intermediate slice
	for _, node := range r.result.callGraph.Nodes {
		batch = append(batch, node)

		// When batch is full, submit it and get a new one
		if len(batch) >= batchSize {
			r.addToWorklist(&deduplicateEdgesWorkItem{nodes: batch, semaphore: semaphore}, -1)

			// Get next slice from semaphore (blocks if all are in use)
			batch = <-semaphore
		}
	}

	// Submit remaining nodes if any
	if len(batch) > 0 {
		r.addToWorklist(&deduplicateEdgesWorkItem{nodes: batch, semaphore: semaphore}, -1)
	} else {
		// Return unused slice to semaphore
		semaphore <- batch
	}

	// Wait for all batches to complete by draining the semaphore
	for i := 0; i < batchSize; i++ {
		<-semaphore
	}
}

// interfaces() and implementations()
//
// In both of these methods, if we are encoutering the concrete type/interface
// for the first time, we broadcast the existence of a
// concreteTypeInfo/interfaceTypeInfo (via a LoadOrStore) and only then
// populate `implements`/`implementations` fields (via Range). This order
// ensures that no work item gets dropped; either a consumer of `interfaces()`
// will end up adding it, or a consumer of `implementations()` will add it.

// interfaces(C) returns all currently known interfaces implemented by C.
func (r *rta) interfaces(C types.Type, workerID int) *concreteTypeInfo {
	switch C.(type) {
	case *types.Tuple,
		*types.Array,
		*types.Slice,
		*types.Chan,
		*types.Signature,
		*types.Map:
		cinfo := &concreteTypeInfo{
			C:        C,
			initDone: make(chan struct{}),
		}
		close(cinfo.initDone)
		return cinfo
	}

	// Create an info for C the first time we see it.
	var cinfo *concreteTypeInfo
	if v := r.concreteTypes.At(C); v != nil {
		cinfo = v.(*concreteTypeInfo)
	} else {
		mset := r.prog.MethodSets.MethodSet(C)
		cinfo = &concreteTypeInfo{
			C:        C,
			mset:     mset,
			fprint:   utils.Fingerprint(mset),
			initDone: make(chan struct{}),
		}
		if oldcinfo, loaded := r.concreteTypes.LoadOrStore(C, cinfo); !loaded {
			// Kumo optimization: Index this concrete type under each of its methods
			for i := 0; i < mset.Len(); i++ {
				methodName := mset.At(i).Obj().Name()
				ctypeSet, _ := r.MC.LoadOrCompute(methodName, func() (*xsync.Map[*concreteTypeInfo, bool], bool) {
					return xsync.NewMap[*concreteTypeInfo, bool](), false
				})
				ctypeSet.Store(cinfo, true)
			}

			// Kumo optimization: Find candidate interfaces by looking up methods
			// instead of iterating all interfaces
			candidates := make(map[*interfaceTypeInfo]bool)
			for i := 0; i < mset.Len(); i++ {
				methodName := mset.At(i).Obj().Name()
				if itypeSet, ok := r.MI.Load(methodName); ok {
					itypeSet.Range(func(iinfo *interfaceTypeInfo, _ bool) bool {
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
					iinfo.implementations.Set(C, struct{}{})
					cinfo.implements.Set(I, struct{}{})
					r.SuccessFromInterfacesCounts.Inc(workerID)
				} else {
					r.FailsFromInterfacesCounts.Inc(workerID)
				}
			}

			close(cinfo.initDone)
		} else {
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
		mset := r.prog.MethodSets.MethodSet(I)
		iinfo = &interfaceTypeInfo{
			I:        I,
			mset:     mset,
			fprint:   utils.Fingerprint(mset),
			initDone: make(chan struct{}),
		}
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
				itypeSet, _ := r.MI.LoadOrCompute(selectedMethod, func() (*xsync.Map[*interfaceTypeInfo, bool], bool) {
					return xsync.NewMap[*interfaceTypeInfo, bool](), false
				})
				itypeSet.Store(iinfo, true)
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
				// Early termination: if bestSize <= remaining methods, stop scanning
				if bestSize <= mset.Len()-(i+1) {
					break
				}
			}

			// Ascertain set of concrete types that implement I
			// and update the 'implements' relation.
			if bestMethod != "" && !noCandidates {
				if ctypeSet, ok := r.MC.Load(bestMethod); ok {
					ctypeSet.Range(func(cinfo *concreteTypeInfo, _ bool) bool {
						r.ChecksFromImplementationsCounts.Inc(workerID)
						if r.implements(cinfo, iinfo, workerID) {
							cinfo.implements.Set(I, struct{}{})
							iinfo.implementations.Set(cinfo.C, struct{}{})
							r.SuccessFromImplementationsCounts.Inc(workerID)
						} else {
							r.FailsFromImplementationsCounts.Inc(workerID)
						}
						return true
					})
				}
			}

			close(iinfo.initDone)
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

	mset := r.prog.MethodSets.MethodSet(T)

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

		// Add callgraph edge for each existing dynamic
		// "invoke"-mode call via that interface.
		cinfo := r.interfaces(T, workerID)

		// Wait for initialization to complete
		<-cinfo.initDone

		cinfo.implements.Iterate(func(k types.Type, v any) {
			I := k.(*types.Interface)
			val, _ := r.invokeSites.LoadOrStore(I, newCallSites())
			s := val.(*callSites)
			// One add per (T, I) using xsync.Map.Size(); per-worker counter
			// avoids the per-site atomic Inc inside the Range callback.
			r.InvokeLookupsFromAddRuntimeTypeCounts.Add(workerID, s.sites.Size())
			s.sites.Range(func(site ssa.CallInstruction, _ struct{}) bool {
				r.addInvokeEdge(site, T, workerID)
				return true
			})
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
