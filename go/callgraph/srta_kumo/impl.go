package srta_kumo

import (
	"fmt"
	"go/types"
	"math/rand"
	"sync"

	"golang.org/x/tools/go/callgraph"
	rtapkg "golang.org/x/tools/go/callgraph/rta"
	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/rtalib/utils"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types/typeutil"
)

// Result implements the rta.Result interface for sequential RTA with method indexing.
// It uses regular typeutil.Map for optimal sequential performance.
type Result struct {
	rtalib.BaseMetrics
	callGraph    *callgraph.Graph
	reachable    map[*ssa.Function]struct{ AddrTaken bool }
	runtimeTypes typeutil.Map

	// Lazy index stats
	mc                        map[string]map[*concreteTypeInfo]struct{}
	mi                        map[string](map[*interfaceTypeInfo]struct{})
	interfaceTypesMap         typeutil.Map
	concreteTypesMap          typeutil.Map
	mcBucketOnce              sync.Once
	mcBucketPercentiles       rtalib.Percentiles
	miBucketOnce              sync.Once
	miBucketPercentiles       rtalib.Percentiles
	ifaceMethodBucketsOnce    sync.Once
	ifaceMethodBuckets        [8]float64
	methodsPerConcreteOnce    sync.Once
	methodsPerConcrete        rtalib.Percentiles
	methodsPerInterfaceOnce   sync.Once
	methodsPerInterface       rtalib.Percentiles
	concreteMethodBucketsOnce sync.Once
	concreteMethodBuckets     [8]float64
}

// GetCallGraph returns the discovered callgraph.
func (r *Result) GetCallGraph() *callgraph.Graph {
	return r.callGraph
}

// GetReachable returns the set of reachable functions.
func (r *Result) GetReachable() map[*ssa.Function]struct{ AddrTaken bool } {
	return r.reachable
}

// GetRuntimeTypes returns the set of runtime types.
func (r *Result) GetRuntimeTypes() typeutil.Map {
	return r.runtimeTypes
}

// Materialize converts the result to the standard rta.Result.
// For sequential implementation, this is a simple copy since we already use the target types.
func (r *Result) Materialize() *rtapkg.Result {
	return &rtapkg.Result{
		CallGraph:    r.callGraph,
		Reachable:    r.reachable,
		RuntimeTypes: r.runtimeTypes,
	}
}

// MCBucketSizePercentiles returns percentiles for MC bucket sizes (lazy computed).
func (r *Result) MCBucketSizePercentiles() rtalib.Percentiles {
	r.mcBucketOnce.Do(func() {
		sizes := make([]int, 0, len(r.mc))
		for _, bucket := range r.mc {
			sizes = append(sizes, len(bucket))
		}
		r.mcBucketPercentiles = rtalib.ComputePercentiles(sizes)
	})
	return r.mcBucketPercentiles
}

// MIBucketSizePercentiles returns percentiles for MI bucket sizes (lazy computed).
func (r *Result) MIBucketSizePercentiles() rtalib.Percentiles {
	r.miBucketOnce.Do(func() {
		sizes := make([]int, 0, len(r.mi))
		for _, bucket := range r.mi {
			sizes = append(sizes, len(bucket))
		}
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

// MethodsPerConcretePercentiles returns percentiles for methods-per-concrete-type (lazy computed).
func (r *Result) MethodsPerConcretePercentiles() rtalib.Percentiles {
	r.methodsPerConcreteOnce.Do(func() {
		var sizes []int
		r.concreteTypesMap.Iterate(func(_ types.Type, v any) {
			cinfo := v.(*concreteTypeInfo)
			sizes = append(sizes, cinfo.mset.Len())
		})
		r.methodsPerConcrete = rtalib.ComputePercentiles(sizes)
	})
	return r.methodsPerConcrete
}

// MethodsPerInterfacePercentiles returns percentiles for methods-per-interface-type (lazy computed).
func (r *Result) MethodsPerInterfacePercentiles() rtalib.Percentiles {
	r.methodsPerInterfaceOnce.Do(func() {
		var sizes []int
		r.interfaceTypesMap.Iterate(func(_ types.Type, v any) {
			iinfo := v.(*interfaceTypeInfo)
			sizes = append(sizes, iinfo.mset.Len())
		})
		r.methodsPerInterface = rtalib.ComputePercentiles(sizes)
	})
	return r.methodsPerInterface
}

// ConcreteMethodCountBuckets returns the fraction of concrete types in each
// method-count bucket: [1, 2, 4, 8, 16, 32, 64, >64].
func (r *Result) ConcreteMethodCountBuckets() [8]float64 {
	r.concreteMethodBucketsOnce.Do(func() {
		var counts [8]int
		total := 0
		r.concreteTypesMap.Iterate(func(_ types.Type, v any) {
			cinfo := v.(*concreteTypeInfo)
			total++
			n := cinfo.mset.Len()
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
				r.concreteMethodBuckets[i] = float64(c) / float64(total)
			}
		}
	})
	return r.concreteMethodBuckets
}

// Working state of the RTA algorithm.
type rta struct {
	// Use internal map during analysis
	reachable map[*ssa.Function]struct{ AddrTaken bool }
	result    *Result

	prog *ssa.Program

	reflectValueCall *ssa.Function // (*reflect.Value).Call, iff part of prog

	worklist []*ssa.Function // list of functions to visit

	// addrTakenFuncsBySig contains all address-taken *Functions, grouped by signature.
	// Keys are *types.Signature, values are map[*ssa.Function]struct{} sets.
	addrTakenFuncsBySig typeutil.Map

	// dynCallSites contains all dynamic "call"-mode call sites, grouped by signature.
	// Keys are *types.Signature, values are unordered []ssa.CallInstruction.
	dynCallSites typeutil.Map

	// invoke sites are grouped per (interface, method) by
	// interfaceTypeInfo.sitesByMethod; no flat per-interface list is kept.

	// The following two maps together define the subset of the
	// m:n "implements" relation needed by the algorithm.

	// concreteTypes maps each concrete type to information about it.
	// Keys are types.Type, values are *concreteTypeInfo.
	// Only concrete types used as MakeInterface operands are included.
	concreteTypes typeutil.Map

	// interfaceTypes maps each interface type to information about it.
	// Keys are *types.Interface, values are *interfaceTypeInfo.
	// Only interfaces used in "invoke"-mode CallInstructions are included.
	interfaceTypes typeutil.Map

	MC           map[string]map[*concreteTypeInfo]struct{}
	MI           map[string](map[*interfaceTypeInfo]struct{})
	methodCounts map[string]int32
	strategy     rtalib.MethodSelectionStrategy

	// methodValueCache memoizes prog.LookupMethod results, avoiding repeated
	// MethodSet.Lookup + MethodValue work that dominates addInvokeEdge on
	// large services.
	methodValueCache map[utils.MethodKey]*ssa.Function

	implementsCallCount       int
	implementsSuccessCount    int
	implementsFailCount       int
	checksFromInterfaces      int
	checksFromImplementations int
	staticCallSites           int
	indirectCallSites         int
	invokeCallSites           int

	// invokeLookupsFromVisitInvoke counts utils.LookupMethodSeq calls
	// performed in visitInvoke. Only the first-time path at (I, method)
	// actually does lookups (one per impl); subsequent sites reuse the
	// cached cmethods and do zero lookups.
	invokeLookupsFromVisitInvoke int
	// invokeLookupsFromAddRuntimeType counts utils.LookupMethodSeq calls
	// performed in addRuntimeType. One per (T, I, method) entry — not
	// per site — so this falls below a non-sitesByMethod flavor by the
	// per-entry site-count factor.
	invokeLookupsFromAddRuntimeType int

	// Per-direction success/fail counts
	failsFromInterfaces        int
	successFromInterfaces      int
	failsFromImplementations   int
	successFromImplementations int
}

type concreteTypeInfo struct {
	C          types.Type
	mset       *types.MethodSet
	fprint     uint64             // fingerprint of method set
	implements []*types.Interface // unordered set of implemented interfaces
}

type interfaceTypeInfo struct {
	I               *types.Interface
	mset            *types.MethodSet
	fprint          uint64
	implementations []types.Type // unordered set of concrete implementations

	// sitesByMethod groups invoke sites at I by method name. Each entry
	// caches a parallel list of resolved concrete-method functions, one
	// per impl of I currently known to call this method. visitInvoke's
	// "first site at (I, m)" path builds the cmethods cache and marks
	// each reachable; subsequent sites for the same method skip the
	// per-impl LookupMethod + addReachable and only attach a callgraph
	// edge per cached cmethod. addRuntimeType for a new T extends each
	// entry.cmethods with T's cmethod (one LookupMethod + addReachable
	// per (T, I, m) instead of per (T, I, site)).
	sitesByMethod map[string]*methodSitesEntry
}

// methodSitesEntry holds invoke sites at (I, m.Name()) and the resolved
// concrete-method functions for each impl of I that has been processed
// for this method. cmethods is the cache; visitInvoke fast path
// iterates it without any LookupMethod or addReachable calls.
type methodSitesEntry struct {
	method      *types.Func           // abstract method on I (carries Pkg + Name for LookupMethod)
	sites       []ssa.CallInstruction // sites at (I, method.Name())
	cmethods    []*ssa.Function       // one cmethod per impl of I that maps method (nils filtered out)
	initialized bool                  // true once visitInvoke first-time has built cmethods
}

// addReachable marks a function as potentially callable at run-time,
// and ensures that it gets processed.
func (r *rta) addReachable(f *ssa.Function, addrTaken bool) {
	reachable := r.reachable
	n := len(reachable)
	v := reachable[f]
	if addrTaken {
		v.AddrTaken = true
	}
	reachable[f] = v
	if len(reachable) > n {
		// First time seeing f.  Add it to the worklist.
		r.worklist = append(r.worklist, f)
	}
}

// addEdge adds the specified call graph edge, and marks it reachable.
// addrTaken indicates whether to mark the callee as "address-taken".
// site is nil for calls made via reflection.
func (r *rta) addEdge(caller *ssa.Function, site ssa.CallInstruction, callee *ssa.Function, addrTaken bool) {
	r.addReachable(callee, addrTaken)

	if g := r.result.callGraph; g != nil {
		if caller == nil {
			panic(site)
		}
		from := g.CreateNode(caller)
		to := g.CreateNode(callee)
		callgraph.AddEdge(from, site, to)
	}
}

// ---------- addrTakenFuncs × dynCallSites ----------

// visitAddrTakenFunc is called each time we encounter an address-taken function f.
func (r *rta) visitAddrTakenFunc(f *ssa.Function) {
	// Create two-level map (Signature -> Function presence set).
	// struct{} value (instead of bool) saves ~1 B per entry: on services
	// with many address-taken functions this is a free win.
	S := f.Signature
	funcs, _ := r.addrTakenFuncsBySig.At(S).(map[*ssa.Function]struct{})
	if funcs == nil {
		funcs = make(map[*ssa.Function]struct{})
		r.addrTakenFuncsBySig.Set(S, funcs)
	}
	if _, seen := funcs[f]; !seen {
		// First time seeing f.
		funcs[f] = struct{}{}

		// If we've seen any dyncalls of this type, mark it reachable,
		// and add call graph edges.
		sites, _ := r.dynCallSites.At(S).([]ssa.CallInstruction)
		for _, site := range sites {
			r.addEdge(site.Parent(), site, f, true)
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
		if r.reflectValueCall != nil {
			var site ssa.CallInstruction = nil // can't find actual call site
			r.addEdge(r.reflectValueCall, site, f, true)
		}
	}
}

// visitDynCall is called each time we encounter a dynamic "call"-mode call.
func (r *rta) visitDynCall(site ssa.CallInstruction) {
	S := site.Common().Signature()

	// Record the call site.
	sites, _ := r.dynCallSites.At(S).([]ssa.CallInstruction)
	r.dynCallSites.Set(S, append(sites, site))

	// For each function of signature S that we know is address-taken,
	// add an edge and mark it reachable.
	funcs, _ := r.addrTakenFuncsBySig.At(S).(map[*ssa.Function]struct{})
	for g := range funcs {
		r.addEdge(site.Parent(), site, g, true)
	}
}

// ---------- concrete types × invoke sites ----------

// addInvokeEdge is called for each new pair (site, C) in the matrix.
// (Kept for callers outside the optimized fast paths.)
func (r *rta) addInvokeEdge(site ssa.CallInstruction, C types.Type) {
	// Ascertain the concrete method of C to be called.
	imethod := site.Common().Method
	cmethod := utils.LookupMethodSeq(r.prog, r.methodValueCache, C, imethod.Pkg(), imethod.Name())
	if cmethod == nil {
		return
	}
	r.addEdge(site.Parent(), site, cmethod, true)
}

// visitInvoke is called each time the algorithm encounters an
// "invoke"-mode call site. Per-method-site indexing: invoke sites are
// grouped under iinfo.sitesByMethod[methodName]; the first site at
// (I, methodName) resolves cmethod for each impl C of I and caches them
// in entry.cmethods, while subsequent sites for the same method skip
// the per-impl LookupMethod + addReachable and only attach a callgraph
// edge per cached cmethod. addRuntimeType for a new T extends each
// entry.cmethods with T's cmethod, so future visitInvoke fast-path
// snapshots see it.
//
// Sequential code: no mutex, and no cmethodSet for dedup. By causal
// order, visitInvoke first-time can only add cmethod_T_m if T was in
// iinfo.implementations at that moment — which (in sequential
// execution) requires interfaces(T) to have already run and added T to
// concreteTypes. addRuntimeType(T) is the unique caller of
// interfaces(T) and processes sitesByMethod[I] immediately after, so
// at the time visitInvoke first-time fires, addRuntimeType has either
// already finished (and found no entry to extend, since visitInvoke
// hadn't created it yet) or hasn't yet started. Either path adds
// cmethod_T_m exactly once.
func (r *rta) visitInvoke(site ssa.CallInstruction) {
	I := site.Common().Value.Type().Underlying().(*types.Interface)
	imethod := site.Common().Method

	// Compute implementations(I); also ensures iinfo exists.
	impls := r.implementations(I)
	iinfo := r.interfaceTypes.At(I).(*interfaceTypeInfo)

	// Get (or create) the per-method entry and append this site.
	if iinfo.sitesByMethod == nil {
		iinfo.sitesByMethod = make(map[string]*methodSitesEntry)
	}
	methodName := imethod.Name()
	entry, methodSeen := iinfo.sitesByMethod[methodName]
	if !methodSeen {
		entry = &methodSitesEntry{method: imethod}
		iinfo.sitesByMethod[methodName] = entry
	}
	entry.sites = append(entry.sites, site)

	if !entry.initialized {
		// First site at (I, methodName): resolve cmethod for each impl
		// of I, mark each reachable, cache in entry.cmethods, and
		// attach the edge from this site to each.
		entry.initialized = true
		entry.cmethods = make([]*ssa.Function, 0, len(impls))
		g := r.result.callGraph
		var from *callgraph.Node // created lazily to avoid empty-impls noise
		for _, C := range impls {
			cmethod := utils.LookupMethodSeq(r.prog, r.methodValueCache, C, imethod.Pkg(), methodName)
			if cmethod == nil {
				continue
			}
			r.addReachable(cmethod, true)
			entry.cmethods = append(entry.cmethods, cmethod)
			if g != nil {
				if from == nil {
					from = g.CreateNode(site.Parent())
				}
				to := g.CreateNode(cmethod)
				callgraph.AddEdge(from, site, to)
			}
		}
		// First-time path: one LookupMethodSeq call per impl iterated above.
		r.invokeLookupsFromVisitInvoke += len(impls)
		return
	}

	// methodName already seen on I — entry.cmethods is the up-to-date
	// cache (extended by addRuntimeType as new impls arrived). Just
	// attach edges from this site to each cached cmethod; no
	// LookupMethodSeq call happens on this path.
	g := r.result.callGraph
	if g == nil || len(entry.cmethods) == 0 {
		return
	}
	from := g.CreateNode(site.Parent())
	for _, cmethod := range entry.cmethods {
		to := g.CreateNode(cmethod)
		callgraph.AddEdge(from, site, to)
	}
}

// ---------- main algorithm ----------

// visitFunc processes function f.
func (r *rta) visitFunc(f *ssa.Function) {
	var space [32]*ssa.Value // preallocate space for common case

	for _, b := range f.Blocks {
		for _, instr := range b.Instrs {
			rands := instr.Operands(space[:0])

			switch instr := instr.(type) {
			case ssa.CallInstruction:
				call := instr.Common()
				if call.IsInvoke() {
					r.invokeCallSites++
					r.visitInvoke(instr)
				} else if g := call.StaticCallee(); g != nil {
					r.staticCallSites++
					r.addEdge(f, instr, g, false)
				} else if _, ok := call.Value.(*ssa.Builtin); !ok {
					r.indirectCallSites++
					r.visitDynCall(instr)
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
				r.addRuntimeType(instr.X.Type(), false)
			}

			// Process all address-taken functions.
			for _, op := range rands {
				if g, ok := (*op).(*ssa.Function); ok {
					r.visitAddrTakenFunc(g)
				}
			}
		}
	}
}

// RTA is a sequential RTA implementation with method-based indexing.
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

	r := &rta{
		reachable:        make(map[*ssa.Function]struct{ AddrTaken bool }),
		result:           &Result{},
		prog:             roots[0].Prog,
		MC:               make(map[string]map[*concreteTypeInfo]struct{}),
		MI:               make(map[string](map[*interfaceTypeInfo]struct{})),
		methodCounts:     make(map[string]int32),
		strategy:         cfg.MethodSelectionStrategy,
		methodValueCache: make(map[utils.MethodKey]*ssa.Function),
	}

	if buildCallGraph {
		// TODO(adonovan): change callgraph API to eliminate the
		// notion of a distinguished root node.  Some callgraphs
		// have many roots, or none.
		r.result.callGraph = callgraph.New(roots[0])
	}

	// Grab ssa.Function for (*reflect.Value).Call,
	// if "reflect" is among the dependencies.
	if reflectPkg := r.prog.ImportedPackage("reflect"); reflectPkg != nil {
		reflectValue := reflectPkg.Members["Value"].(*ssa.Type)
		r.reflectValueCall = utils.LookupMethodSeq(r.prog, r.methodValueCache, reflectValue.Object().Type(), reflectPkg.Pkg, "Call")
	}

	hasher := typeutil.MakeHasher()
	r.result.runtimeTypes.SetHasher(hasher)
	r.addrTakenFuncsBySig.SetHasher(hasher)
	r.dynCallSites.SetHasher(hasher)
	r.concreteTypes.SetHasher(hasher)
	r.interfaceTypes.SetHasher(hasher)

	for _, root := range roots {
		r.addReachable(root, false)
	}

	// Visit functions, processing their instructions, and adding
	// new functions to the worklist, until a fixed point is
	// reached.
	var shadow []*ssa.Function // for efficiency, we double-buffer the worklist
	for len(r.worklist) > 0 {
		shadow, r.worklist = r.worklist, shadow[:0]
		for _, f := range shadow {
			r.visitFunc(f)
		}
	}

	// Copy internal map to result
	r.result.reachable = r.reachable
	r.result.NumConcreteTypesVal = r.concreteTypes.Len()
	r.result.NumInterfaceTypesVal = r.interfaceTypes.Len()
	r.result.ImplementsCallCountVal = r.implementsCallCount
	r.result.ImplementsSuccessCountVal = r.implementsSuccessCount
	r.result.ImplementsFailCountVal = r.implementsFailCount
	r.result.ChecksFromInterfacesVal = r.checksFromInterfaces
	r.result.ChecksFromImplementationsVal = r.checksFromImplementations
	r.result.FailsFromInterfacesVal = r.failsFromInterfaces
	r.result.SuccessFromInterfacesVal = r.successFromInterfaces
	r.result.FailsFromImplementationsVal = r.failsFromImplementations
	r.result.SuccessFromImplementationsVal = r.successFromImplementations
	r.result.StaticCallSitesVal = r.staticCallSites
	r.result.IndirectCallSitesVal = r.indirectCallSites
	r.result.InvokeCallSitesVal = r.invokeCallSites
	r.result.InvokeLookupsFromVisitInvokeVal = r.invokeLookupsFromVisitInvoke
	r.result.InvokeLookupsFromAddRuntimeTypeVal = r.invokeLookupsFromAddRuntimeType

	// Store raw data for lazy percentile computation
	r.result.mc = r.MC
	r.result.mi = r.MI
	r.result.interfaceTypesMap = r.interfaceTypes
	r.result.concreteTypesMap = r.concreteTypes

	return r.result
}

// interfaces(C) returns all currently known interfaces implemented by C.
func (r *rta) interfaces(C types.Type) []*types.Interface {
	// These types have no methods and cannot implement any interface.
	// Skip them to avoid unnecessary method set computation and indexing.
	switch C.(type) {
	case *types.Tuple, *types.Array, *types.Slice,
		*types.Chan, *types.Signature, *types.Map:
		return nil
	}

	// Create an info for C the first time we see it.
	var cinfo *concreteTypeInfo
	if v := r.concreteTypes.At(C); v != nil {
		cinfo = v.(*concreteTypeInfo)
	} else {
		mset := r.prog.MethodSets.MethodSet(C)
		cinfo = &concreteTypeInfo{
			C:      C,
			mset:   mset,
			fprint: utils.Fingerprint(mset),
		}
		r.concreteTypes.Set(C, cinfo)

		// Kumo optimization

		for i := 0; i < mset.Len(); i++ {
			methodName := mset.At(i).Obj().Name()
			bucket := r.MC[methodName]
			if bucket == nil {
				bucket = make(map[*concreteTypeInfo]struct{})
				r.MC[methodName] = bucket
			}
			bucket[cinfo] = struct{}{}
		}
		cands := make(map[*interfaceTypeInfo]struct{})
		for i := 0; i < mset.Len(); i++ {
			methodName := mset.At(i).Obj().Name()
			if itypes, ok := r.MI[methodName]; ok {
				for iinfo := range itypes {
					cands[iinfo] = struct{}{}
				}
			}
		}
		for iinfo := range cands {
			r.checksFromInterfaces++
			if I := types.Unalias(iinfo.I).(*types.Interface); r.implements(cinfo, iinfo) {
				iinfo.implementations = append(iinfo.implementations, C)
				cinfo.implements = append(cinfo.implements, I)
				r.successFromInterfaces++
			} else {
				r.failsFromInterfaces++
			}
		}
	}

	return cinfo.implements
}

// implementations(I) returns all currently known concrete types that implement I.
func (r *rta) implementations(I *types.Interface) []types.Type {
	// Create an info for I the first time we see it.
	var iinfo *interfaceTypeInfo
	if v := r.interfaceTypes.At(I); v != nil {
		iinfo = v.(*interfaceTypeInfo)
	} else {
		mset := r.prog.MethodSets.MethodSet(I)
		iinfo = &interfaceTypeInfo{
			I:      I,
			mset:   mset,
			fprint: utils.Fingerprint(mset),
		}
		r.interfaceTypes.Set(I, iinfo)

		// Kumo optimization: MI indexing with configurable method selection strategy.
		var selectedMethod string
		switch r.strategy {
		case rtalib.RandomMethodStrategy:
			for i := 0; i < mset.Len(); i++ {
				r.methodCounts[mset.At(i).Obj().Name()]++
			}
			if mset.Len() > 0 {
				selectedMethod = mset.At(rand.Intn(mset.Len())).Obj().Name()
			}
		case rtalib.RarestMethodByInterfaceType:
			var rarestCount int32 = 1<<31 - 1
			for i := 0; i < mset.Len(); i++ {
				methodName := mset.At(i).Obj().Name()
				count := r.methodCounts[methodName] + 1
				r.methodCounts[methodName] = count
				if count < rarestCount {
					rarestCount = count
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
					mcSize := len(r.MC[methodName])
					if mcSize < bestMCSize {
						bestMCSize = mcSize
						selectedMethod = methodName
					}
				}
			}
		}
		if selectedMethod != "" {
			if _, ok := r.MI[selectedMethod]; !ok {
				r.MI[selectedMethod] = make(map[*interfaceTypeInfo]struct{})
			}
			r.MI[selectedMethod][iinfo] = struct{}{}
		}

		// Find the method with the smallest MC bucket to scan candidates
		var bestMethod string
		var bestSize int = 1<<31 - 1
		noCandidates := false

		for i := 0; i < mset.Len(); i++ {
			methodName := mset.At(i).Obj().Name()
			if bucket := r.MC[methodName]; bucket != nil {
				if len(bucket) < bestSize {
					bestSize = len(bucket)
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
		if bestMethod != "" && !noCandidates {
			if bucket := r.MC[bestMethod]; bucket != nil {
				// Pre-size implementations to the candidate bucket size:
				// the final list is at most len(bucket) entries after the
				// implements() filter. Avoids slice-doubling garbage on
				// hot interfaces with many candidate concretes.
				iinfo.implementations = make([]types.Type, 0, len(bucket))
				for cinfo := range bucket {
					r.checksFromImplementations++
					if r.implements(cinfo, iinfo) {
						iinfo.implementations = append(iinfo.implementations, cinfo.C)
						cinfo.implements = append(cinfo.implements, I)
						r.successFromImplementations++
					} else {
						r.failsFromImplementations++
					}
				}
			}
		}
	}
	return iinfo.implementations
}

// addRuntimeType is called for each concrete type that can be the
// dynamic type of some interface or reflect.Value.
// Adapted from needMethods in go/ssa/builder.go
func (r *rta) addRuntimeType(T types.Type, skip bool) {
	// Never record aliases.
	T = types.Unalias(T)

	if prev, ok := r.result.runtimeTypes.At(T).(bool); ok {
		if skip && !prev {
			r.result.runtimeTypes.Set(T, skip)
		}
		return
	}
	r.result.runtimeTypes.Set(T, skip)

	mset := r.prog.MethodSets.MethodSet(T)

	if _, ok := T.Underlying().(*types.Interface); !ok {
		// T is a new concrete type.
		for i, n := 0, mset.Len(); i < n; i++ {
			sel := mset.At(i)
			m := sel.Obj()

			if m.Exported() {
				// Exported methods are always potentially callable via reflection.
				if fn := utils.MethodValueFast(r.prog, sel); fn != nil {
					r.addReachable(fn, true)
				}
			}
		}

		// Add callgraph edges for each existing dynamic "invoke"-mode
		// call via every interface T implements. Iterate per-method
		// groups (iinfo.sitesByMethod): one LookupMethod + addReachable
		// per (T, I, m) — not per site — then attach edges from every
		// site in the group to T's cmethod. Also extend entry.cmethods
		// so subsequent visitInvoke calls at (I, m) reuse it.
		for _, I := range r.interfaces(T) {
			iinfo, _ := r.interfaceTypes.At(I).(*interfaceTypeInfo)
			if iinfo == nil || iinfo.sitesByMethod == nil {
				continue
			}
			g := r.result.callGraph
			// One LookupMethodSeq call per (T, I, method) entry.
			r.invokeLookupsFromAddRuntimeType += len(iinfo.sitesByMethod)
			for _, entry := range iinfo.sitesByMethod {
				cmethod := utils.LookupMethodSeq(r.prog, r.methodValueCache, T, entry.method.Pkg(), entry.method.Name())
				if cmethod == nil {
					continue
				}
				// Sequential: visitInvoke first-time at (I, m) cannot
				// have already added cmethod_T_m, because T was not yet
				// in iinfo.implementations at that earlier moment
				// (interfaces(T) had not yet run; this is its first
				// invocation). So no dedup check is needed.
				r.addReachable(cmethod, true)
				entry.cmethods = append(entry.cmethods, cmethod)
				if g == nil || len(entry.sites) == 0 {
					continue
				}
				to := g.CreateNode(cmethod)
				for _, site := range entry.sites {
					from := g.CreateNode(site.Parent())
					callgraph.AddEdge(from, site, to)
				}
			}
		}
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
	for method := range mset.Methods() {
		if method.Obj().Exported() {
			sig := method.Type().(*types.Signature)
			r.addRuntimeType(sig.Params(), true)  // skip the Tuple itself
			r.addRuntimeType(sig.Results(), true) // skip the Tuple itself
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
		r.addRuntimeType(t.Elem(), false)

	case *types.Slice:
		r.addRuntimeType(t.Elem(), false)

	case *types.Chan:
		r.addRuntimeType(t.Elem(), false)

	case *types.Map:
		r.addRuntimeType(t.Key(), false)
		r.addRuntimeType(t.Elem(), false)

	case *types.Signature:
		if t.Recv() != nil {
			panic(fmt.Sprintf("Signature %s has Recv %s", t, t.Recv()))
		}
		r.addRuntimeType(t.Params(), true)  // skip the Tuple itself
		r.addRuntimeType(t.Results(), true) // skip the Tuple itself

	case *types.Named:
		// A pointer-to-named type can be derived from a named
		// type via reflection.  It may have methods too.
		r.addRuntimeType(types.NewPointer(T), false)

		// Consider 'type T struct{S}' where S has methods.
		// Reflection provides no way to get from T to struct{S},
		// only to S, so the method set of struct{S} is unwanted,
		// so set 'skip' flag during recursion.
		r.addRuntimeType(t.Underlying(), true)

	case *types.Array:
		r.addRuntimeType(t.Elem(), false)

	case *types.Struct:
		for i, n := 0, t.NumFields(); i < n; i++ {
			r.addRuntimeType(t.Field(i).Type(), false)
		}

	case *types.Tuple:
		for i, n := 0, t.Len(); i < n; i++ {
			r.addRuntimeType(t.At(i).Type(), false)
		}

	default:
		panic(T)
	}
}

// implements reports whether types.Implements(cinfo.C, iinfo.I),
// but more efficiently.
func (r *rta) implements(cinfo *concreteTypeInfo, iinfo *interfaceTypeInfo) (got bool) {
	r.implementsCallCount++
	result := iinfo.fprint&^cinfo.fprint == 0 && types.Implements(cinfo.C, iinfo.I)
	if result {
		r.implementsSuccessCount++
	} else {
		r.implementsFailCount++
	}
	return result
}
