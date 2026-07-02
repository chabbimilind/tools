// Package srta_opt is a sequential RTA built on top of [srta] with two
// straightforward optimizations:
//
//   - A memoized method-value lookup ([utils.LookupMethodSeq] +
//     [utils.MethodValueFast]) that short-circuits the per-call
//     MethodSet.Lookup + MethodValue work, which dominates addInvokeEdge
//     on large services.
//
//   - Per-method indexing of invoke sites: instead of a flat
//     invokeSites[I] = []site list, each interface keeps a
//     methodName → {sites, cmethods} table. The first invoke site at
//     (I, m) resolves cmethod_C_m + addReachable once per impl C and
//     caches the result; subsequent sites at (I, m) skip both
//     LookupMethod and addReachable and only attach a callgraph edge per
//     cached cmethod. When a new concrete T becomes reachable,
//     addRuntimeType iterates per-method groups (not flat sites) and
//     does a single LookupMethod + addReachable per (T, I, m), then
//     attaches edges from every site in the group to T's cmethod.
//
// Kept as a separate flavor so [srta] can serve as the unmodified
// baseline for benchmark comparisons.
package srta_opt

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/go/callgraph"
	rtapkg "golang.org/x/tools/go/callgraph/rta"
	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/rtalib/utils"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types/typeutil"
)

// Result implements the rta.Result interface for sequential RTA.
// It uses regular typeutil.Map for optimal sequential performance.
type Result struct {
	rtalib.BaseMetrics
	callGraph    *callgraph.Graph
	reachable    map[*ssa.Function]struct{ AddrTaken bool }
	runtimeTypes typeutil.Map
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

// Working state of the RTA algorithm.
type rta struct {
	// Use internal map during analysis
	reachable map[*ssa.Function]struct{ AddrTaken bool }

	result *Result

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

	implementsCallCount    int
	implementsSuccessCount int
	implementsFailCount    int

	checksFromInterfaces       int
	checksFromImplementations  int
	failsFromInterfaces        int
	successFromInterfaces      int
	failsFromImplementations   int
	successFromImplementations int

	staticCallSites   int
	indirectCallSites int
	invokeCallSites   int

	// invokeLookupsFromVisitInvoke counts utils.LookupMethodSeq calls
	// performed inside visitInvoke. With per-method caching only the
	// first-time path at (I, method) actually does lookups (one per
	// impl); subsequent sites at the same (I, method) reuse the cached
	// cmethods list and do zero lookups. Compared to the same metric in
	// a non-sitesByMethod flavor on the same service, the difference is
	// exactly the per-site lookup work that sitesByMethod eliminates.
	invokeLookupsFromVisitInvoke int
	// invokeLookupsFromAddRuntimeType counts utils.LookupMethodSeq calls
	// performed inside addRuntimeType. With per-method caching, one
	// lookup per (T, I, method) entry — independent of how many sites
	// the entry holds. A non-sitesByMethod flavor would do one lookup
	// per (T, I, site), so this metric again exposes the savings.
	invokeLookupsFromAddRuntimeType int

	// methodValueCache memoizes prog.LookupMethod results, avoiding repeated
	// MethodSet.Lookup + MethodValue work that dominates addInvokeEdge on
	// large services.
	methodValueCache map[utils.MethodKey]*ssa.Function
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
	n := len(r.reachable)
	v := r.reachable[f]
	if addrTaken {
		v.AddrTaken = true
	}
	r.reachable[f] = v
	if len(r.reachable) > n {
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
// "invoke"-mode call site. See package doc for the per-method indexing
// optimization. Sequential execution means we don't need a per-entry
// mutex: visitInvoke and addRuntimeType cannot interleave, so
// `entry.initialized` alone (no cmethodSet) is enough to ensure each
// cmethod is added to entry.cmethods at most once.
func (r *rta) visitInvoke(site ssa.CallInstruction) {
	I := site.Common().Value.Type().Underlying().(*types.Interface)
	imethod := site.Common().Method

	// Compute implementations(I); also ensures iinfo exists in the map.
	impls := r.implementations(I)
	iinfo := r.interfaceTypes.At(I).(*interfaceTypeInfo)

	// Get (or create) the per-method entry and append site.
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
		// of I, mark each reachable, cache them, and attach the edge
		// from this site to each.
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
		// First-time path: one LookupMethodSeq call per impl iterated
		// above. Counted once via len(impls) instead of an in-loop Inc.
		r.invokeLookupsFromVisitInvoke += len(impls)
		return
	}

	// methodName already seen on I — every cmethod in entry.cmethods is
	// already reachable (by visitInvoke first-time or by addRuntimeType
	// extending it for new impls). Just attach edges from this site;
	// no LookupMethodSeq call happens on this path.
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

// RTA is a sequential RTA implementation.
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

	r := &rta{
		reachable:        make(map[*ssa.Function]struct{ AddrTaken bool }),
		result:           &Result{},
		prog:             roots[0].Prog,
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

	return r.result
}

// interfaces(C) returns all currently known interfaces implemented by C.
func (r *rta) interfaces(C types.Type) []*types.Interface {
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

		// Ascertain set of interfaces C implements
		// and update the 'implements' relation.
		r.interfaceTypes.Iterate(func(I types.Type, v any) {
			iinfo := v.(*interfaceTypeInfo)
			r.checksFromInterfaces++
			if I := types.Unalias(I).(*types.Interface); r.implements(cinfo, iinfo) {
				iinfo.implementations = append(iinfo.implementations, C)
				cinfo.implements = append(cinfo.implements, I)
				r.successFromInterfaces++
			} else {
				r.failsFromInterfaces++
			}
		})
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

		// Ascertain set of concrete types that implement I
		// and update the 'implements' relation.
		r.concreteTypes.Iterate(func(C types.Type, v any) {
			cinfo := v.(*concreteTypeInfo)
			r.checksFromImplementations++
			if r.implements(cinfo, iinfo) {
				cinfo.implements = append(cinfo.implements, I)
				iinfo.implementations = append(iinfo.implementations, C)
				r.successFromImplementations++
			} else {
				r.failsFromImplementations++
			}
		})
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
			// One LookupMethodSeq call per (T, I, method) entry in the
			// loop below. Counted once via len() per (T, I).
			r.invokeLookupsFromAddRuntimeType += len(iinfo.sitesByMethod)
			for _, entry := range iinfo.sitesByMethod {
				cmethod := utils.LookupMethodSeq(r.prog, r.methodValueCache, T, entry.method.Pkg(), entry.method.Name())
				if cmethod == nil {
					continue
				}
				// In sequential execution this is the only path that
				// can introduce cmethod for (T, I, m) — visitInvoke
				// first-time at (I, m) can only have added cmethod for
				// concretes that were already in iinfo.implementations
				// at that earlier moment, which by causal order
				// excluded T (interfaces(T) had not yet run). So no
				// dedup check is needed.
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
	result := iinfo.fprint & ^cinfo.fprint == 0 && types.Implements(cinfo.C, iinfo.I)
	if result {
		r.implementsSuccessCount++
	} else {
		r.implementsFailCount++
	}
	return result
}
