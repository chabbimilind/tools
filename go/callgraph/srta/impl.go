// Package srta implements a sequential Rapid Type Analysis (RTA) algorithm.
//
// The core algorithm is adapted from https://cs.opensource.google/go/x/tools/+/v0.43.0:go/callgraph/rta/rta.go
// with the addition of fingerprint-based implements pre-filtering and metrics
// collection.
package srta

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/go/callgraph"
	rtalib "golang.org/x/tools/go/callgraph/internal/rtautil"
	"golang.org/x/tools/go/callgraph/internal/rtautil/utils"
	rtapkg "golang.org/x/tools/go/callgraph/rta"
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
	// Keys are *types.Signature, values are map[*ssa.Function]bool sets.
	addrTakenFuncsBySig typeutil.Map

	// dynCallSites contains all dynamic "call"-mode call sites, grouped by signature.
	// Keys are *types.Signature, values are unordered []ssa.CallInstruction.
	dynCallSites typeutil.Map

	// invokeSites contains all "invoke"-mode call sites, grouped by interface.
	// Keys are *types.Interface (never *types.Named),
	// Values are unordered []ssa.CallInstruction sets.
	invokeSites typeutil.Map

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

	// invokeLookupsFromVisitInvoke counts prog.LookupMethod calls
	// performed inside visitInvoke. One call per (site, impl) pair (every
	// addInvokeEdge does one LookupMethod). Counted as len(impls) per
	// visitInvoke instead of an in-loop increment. This is the baseline
	// value that sitesByMethod-optimized flavors reduce.
	invokeLookupsFromVisitInvoke int
	// invokeLookupsFromAddRuntimeType counts prog.LookupMethod calls
	// performed inside addRuntimeType. One call per (T, I, site) tuple;
	// counted as len(sites) per interface.
	invokeLookupsFromAddRuntimeType int
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
	// Create two-level map (Signature -> Function -> bool).
	S := f.Signature
	funcs, _ := r.addrTakenFuncsBySig.At(S).(map[*ssa.Function]bool)
	if funcs == nil {
		funcs = make(map[*ssa.Function]bool)
		r.addrTakenFuncsBySig.Set(S, funcs)
	}
	if !funcs[f] {
		// First time seeing f.
		funcs[f] = true

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
	funcs, _ := r.addrTakenFuncsBySig.At(S).(map[*ssa.Function]bool)
	for g := range funcs {
		r.addEdge(site.Parent(), site, g, true)
	}
}

// ---------- concrete types × invoke sites ----------

// addInvokeEdge is called for each new pair (site, C) in the matrix.
func (r *rta) addInvokeEdge(site ssa.CallInstruction, C types.Type) {
	// Ascertain the concrete method of C to be called.
	imethod := site.Common().Method
	cmethod := r.prog.LookupMethod(C, imethod.Pkg(), imethod.Name())
	r.addEdge(site.Parent(), site, cmethod, true)
}

// visitInvoke is called each time the algorithm encounters an "invoke"-mode call.
func (r *rta) visitInvoke(site ssa.CallInstruction) {
	I := site.Common().Value.Type().Underlying().(*types.Interface)

	// Record the invoke site.
	sites, _ := r.invokeSites.At(I).([]ssa.CallInstruction)
	r.invokeSites.Set(I, append(sites, site))

	// Add callgraph edge for each existing
	// address-taken concrete type implementing I.
	impls := r.implementations(I)
	r.invokeLookupsFromVisitInvoke += len(impls)
	for _, C := range impls {
		r.addInvokeEdge(site, C)
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
		reachable: make(map[*ssa.Function]struct{ AddrTaken bool }),
		result:    &Result{},
		prog:      roots[0].Prog,
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
		r.reflectValueCall = r.prog.LookupMethod(reflectValue.Object().Type(), reflectPkg.Pkg, "Call")
	}

	hasher := typeutil.MakeHasher()
	r.result.runtimeTypes.SetHasher(hasher)
	r.addrTakenFuncsBySig.SetHasher(hasher)
	r.dynCallSites.SetHasher(hasher)
	r.invokeSites.SetHasher(hasher)
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
				r.addReachable(r.prog.MethodValue(sel), true)
			}
		}

		// Add callgraph edge for each existing dynamic
		// "invoke"-mode call via that interface.
		for _, I := range r.interfaces(T) {
			sites, _ := r.invokeSites.At(I).([]ssa.CallInstruction)
			r.invokeLookupsFromAddRuntimeType += len(sites)
			for _, site := range sites {
				r.addInvokeEdge(site, T)
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
