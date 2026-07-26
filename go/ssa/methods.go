// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for population of method sets.

import (
	"fmt"
	"go/types"
	"sync"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typesinternal"
)

// MethodValue returns the Function implementing method sel, building
// wrapper methods on demand. It returns nil if sel denotes an
// interface or generic method.
//
// Precondition: sel.Kind() == MethodVal.
//
// Thread-safe.
//
// Acquires prog.methodsMu briefly to look up or create the per-T *methodSet,
// then a per-(*methodSet) sync.Mutex for the actual method mapping. The cold
// compute (createWrapper / objectMethod) runs under the per-T lock only,
// eliminating the previous global-mutex bottleneck where every worker that
// reached MethodValue serialized behind every other worker's wrapper synthesis.
func (prog *Program) MethodValue(sel *types.Selection) *Function {
	if sel.Kind() != types.MethodVal {
		panic(fmt.Sprintf("MethodValue(%s) kind != MethodVal", sel))
	}
	T := sel.Recv()
	if types.IsInterface(T) {
		return nil // interface method or type parameter
	}

	if prog.isParameterized(T, sel.Type()) {
		return nil // method on generic type or generic method
	}

	if prog.mode&LogSource != 0 {
		defer logStack("MethodValue %s %v", T, sel)()
	}

	var b builder

	// Step 1: get or create the *methodSet for T. Hold prog.methodsMu only
	// for the brief typeutil.Map probe — never for the cold compute below.
	prog.methodsMu.Lock()
	mset, ok := prog.methodSets.At(T).(*methodSet)
	if !ok {
		mset = &methodSet{mapping: make(map[string]*Function)}
		prog.methodSets.Set(T, mset)
	}
	prog.methodsMu.Unlock()

	// Step 2: per-T lock guards mset.mapping and serializes wrapper synthesis
	// only for callers that share the same receiver type. Different Ts proceed
	// in parallel.
	mset.mu.Lock()
	id := sel.Obj().Id()
	fn, exists := mset.mapping[id]
	if !exists {
		obj := sel.Obj().(*types.Func)
		needsPromotion := len(sel.Index()) > 1
		needsIndirection := !isPointer(recvType(obj)) && isPointer(T)
		if needsPromotion || needsIndirection {
			fn = createWrapper(prog, toSelection(sel), nil)
			fn.buildshared = b.shared()
			b.enqueue(fn)
		} else {
			fn = prog.objectMethod(obj, nil, &b)
		}
		if fn.Signature.Recv() == nil {
			panic(fn)
		}
		mset.mapping[id] = fn
	}
	mset.mu.Unlock()

	// waitForSharedFunction must run outside the per-T lock: it just records
	// a task-graph edge into the local builder, but holding the lock across
	// it would re-introduce the serialization we are eliminating.
	if exists {
		b.waitForSharedFunction(fn)
	}

	b.iterate()

	return fn
}

// objectMethod returns the Function for a given method symbol.
// The symbol may be an instance of a generic function. It need not
// belong to an existing SSA package created by a call to
// prog.CreatePackage.
//
// objectMethod panics if the function is not a method.
//
// Acquires prog.objectMethodsMu.
func (prog *Program) objectMethod(obj *types.Func, targs []types.Type, b *builder) *Function {
	sig := obj.Type().(*types.Signature)
	if sig.Recv() == nil {
		panic("not a method: " + obj.String())
	}

	// Instantiation of generic?
	if orig := obj.Origin(); orig != obj || len(targs) > 0 {
		return prog.objectMethod(orig, nil, b).instance(receiverTypeArgs(obj), targs, b)
	}

	// Belongs to a created package?
	if fn := prog.FuncValue(obj); fn != nil {
		return fn
	}

	// Consult/update cache of methods created from types.Func.
	prog.objectMethodsMu.Lock()
	defer prog.objectMethodsMu.Unlock()
	fn, ok := prog.objectMethods[obj]
	if !ok {
		fn = createFunction(prog, obj, obj.Name(), nil, nil, "")
		fn.Synthetic = "from type information (on demand)"
		fn.buildshared = b.shared()
		b.enqueue(fn)

		if prog.objectMethods == nil {
			prog.objectMethods = make(map[*types.Func]*Function)
		}
		prog.objectMethods[obj] = fn
	} else {
		b.waitForSharedFunction(fn)
	}
	return fn
}

// LookupMethod returns the implementation of the method of type T
// identified by (pkg, name).  It returns nil if the method exists but
// is an interface method or generic method, and panics if T has no such method.
func (prog *Program) LookupMethod(T types.Type, pkg *types.Package, name string) *Function {
	sel := prog.MethodSets.MethodSet(T).Lookup(pkg, name)
	if sel == nil {
		panic(fmt.Sprintf("%s has no method %s", T, types.Id(pkg, name)))
	}
	return prog.MethodValue(sel)
}

// methodSet contains the (concrete) methods of a concrete type (non-interface, non-parameterized).
type methodSet struct {
	mu      sync.Mutex           // guards mapping
	mapping map[string]*Function // populated lazily
}

// RuntimeTypes returns a new unordered slice containing all types in
// the program for which a runtime type is required.
//
// A runtime type is required for any non-parameterized, non-interface
// type that is converted to an interface, or for any type (including
// interface types) derivable from one through reflection.
//
// The methods of such types may be reachable through reflection or
// interface calls even if they are never called directly.
//
// Thread-safe.
//
// Acquires prog.makeInterfaceTypesMu.
func (prog *Program) RuntimeTypes() []types.Type {
	prog.makeInterfaceTypesMu.Lock()
	defer prog.makeInterfaceTypesMu.Unlock()

	// Compute the derived types on demand, since many SSA clients
	// never call RuntimeTypes, and those that do typically call
	// it once (often within ssautil.AllFunctions, which will
	// eventually not use it; see Go issue #69291.) This
	// eliminates the need to eagerly compute all the element
	// types during SSA building.
	var runtimeTypes []types.Type
	add := func(t types.Type) { runtimeTypes = append(runtimeTypes, t) }
	var set typeutil.Map // for de-duping identical types
	for t := range prog.makeInterfaceTypes {
		typesinternal.ForEachElement(&set, &prog.MethodSets, t, add)
	}

	return runtimeTypes
}
