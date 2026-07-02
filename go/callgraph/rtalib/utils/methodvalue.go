package utils

import (
	"go/types"

	"github.com/puzpuzpuz/xsync/v4"
	"golang.org/x/tools/go/ssa"
)

// MethodValueFast resolves a *types.Selection to an *ssa.Function.
// For direct methods (no promotion or indirection), it uses prog.FuncValue
// which is a lock-free read-only map lookup, avoiding prog.methodsMu contention.
// Falls back to prog.MethodValue for promoted/indirected methods that need wrappers.
func MethodValueFast(prog *ssa.Program, sel *types.Selection) *ssa.Function {
	obj := sel.Obj().(*types.Func)
	needsPromotion := len(sel.Index()) > 1
	if !needsPromotion {
		recv := obj.Signature().Recv().Type()
		_, recvIsPtr := types.Unalias(recv).(*types.Pointer)
		_, selIsPtr := types.Unalias(sel.Recv()).(*types.Pointer)
		needsIndirection := !recvIsPtr && selIsPtr
		if !needsIndirection {
			if fn := prog.FuncValue(obj); fn != nil {
				return fn
			}
		}
	}
	return prog.MethodValue(sel)
}

// MethodKey is a composite key for the method value cache.
type MethodKey struct {
	Typ  types.Type
	Pkg  *types.Package
	Name string
}

// LookupMethod resolves a method on type T, caching results in the provided xsync.Map.
//
// Cold-path resolution uses types.LookupSelection (Go 1.25+) which walks the type
// graph and short-circuits on the first match, avoiding the global mutex inside
// typeutil.MethodSetCache that bottlenecks parallel RTA at high worker counts.
func LookupMethod(prog *ssa.Program, cache *xsync.Map[MethodKey, *ssa.Function], T types.Type, pkg *types.Package, name string) *ssa.Function {
	key := MethodKey{Typ: T, Pkg: pkg, Name: name}
	if cached, ok := cache.Load(key); ok {
		return cached
	}
	sel, ok := types.LookupSelection(T, false, pkg, name)
	if !ok {
		cache.Store(key, nil)
		return nil
	}
	result := MethodValueFast(prog, &sel)
	cache.Store(key, result)
	return result
}

// LookupMethodSeq is the sequential-only counterpart to LookupMethod: it caches
// in a plain map to avoid xsync.Map atomic overhead. Use only from a single
// goroutine. The cache must be non-nil.
func LookupMethodSeq(prog *ssa.Program, cache map[MethodKey]*ssa.Function, T types.Type, pkg *types.Package, name string) *ssa.Function {
	key := MethodKey{Typ: T, Pkg: pkg, Name: name}
	if cached, ok := cache[key]; ok {
		return cached
	}
	sel, ok := types.LookupSelection(T, false, pkg, name)
	if !ok {
		cache[key] = nil
		return nil
	}
	result := MethodValueFast(prog, &sel)
	cache[key] = result
	return result
}
