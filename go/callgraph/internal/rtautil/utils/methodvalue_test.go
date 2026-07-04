package utils

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"github.com/puzpuzpuz/xsync/v4"
	"golang.org/x/tools/go/ssa"
)

func buildTestSSA(t *testing.T, src string) (*ssa.Program, *types.Package) {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf types.Config
	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	pkg, err := conf.Check("test/pkg", fset, []*ast.File{file}, info)
	if err != nil {
		t.Fatal(err)
	}

	prog := ssa.NewProgram(fset, ssa.InstantiateGenerics)
	ssaPkg := prog.CreatePackage(pkg, []*ast.File{file}, info, true)
	ssaPkg.Build()
	return prog, pkg
}

func TestMethodValueFast_DirectMethod(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) DirectMethod() {}
`
	prog, pkg := buildTestSSA(t, src)

	myType := pkg.Scope().Lookup("MyType").Type()
	mset := types.NewMethodSet(myType)
	if mset.Len() == 0 {
		t.Fatal("expected methods on MyType")
	}

	sel := mset.At(0)
	fn := MethodValueFast(prog, sel)
	if fn == nil {
		t.Fatal("MethodValueFast returned nil for direct method")
	}
	if fn.Name() != "DirectMethod" {
		t.Fatalf("expected DirectMethod, got %s", fn.Name())
	}
}

func TestMethodValueFast_PointerReceiver(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m *MyType) PtrMethod() {}
`
	prog, pkg := buildTestSSA(t, src)

	myType := pkg.Scope().Lookup("MyType").Type()
	ptrType := types.NewPointer(myType)
	mset := types.NewMethodSet(ptrType)
	if mset.Len() == 0 {
		t.Fatal("expected methods on *MyType")
	}

	sel := mset.At(0)
	fn := MethodValueFast(prog, sel)
	if fn == nil {
		t.Fatal("MethodValueFast returned nil for pointer receiver method")
	}
	if fn.Name() != "PtrMethod" {
		t.Fatalf("expected PtrMethod, got %s", fn.Name())
	}
}

func TestMethodValueFast_PromotedMethod(t *testing.T) {
	const src = `
package pkg

type Base struct{}

func (b Base) BaseMethod() {}

type Outer struct {
	Base
}
`
	prog, pkg := buildTestSSA(t, src)

	outerType := pkg.Scope().Lookup("Outer").Type()
	mset := types.NewMethodSet(outerType)

	var sel *types.Selection
	for i := 0; i < mset.Len(); i++ {
		if mset.At(i).Obj().Name() == "BaseMethod" {
			sel = mset.At(i)
			break
		}
	}
	if sel == nil {
		t.Fatal("BaseMethod not found in Outer's method set")
	}
	if len(sel.Index()) <= 1 {
		t.Fatal("expected promoted method with Index > 1")
	}

	fn := MethodValueFast(prog, sel)
	if fn == nil {
		t.Fatal("MethodValueFast returned nil for promoted method")
	}
}

func TestMethodValueFast_IndirectedMethod(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) ValMethod() {}
`
	prog, pkg := buildTestSSA(t, src)

	// Access value receiver method via pointer type (*MyType).ValMethod
	myType := pkg.Scope().Lookup("MyType").Type()
	ptrType := types.NewPointer(myType)
	mset := types.NewMethodSet(ptrType)

	var sel *types.Selection
	for i := 0; i < mset.Len(); i++ {
		if mset.At(i).Obj().Name() == "ValMethod" {
			sel = mset.At(i)
			break
		}
	}
	if sel == nil {
		t.Fatal("ValMethod not found in *MyType's method set")
	}

	fn := MethodValueFast(prog, sel)
	if fn == nil {
		t.Fatal("MethodValueFast returned nil for indirected method")
	}
}

func TestLookupMethod(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) Foo() {}
`
	prog, pkg := buildTestSSA(t, src)

	myType := pkg.Scope().Lookup("MyType").Type()

	cache := xsync.NewMap[MethodKey, *ssa.Function]()

	// First call: cache miss
	fn1 := LookupMethod(prog, cache, myType, pkg, "Foo")
	if fn1 == nil {
		t.Fatal("LookupMethod returned nil")
	}
	if fn1.Name() != "Foo" {
		t.Fatalf("expected Foo, got %s", fn1.Name())
	}

	// Second call: cache hit
	fn2 := LookupMethod(prog, cache, myType, pkg, "Foo")
	if fn2 != fn1 {
		t.Fatal("LookupMethod cache miss on second call")
	}
}

func TestLookupMethod_NotFound(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) Foo() {}
`
	prog, pkg := buildTestSSA(t, src)
	myType := pkg.Scope().Lookup("MyType").Type()
	cache := xsync.NewMap[MethodKey, *ssa.Function]()

	// First call: method does not exist; nil is cached.
	fn := LookupMethod(prog, cache, myType, pkg, "DoesNotExist")
	if fn != nil {
		t.Fatalf("expected nil for missing method, got %v", fn)
	}

	// Second call: cache hit returns the cached nil without re-resolving.
	if got := LookupMethod(prog, cache, myType, pkg, "DoesNotExist"); got != nil {
		t.Fatalf("expected cached nil, got %v", got)
	}
}

func TestLookupMethodSeq(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) Foo() {}
`
	prog, pkg := buildTestSSA(t, src)
	myType := pkg.Scope().Lookup("MyType").Type()
	cache := make(map[MethodKey]*ssa.Function)

	fn1 := LookupMethodSeq(prog, cache, myType, pkg, "Foo")
	if fn1 == nil {
		t.Fatal("LookupMethodSeq returned nil")
	}
	if fn1.Name() != "Foo" {
		t.Fatalf("expected Foo, got %s", fn1.Name())
	}
	if len(cache) != 1 {
		t.Fatalf("expected 1 cache entry, got %d", len(cache))
	}

	fn2 := LookupMethodSeq(prog, cache, myType, pkg, "Foo")
	if fn2 != fn1 {
		t.Fatal("LookupMethodSeq cache miss on second call")
	}
	if len(cache) != 1 {
		t.Fatalf("cache grew on hit: got %d entries", len(cache))
	}
}

func TestLookupMethodSeq_NotFound(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) Foo() {}
`
	prog, pkg := buildTestSSA(t, src)
	myType := pkg.Scope().Lookup("MyType").Type()
	cache := make(map[MethodKey]*ssa.Function)

	fn := LookupMethodSeq(prog, cache, myType, pkg, "DoesNotExist")
	if fn != nil {
		t.Fatalf("expected nil for missing method, got %v", fn)
	}

	key := MethodKey{Typ: myType, Pkg: pkg, Name: "DoesNotExist"}
	if _, ok := cache[key]; !ok {
		t.Fatal("expected negative result to be cached")
	}

	if got := LookupMethodSeq(prog, cache, myType, pkg, "DoesNotExist"); got != nil {
		t.Fatalf("expected cached nil, got %v", got)
	}
}

func TestLookupMethodSeq_PromotedMethod(t *testing.T) {
	const src = `
package pkg

type Base struct{}

func (b Base) BaseMethod() {}

type Outer struct {
	Base
}
`
	prog, pkg := buildTestSSA(t, src)
	outerType := pkg.Scope().Lookup("Outer").Type()
	cache := make(map[MethodKey]*ssa.Function)

	fn := LookupMethodSeq(prog, cache, outerType, pkg, "BaseMethod")
	if fn == nil {
		t.Fatal("LookupMethodSeq returned nil for promoted method")
	}
	if fn.Name() != "BaseMethod" {
		t.Fatalf("expected BaseMethod, got %s", fn.Name())
	}
}

func TestLookupMethodSeq_PointerReceiver(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m *MyType) PtrMethod() {}
`
	prog, pkg := buildTestSSA(t, src)
	ptrType := types.NewPointer(pkg.Scope().Lookup("MyType").Type())
	cache := make(map[MethodKey]*ssa.Function)

	fn := LookupMethodSeq(prog, cache, ptrType, pkg, "PtrMethod")
	if fn == nil {
		t.Fatal("LookupMethodSeq returned nil for pointer receiver method")
	}
	if fn.Name() != "PtrMethod" {
		t.Fatalf("expected PtrMethod, got %s", fn.Name())
	}
}

func TestLookupMethodSeq_MatchesLookupMethod(t *testing.T) {
	const src = `
package pkg

type MyType struct{}

func (m MyType) Foo() {}
func (m MyType) Bar() {}
func (m *MyType) PtrOnly() {}

type Base struct{}

func (b Base) Inherited() {}

type Outer struct{ Base }
`
	prog, pkg := buildTestSSA(t, src)
	myType := pkg.Scope().Lookup("MyType").Type()
	myPtr := types.NewPointer(myType)
	outer := pkg.Scope().Lookup("Outer").Type()

	cases := []struct {
		T    types.Type
		name string
	}{
		{myType, "Foo"},
		{myType, "Bar"},
		{myType, "PtrOnly"}, // value receiver, ptr-only method => should resolve to nil
		{myPtr, "PtrOnly"},
		{outer, "Inherited"},
		{myType, "Missing"},
	}

	seqCache := make(map[MethodKey]*ssa.Function)
	xsyncCache := xsync.NewMap[MethodKey, *ssa.Function]()
	for _, c := range cases {
		got := LookupMethodSeq(prog, seqCache, c.T, pkg, c.name)
		want := LookupMethod(prog, xsyncCache, c.T, pkg, c.name)
		if got != want {
			t.Errorf("Lookup(%v, %q): seq=%v xsync=%v", c.T, c.name, got, want)
		}
	}
}
