package utils

import (
	"go/token"
	"go/types"
	"testing"
)

func TestFingerprint_EmptyMethodSet(t *testing.T) {
	// Empty struct has no methods
	empty := types.NewStruct(nil, nil)
	mset := types.NewMethodSet(empty)
	fp := Fingerprint(mset)
	if fp != 0 {
		t.Fatalf("expected 0 fingerprint for empty method set, got %d", fp)
	}
}

func TestFingerprint_Deterministic(t *testing.T) {
	// Create a named type with a method
	pkg := types.NewPackage("test/pkg", "pkg")
	named := types.NewNamed(types.NewTypeName(token.NoPos, pkg, "MyType", nil), types.NewStruct(nil, nil), nil)
	sig := types.NewSignatureType(types.NewVar(token.NoPos, pkg, "recv", named), nil, nil, types.NewTuple(), types.NewTuple(), false)
	named.AddMethod(types.NewFunc(token.NoPos, pkg, "Foo", sig))

	mset := types.NewMethodSet(named)
	fp1 := Fingerprint(mset)
	fp2 := Fingerprint(mset)
	if fp1 != fp2 {
		t.Fatalf("fingerprint not deterministic: %d != %d", fp1, fp2)
	}
	if fp1 == 0 {
		t.Fatal("fingerprint should be non-zero for non-empty method set")
	}
}

func TestFingerprint_DifferentMethods(t *testing.T) {
	pkg := types.NewPackage("test/pkg", "pkg")

	// Type with method "Foo"
	named1 := types.NewNamed(types.NewTypeName(token.NoPos, pkg, "Type1", nil), types.NewStruct(nil, nil), nil)
	sig1 := types.NewSignatureType(types.NewVar(token.NoPos, pkg, "recv", named1), nil, nil, types.NewTuple(), types.NewTuple(), false)
	named1.AddMethod(types.NewFunc(token.NoPos, pkg, "Foo", sig1))

	// Type with method "Bar"
	named2 := types.NewNamed(types.NewTypeName(token.NoPos, pkg, "Type2", nil), types.NewStruct(nil, nil), nil)
	sig2 := types.NewSignatureType(types.NewVar(token.NoPos, pkg, "recv", named2), nil, nil, types.NewTuple(), types.NewTuple(), false)
	named2.AddMethod(types.NewFunc(token.NoPos, pkg, "Bar", sig2))

	fp1 := Fingerprint(types.NewMethodSet(named1))
	fp2 := Fingerprint(types.NewMethodSet(named2))
	// Different methods should (usually) produce different fingerprints
	// This isn't guaranteed due to hash collisions, but is very likely for these names
	if fp1 == fp2 {
		t.Log("warning: fingerprints collided for different methods (unlikely but possible)")
	}
}

func TestFingerprint_SubsetProperty(t *testing.T) {
	// If I is a subset of C's methods, then I.fprint & ^C.fprint == 0
	pkg := types.NewPackage("test/pkg", "pkg")

	// Create concrete type with methods Foo and Bar
	concrete := types.NewNamed(types.NewTypeName(token.NoPos, pkg, "Concrete", nil), types.NewStruct(nil, nil), nil)
	sigFoo := types.NewSignatureType(types.NewVar(token.NoPos, pkg, "recv", concrete), nil, nil, types.NewTuple(), types.NewTuple(), false)
	sigBar := types.NewSignatureType(types.NewVar(token.NoPos, pkg, "recv", concrete), nil, nil, types.NewTuple(), types.NewTuple(), false)
	concrete.AddMethod(types.NewFunc(token.NoPos, pkg, "Foo", sigFoo))
	concrete.AddMethod(types.NewFunc(token.NoPos, pkg, "Bar", sigBar))

	// Create interface with just Foo
	ifaceFoo := types.NewInterfaceType([]*types.Func{
		types.NewFunc(token.NoPos, pkg, "Foo",
			types.NewSignatureType(nil, nil, nil, types.NewTuple(), types.NewTuple(), false)),
	}, nil)
	ifaceFoo.Complete()

	cfp := Fingerprint(types.NewMethodSet(concrete))
	ifp := Fingerprint(types.NewMethodSet(ifaceFoo))

	// Subset property: interface fingerprint bits should be a subset of concrete's
	if ifp&^cfp != 0 {
		t.Fatalf("subset property violated: interface fp %064b, concrete fp %064b", ifp, cfp)
	}
}
