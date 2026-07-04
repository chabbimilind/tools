package utils

import (
	"testing"
	"unsafe"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
)

func makeEdge(caller, callee *callgraph.Node) *callgraph.Edge {
	return &callgraph.Edge{Caller: caller, Callee: callee}
}

func TestDedupEdges_NoDuplicates(t *testing.T) {
	n1 := &callgraph.Node{Func: &ssa.Function{}}
	n2 := &callgraph.Node{Func: &ssa.Function{}}
	n3 := &callgraph.Node{Func: &ssa.Function{}}
	e1 := makeEdge(n1, n2)
	e2 := makeEdge(n1, n3)
	result := DedupEdges([]*callgraph.Edge{e1, e2})
	if len(result) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(result))
	}
}

func TestDedupEdges_WithDuplicates(t *testing.T) {
	caller := &callgraph.Node{Func: &ssa.Function{}}
	callee := &callgraph.Node{Func: &ssa.Function{}}
	e1 := &callgraph.Edge{Caller: caller, Callee: callee}
	e2 := &callgraph.Edge{Caller: caller, Callee: callee}
	result := DedupEdges([]*callgraph.Edge{e1, e2})
	if len(result) != 1 {
		t.Fatalf("expected 1 edge after dedup, got %d", len(result))
	}
}

func TestDedupEdges_NilEdges(t *testing.T) {
	n1 := &callgraph.Node{Func: &ssa.Function{}}
	n2 := &callgraph.Node{Func: &ssa.Function{}}
	e1 := makeEdge(n1, n2)
	result := DedupEdges([]*callgraph.Edge{nil, e1, nil})
	if len(result) != 1 {
		t.Fatalf("expected 1 edge, got %d", len(result))
	}
}

func TestDedupEdges_Empty(t *testing.T) {
	result := DedupEdges(nil)
	if len(result) != 0 {
		t.Fatalf("expected 0 edges, got %d", len(result))
	}
}

func TestDedupEdgesLarge(t *testing.T) {
	// Create > dedupEdgesSmallThreshold edges with some duplicates
	caller := &callgraph.Node{Func: &ssa.Function{}}
	var edges []*callgraph.Edge
	callees := make([]*callgraph.Node, 30)
	for i := range callees {
		callees[i] = &callgraph.Node{Func: &ssa.Function{}}
	}
	for _, c := range callees {
		edges = append(edges, &callgraph.Edge{Caller: caller, Callee: c})
	}
	// Add duplicates of the first 5
	for i := 0; i < 5; i++ {
		edges = append(edges, &callgraph.Edge{Caller: caller, Callee: callees[i]})
	}
	result := DedupEdges(edges)
	if len(result) != 30 {
		t.Fatalf("expected 30 unique edges, got %d", len(result))
	}
}

func TestWorkerCounter(t *testing.T) {
	wc := NewWorkerCounter(4)
	wc.Inc(0)
	wc.Inc(1)
	wc.Inc(1)
	wc.Add(2, 5)
	wc.Inc(3)
	if got := wc.Sum(); got != 9 {
		t.Fatalf("expected sum 9, got %d", got)
	}
}

func TestPaddedCounterSize(t *testing.T) {
	// Verify cache line padding
	var pc PaddedCounter
	if sz := int(unsafe.Sizeof(pc)); sz != 64 {
		t.Fatalf("PaddedCounter should be 64 bytes, got %d", sz)
	}
}
