package utils

import (
	"sync"
	"testing"

	"golang.org/x/tools/go/ssa"
)

func TestConcurrentCallGraph_CreateNode(t *testing.T) {
	root := &ssa.Function{}
	g := NewConcurrentCallGraph(root)

	// Root node should already exist
	rootNode := g.CreateNode(root)
	if rootNode == nil {
		t.Fatal("root node should exist")
	}
	if rootNode.ID != 0 {
		t.Fatalf("root node ID should be 0, got %d", rootNode.ID)
	}

	// Create a new node
	fn1 := &ssa.Function{}
	node1 := g.CreateNode(fn1)
	if node1 == nil {
		t.Fatal("node1 should be created")
	}
	if node1.ID != 1 {
		t.Fatalf("node1 ID should be 1, got %d", node1.ID)
	}

	// Same function returns same node
	node1Again := g.CreateNode(fn1)
	if node1Again != node1 {
		t.Fatal("CreateNode should return same node for same function")
	}
}

func TestConcurrentCallGraph_AddCallGraphEdge(t *testing.T) {
	root := &ssa.Function{}
	g := NewConcurrentCallGraph(root)

	fn1 := &ssa.Function{}
	fn2 := &ssa.Function{}
	caller := g.CreateNode(fn1)
	callee := g.CreateNode(fn2)

	g.AddCallGraphEdge(caller, nil, callee)

	if len(caller.Out) != 1 {
		t.Fatalf("caller should have 1 outgoing edge, got %d", len(caller.Out))
	}
	if len(callee.In) != 1 {
		t.Fatalf("callee should have 1 incoming edge, got %d", len(callee.In))
	}
	if caller.Out[0].Callee != callee {
		t.Fatal("edge should point to callee")
	}
}

func TestConcurrentCallGraph_GetGraph(t *testing.T) {
	root := &ssa.Function{}
	g := NewConcurrentCallGraph(root)

	fn1 := &ssa.Function{}
	fn2 := &ssa.Function{}
	g.CreateNode(fn1)
	g.CreateNode(fn2)

	graph := g.GetGraph()
	if graph.Root == nil {
		t.Fatal("graph should have root")
	}
	if len(graph.Nodes) != 3 {
		t.Fatalf("graph should have 3 nodes (root + 2), got %d", len(graph.Nodes))
	}
}

func TestConcurrentCallGraph_ConcurrentCreateNode(t *testing.T) {
	root := &ssa.Function{}
	g := NewConcurrentCallGraph(root)

	fns := make([]*ssa.Function, 100)
	for i := range fns {
		fns[i] = &ssa.Function{}
	}

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, fn := range fns {
				g.CreateNode(fn)
			}
		}()
	}
	wg.Wait()

	graph := g.GetGraph()
	// root + 100 functions
	if len(graph.Nodes) != 101 {
		t.Fatalf("expected 101 nodes, got %d", len(graph.Nodes))
	}
}

func TestConcurrentCallGraph_ConcurrentAddEdge(t *testing.T) {
	root := &ssa.Function{}
	g := NewConcurrentCallGraph(root)

	fns := make([]*ssa.Function, 10)
	for i := range fns {
		fns[i] = &ssa.Function{}
		g.CreateNode(fns[i])
	}

	rootNode := g.CreateNode(root)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, fn := range fns {
				callee := g.CreateNode(fn)
				g.AddCallGraphEdge(rootNode, nil, callee)
			}
		}()
	}
	wg.Wait()

	// Each of 4 goroutines adds 10 edges
	if len(rootNode.Out) != 40 {
		t.Fatalf("expected 40 outgoing edges, got %d", len(rootNode.Out))
	}
}
