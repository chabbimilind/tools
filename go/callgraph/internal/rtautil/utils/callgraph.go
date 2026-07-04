package utils

import (
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/puzpuzpuz/xsync/v4"
	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
)

// nodeEntry holds a callgraph node plus its per-node lock used to
// serialize concurrent appends to node.In / node.Out.
//
// sync.Mutex is embedded by value (not pointer) so each node costs one
// allocation (the nodeEntry itself) rather than two — saves an alloc
// and a pointer indirection per node.
type nodeEntry struct {
	node *callgraph.Node
	lock sync.Mutex
}

// ConcurrentCallGraph represents a call graph safe for concurrent use.
//
// A graph may contain nodes that are not reachable from the root.
// If the call graph is sound, such nodes indicate unreachable
// functions.
type ConcurrentCallGraph struct {
	root *ssa.Function

	nodes  *xsync.Map[*ssa.Function, *nodeEntry] // map[*ssa.Function]*nodeEntry
	nextID int32                                 // atomic counter for node IDs
}

// New returns a new CallGraph with the specified (optional) root node.
func NewConcurrentCallGraph(root *ssa.Function) *ConcurrentCallGraph {
	g := &ConcurrentCallGraph{
		root:  root,
		nodes: xsync.NewMap[*ssa.Function, *nodeEntry](),
	}
	// Create the root node
	rootNode := &callgraph.Node{Func: root, ID: 0}
	entry := &nodeEntry{node: rootNode}
	g.nodes.Store(root, entry)
	g.nextID = 1
	return g
}

// CreateNode returns the Node for fn, creating it if not present.
// The root node may have fn=nil.
func (g *ConcurrentCallGraph) CreateNode(fn *ssa.Function) *callgraph.Node {
	node, _ := g.CreateNodeWithLock(fn)
	return node
}

// CreateNodeWithLock returns the Node for fn AND the per-node lock used
// to serialize appends to node.In / node.Out. Hot edge-construction
// paths (every addEdge call) should prefer this + AddCallGraphEdgeFast
// to skip the redundant xsync.Map.Load AddCallGraphEdge performs to
// rediscover the locks — empirically that lookup costs ~100ns per edge,
// which on services with tens of millions of edges is several seconds
// of CPU.
//
// The returned lock pointer is stable for the lifetime of the graph;
// callers may safely cache it across edges incident to the same node.
func (g *ConcurrentCallGraph) CreateNodeWithLock(fn *ssa.Function) (*callgraph.Node, *sync.Mutex) {
	// Try to load existing node
	if entry, ok := g.nodes.Load(fn); ok {
		return entry.node, &entry.lock
	}

	// Generate new ID atomically
	id := int(atomic.AddInt32(&g.nextID, 1) - 1)

	// Create new node entry with both node and lock
	newNode := &callgraph.Node{Func: fn, ID: id}
	entry := &nodeEntry{node: newNode}

	actual, _ := g.nodes.LoadOrStore(fn, entry)
	return actual.node, &actual.lock
}

// AddCallGraphEdge attaches a (caller, site, callee) edge to the call
// graph. Elimination of duplicate edges is the caller's responsibility.
// This convenience signature performs an xsync.Map.Load per node to
// recover the locks — on hot paths prefer AddCallGraphEdgeFast with
// locks obtained from CreateNodeWithLock.
func (g *ConcurrentCallGraph) AddCallGraphEdge(caller *callgraph.Node, site ssa.CallInstruction, callee *callgraph.Node) {
	calleeEntry, ok := g.nodes.Load(callee.Func)
	if !ok {
		panic(fmt.Sprintf("Could not find callee: %v", callee))
	}
	callerEntry, ok := g.nodes.Load(caller.Func)
	if !ok {
		panic(fmt.Sprintf("Could not find caller: %v", caller))
	}
	g.AddCallGraphEdgeFast(caller, &callerEntry.lock, site, callee, &calleeEntry.lock)
}

// AddCallGraphEdgeFast is the lock-pointer-taking variant of
// AddCallGraphEdge: callers obtain caller/callee locks once via
// CreateNodeWithLock and reuse them across all edges incident to those
// nodes. Eliminates the two xsync.Map.Load operations
// AddCallGraphEdge performs internally to rediscover the locks.
func (g *ConcurrentCallGraph) AddCallGraphEdgeFast(caller *callgraph.Node, callerLock *sync.Mutex, site ssa.CallInstruction, callee *callgraph.Node, calleeLock *sync.Mutex) {
	e := &callgraph.Edge{
		Caller: caller,
		Site:   site,
		Callee: callee,
	}
	calleeLock.Lock()
	callee.In = append(callee.In, e)
	calleeLock.Unlock()

	callerLock.Lock()
	caller.Out = append(caller.Out, e)
	callerLock.Unlock()
}

// GetGraph materializes and returns a callgraph.Graph by iterating over
// the xsync.Map and reconstructing the graph.
// GetGraph must be called only after all AddCallGraphEdge calls have completed.
func (g *ConcurrentCallGraph) GetGraph() *callgraph.Graph {
	// Create the graph with the root
	var rootNode *callgraph.Node
	if entry, ok := g.nodes.Load(g.root); ok {
		rootNode = entry.node
	}

	graph := &callgraph.Graph{
		Root:  rootNode,
		Nodes: make(map[*ssa.Function]*callgraph.Node),
	}

	// Iterate over all nodes in the xsync.Map and add them to the graph
	g.nodes.Range(func(fn *ssa.Function, entry *nodeEntry) bool {
		graph.Nodes[fn] = entry.node
		return true
	})

	return graph
}
