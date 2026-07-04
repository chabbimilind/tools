package utils

import (
	"math/rand"
	"slices"
	"sync"
	"unsafe"

	"golang.org/x/tools/go/callgraph"
)

// dedupEdgesSmallThreshold is the edge count below which we use an O(n²) scan
// instead of a map. For small n the quadratic scan fits in cache and avoids
// map allocation overhead, making it faster in practice.
const dedupEdgesSmallThreshold = 25

// SeenMapPool is a sync.Pool for reusing map[callgraph.Edge]int in DedupEdgesLarge.
var SeenMapPool = sync.Pool{
	New: func() any {
		return make(map[callgraph.Edge]int)
	},
}

// PaddedCounter is a counter padded to a cache line to avoid false sharing.
type PaddedCounter struct {
	Count int
	_     [56]byte // pad to 64-byte cache line
}

// PaddedRand is a per-worker *rand.Rand padded to a cache line to avoid false sharing.
type PaddedRand struct {
	Rng *rand.Rand
	_   [56]byte // pad to 64-byte cache line
}

// WorkerCounter is a per-worker counter array padded to avoid false sharing.
type WorkerCounter []PaddedCounter

// NewWorkerCounter creates a WorkerCounter for n workers.
func NewWorkerCounter(n int) WorkerCounter {
	return make(WorkerCounter, n)
}

// Inc increments the counter for the given worker.
func (wc WorkerCounter) Inc(workerID int) {
	wc[workerID].Count++
}

// Add adds delta to the counter for the given worker.
func (wc WorkerCounter) Add(workerID int, delta int) {
	wc[workerID].Count += delta
}

// Sum returns the total across all workers.
func (wc WorkerCounter) Sum() int {
	total := 0
	for i := range wc {
		total += wc[i].Count
	}
	return total
}

// DedupEdges deduplicates call graph edges in-place.
func DedupEdges(edges []*callgraph.Edge) []*callgraph.Edge {
	if len(edges) < dedupEdgesSmallThreshold {
		return dedupEdgesSmall(edges)
	}
	return dedupEdgesLarge(edges)
}

// dedupEdgesLarge deduplicates edges in-place using a map for larger edge counts.
// Edge equality (callgraph.Edge struct comparison) works because SSA functions
// are interned, so pointer identity on Caller/Callee nodes is sufficient.
func dedupEdgesLarge(edges []*callgraph.Edge) []*callgraph.Edge {
	seen := SeenMapPool.Get().(map[callgraph.Edge]int)
	clear(seen)
	defer SeenMapPool.Put(seen)

	writeIdx := 0

	for _, edge := range edges {
		if edge == nil {
			continue
		}

		if idx, ok := seen[*edge]; !ok {
			seen[*edge] = writeIdx
			edges[writeIdx] = edge
			writeIdx++
		} else {
			existing := edges[idx]
			if uintptr(unsafe.Pointer(edge)) < uintptr(unsafe.Pointer(existing)) {
				edges[idx] = edge
			}
		}
	}

	return slices.Clip(edges[:writeIdx])
}

// dedupEdgesSmall deduplicates edges in-place using O(N^2) algorithm for small edge counts
// to avoid map allocation overhead.
// Edge equality (callgraph.Edge struct comparison) works because SSA functions
// are interned, so pointer identity on Caller/Callee nodes is sufficient.
func dedupEdgesSmall(edges []*callgraph.Edge) []*callgraph.Edge {
	writeIdx := 0

	for _, edge := range edges {
		if edge == nil {
			continue
		}

		found := false
		for i := 0; i < writeIdx; i++ {
			if *edge == *edges[i] {
				found = true
				if uintptr(unsafe.Pointer(edge)) < uintptr(unsafe.Pointer(edges[i])) {
					edges[i] = edge
				}
				break
			}
		}

		if !found {
			edges[writeIdx] = edge
			writeIdx++
		}
	}

	return slices.Clip(edges[:writeIdx])
}
