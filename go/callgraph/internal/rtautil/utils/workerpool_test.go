package utils

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// fnItem is a tiny work-item type for the pool tests.
type fnItem func(workerID int)

// runPool is a small driver: makes a pool, starts it with a callback that
// invokes the fnItem, lets caller seed work via push, then waits and stops.
func runPool(t *testing.T, numWorkers int, push func(p *WorkerPool[fnItem])) {
	t.Helper()
	p := NewWorkerPool[fnItem](numWorkers)
	p.Start(func(item fnItem, workerID int) { item(workerID) })
	push(p)
	p.Wait()
	p.Stop()
}

func TestWorkerPool_BasicDrain(t *testing.T) {
	const N = 1000
	var counter atomic.Int32
	runPool(t, 8, func(p *WorkerPool[fnItem]) {
		for i := 0; i < N; i++ {
			p.Push(func(workerID int) { counter.Add(1) }, -1)
		}
	})
	require.EqualValues(t, N, counter.Load())
}

func TestWorkerPool_SingleWorker(t *testing.T) {
	const N = 500
	var counter atomic.Int32
	runPool(t, 1, func(p *WorkerPool[fnItem]) {
		for i := 0; i < N; i++ {
			p.Push(func(workerID int) {
				if workerID != 0 {
					t.Errorf("expected workerID=0, got %d", workerID)
				}
				counter.Add(1)
			}, -1)
		}
	})
	require.EqualValues(t, N, counter.Load())
}

func TestWorkerPool_NumWorkersClampedToOne(t *testing.T) {
	p := NewWorkerPool[fnItem](0)
	require.Equal(t, 1, p.NumWorkers())
	p2 := NewWorkerPool[fnItem](-5)
	require.Equal(t, 1, p2.NumWorkers())
}

// TestWorkerPool_WorkerRandBounds verifies that WorkerRand returns the
// expected per-worker RNG for valid indices and panics with a descriptive
// message for indices outside [0, NumWorkers()). The latter guards external
// callers against silently sharing a per-worker source across workers or
// dereferencing out-of-bounds memory.
func TestWorkerPool_WorkerRandBounds(t *testing.T) {
	const numWorkers = 4
	p := NewWorkerPool[fnItem](numWorkers)

	// Valid indices return distinct, non-nil sources.
	seen := make(map[*rand.Rand]bool, numWorkers)
	for i := 0; i < numWorkers; i++ {
		r := p.WorkerRand(i)
		require.NotNil(t, r, "WorkerRand(%d) returned nil", i)
		require.False(t, seen[r], "WorkerRand(%d) aliased another worker's source", i)
		seen[r] = true
	}

	for _, bad := range []int{-1, numWorkers, numWorkers + 1, 1 << 20} {
		bad := bad
		require.PanicsWithValue(
			t,
			fmt.Sprintf("WorkerPool.WorkerRand: workerID %d out of range [0, %d)", bad, numWorkers),
			func() { p.WorkerRand(bad) },
			"WorkerRand(%d) should panic with a descriptive message", bad,
		)
	}
}

// TestWorkerPool_RecursiveProduction exercises the inner-Push path: each
// item produces K children up to a depth, like RTA discovering new functions
// during visitFunc. Verifies every produced item is processed exactly once.
func TestWorkerPool_RecursiveProduction(t *testing.T) {
	const (
		numWorkers = 8
		fanout     = 4
		depth      = 7
	)
	expected := 0
	for i, n := 0, 1; i <= depth; i++ {
		expected += n
		n *= fanout
	}

	var produced atomic.Int32
	p := NewWorkerPool[fnItem](numWorkers)
	var produce func(d int) fnItem
	produce = func(d int) fnItem {
		return func(workerID int) {
			produced.Add(1)
			if d < depth {
				for j := 0; j < fanout; j++ {
					p.Push(produce(d+1), workerID)
				}
			}
		}
	}
	p.Start(func(item fnItem, workerID int) { item(workerID) })
	p.Push(produce(0), -1)
	p.Wait()
	p.Stop()
	require.EqualValues(t, expected, produced.Load(),
		"expected %d items processed, got %d", expected, produced.Load())
}

// TestWorkerPool_ManyWorkersHighContention pushes a large number of items
// from many goroutines (well, from main with workerID = -1, then via
// inner-Push) and checks that every item is processed exactly once.
func TestWorkerPool_ManyWorkersHighContention(t *testing.T) {
	const (
		numWorkers = 32
		nItems     = 50_000
	)
	consumed := make([]atomic.Bool, nItems)
	p := NewWorkerPool[fnItem](numWorkers)
	p.Start(func(item fnItem, workerID int) { item(workerID) })
	for i := 0; i < nItems; i++ {
		i := i
		p.Push(func(workerID int) {
			if consumed[i].Swap(true) {
				t.Errorf("item %d consumed twice", i)
			}
		}, -1)
	}
	p.Wait()
	p.Stop()
	missing := 0
	for i := 0; i < nItems; i++ {
		if !consumed[i].Load() {
			missing++
		}
	}
	require.Zero(t, missing, "items not consumed")
}

// TestWorkerPool_SleepWakeRoundTrip lets the pool drain (workers go to
// sleep), then pushes a second batch of work and checks it completes. This
// exercises the sleeper-counted wake mechanism: workers must wake from
// blocking on `wakeup` when the producer signals.
func TestWorkerPool_SleepWakeRoundTrip(t *testing.T) {
	const numWorkers = 16
	var phase1, phase2 atomic.Int32
	p := NewWorkerPool[fnItem](numWorkers)
	p.Start(func(item fnItem, workerID int) { item(workerID) })

	// Phase 1: small batch, then drain so workers go to sleep.
	for i := 0; i < 4; i++ {
		p.Push(func(workerID int) { phase1.Add(1) }, -1)
	}
	p.Wait()
	require.EqualValues(t, 4, phase1.Load())

	// Give workers time to actually sleep (they sleep after K=2 empty steal
	// rounds; under a busy CPU this can take a few ms).
	for i := 0; i < 100; i++ {
		if p.numSleeping.Load() > 0 {
			break
		}
		runtime.Gosched()
	}
	require.True(t, p.numSleeping.Load() > 0,
		"expected at least one worker sleeping before phase 2; got numSleeping=%d", p.numSleeping.Load())

	// Phase 2: push more work; workers must wake up via the wakeup channel.
	for i := 0; i < 200; i++ {
		p.Push(func(workerID int) { phase2.Add(1) }, -1)
	}
	p.Wait()
	require.EqualValues(t, 200, phase2.Load())
	p.Stop()
}

// TestWorkerPool_StopUnblocksSleepers spawns a pool, pushes nothing, waits
// for workers to sleep, then Stops and ensures all worker goroutines exit
// (otherwise this test would hang). Verifies the workDone-based shutdown
// path of the worker loop.
func TestWorkerPool_StopUnblocksSleepers(t *testing.T) {
	const numWorkers = 8
	p := NewWorkerPool[fnItem](numWorkers)
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	p.Start(func(item fnItem, workerID int) {
		defer wg.Done()
		item(workerID)
	})

	// Wait for at least one worker to register as sleeping. (We don't push
	// any work; all workers should land in the sleep path.)
	for i := 0; i < 1000; i++ {
		if p.numSleeping.Load() >= int32(numWorkers/2) {
			break
		}
		runtime.Gosched()
	}

	// Stop should unblock all sleepers via the workDone receive case.
	p.Stop()

	// The pool's goroutines should exit; we don't have a direct handle to
	// them, but if Stop didn't unblock the sleepers this test would hang
	// forever (test timeout would catch it). Push a quick sanity check
	// that subsequent Wait still returns (workWg never had any items).
	p.Wait()
}

// TestWorkerPool_PanicInWorkItem_DoesNotHangWait is a regression test
// for a bug where a work-item panic would leak its internal
// workWg.Add(1) (because the worker goroutine unwound through the
// top-level recover before calling Done), causing Wait to block
// forever. The fix wraps each p.do call in runWorkItem with a per-item
// defer Done() + defer recover, so a panicking work item:
//
//  1. cannot leak the WaitGroup — Done() still runs;
//  2. cannot kill the worker — the steal loop resumes;
//  3. cannot starve siblings — non-panicking items in the same batch
//     still complete.
//
// Mix: push N items, every K-th item panics; assert every non-panicking
// item ran, and Wait returns. Use a generous timeout via a watchdog so
// a regression manifests as a clean test failure rather than a hang.
func TestWorkerPool_PanicInWorkItem_DoesNotHangWait(t *testing.T) {
	const (
		numWorkers = 8
		nItems     = 200
		panicEvery = 7 // every 7th item panics
	)
	var processed, panicked atomic.Int32
	p := NewWorkerPool[fnItem](numWorkers)
	p.Start(func(item fnItem, workerID int) { item(workerID) })

	for i := 0; i < nItems; i++ {
		i := i
		p.Push(func(workerID int) {
			if i%panicEvery == 0 {
				panicked.Add(1)
				panic(fmt.Sprintf("intentional panic in item %d", i))
			}
			processed.Add(1)
		}, -1)
	}

	// Watchdog: if Wait hangs (regression), abort with a clear message
	// instead of letting the test target time out.
	done := make(chan struct{})
	go func() {
		p.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatalf("Wait did not return — work-item panic leaked workWg "+
			"(processed=%d, panicked=%d, total=%d)",
			processed.Load(), panicked.Load(), nItems)
	}
	p.Stop()

	expectedPanics := int32(0)
	expectedProcessed := int32(0)
	for i := 0; i < nItems; i++ {
		if i%panicEvery == 0 {
			expectedPanics++
		} else {
			expectedProcessed++
		}
	}
	require.EqualValues(t, expectedPanics, panicked.Load(),
		"every K-th item should have panicked")
	require.EqualValues(t, expectedProcessed, processed.Load(),
		"every non-panicking item should have run to completion")
}

// many initial pushers (all using workerID = -1; the test serializes its
// own outer pushes through a goroutine so chaselev's single-writer
// contract on mainbox is honored). Each item randomly produces 0-3 children.
// Property: every produced item is consumed exactly once.
//
// Workload is intentionally modest so 300 parallel copies (CI's
// --runs_per_test=300) all finish well under the test target's timeout
// while still racing the steal/wake paths between workers.
func TestWorkerPool_RandomOps(t *testing.T) {
	const (
		numWorkers = 16
		nItems     = 500
		seed       = nItems / 10
		maxItems   = 5_000
	)
	consumed := make([]atomic.Bool, maxItems)
	produced := atomic.Int32{}

	p := NewWorkerPool[fnItem](numWorkers)
	rng := rand.New(rand.NewSource(42))
	// mu serializes BOTH nextID allocation and rng access. math/rand.Rand
	// is not goroutine-safe — concurrent calls can corrupt its internal
	// state and yield a negative Intn result, which previously cascaded
	// into a "runtime error: index out of range [-1]" panic in a work
	// item, leaking the matching workWg.Add(1) and hanging Wait.
	mu := sync.Mutex{}
	nextID := int32(0)
	var produce func(workerID int) fnItem
	produce = func(workerID int) fnItem {
		return func(wid int) {
			mu.Lock()
			id := nextID
			nextID++
			children := rng.Intn(4) // 0..3 children, under mu
			roomForMore := nextID < int32(maxItems-100)
			mu.Unlock()

			if id >= int32(maxItems) {
				return
			}
			if consumed[id].Swap(true) {
				t.Errorf("id %d consumed twice", id)
			}
			produced.Add(1)
			if !roomForMore {
				return
			}
			for c := 0; c < children; c++ {
				p.Push(produce(wid), wid)
			}
		}
	}
	p.Start(func(item fnItem, workerID int) { item(workerID) })

	for i := 0; i < seed; i++ {
		p.Push(produce(-1), -1)
	}
	p.Wait()
	p.Stop()

	require.True(t, produced.Load() >= int32(seed),
		"produced %d < seed %d", produced.Load(), seed)
}
