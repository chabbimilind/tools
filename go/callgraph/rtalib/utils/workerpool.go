package utils

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	"golang.org/x/tools/go/callgraph/rtalib/chaselev"
)

// WorkerPool is a fan-out worker pool with per-worker Chase-Lev work-stealing
// deques and a sleeper-counted wake mechanism. Each worker owns its own deque
// and Pushes new work to its own bottom (single-writer fast path); other
// workers Steal from the top via a single CAS. Items dispatched from outside
// the worker pool — initial roots, downstream batches submitted by the main
// goroutine — go into a separate "mainbox" deque that workers steal from.
//
// Idle workers register on numSleeping and block on a shared wakeup channel.
// Producers send a non-blocking signal on wakeup only when numSleeping > 0;
// surplus signals beyond the channel's capacity drop because the cascade of
// woken thieves stealing and adding work will re-signal.
//
// The pool is generic over the work-item type T. Common shape: T is an
// interface with one method that takes the workerID, e.g.
//
//	type workItem interface { doWork(r *flavorState, workerID int) }
//
//	pool := utils.NewWorkerPool[workItem](numWorkers)
//	pool.Start(func(item workItem, workerID int) { item.doWork(r, workerID) })
//	pool.Push(initialRoot, -1)         // from outside the pool
//	pool.Push(itemFromWorker, workerID) // from inside DoWork
//	pool.Wait()
//	pool.Stop()
//
// Memory model. chaselev.Deque assumes a single Pusher per deque. The pool
// enforces this contract by routing workerID >= 0 to workQueues[workerID]
// (whose only Pusher is worker workerID itself) and workerID < 0 to the
// mainbox (whose only Pusher is the goroutine that called Push with -1,
// typically the main goroutine driving the pool). Callers must not Push
// from multiple goroutines with the same workerID, and must not concurrently
// Push with workerID < 0 from more than one goroutine.
type WorkerPool[T any] struct {
	workQueues []*chaselev.Deque[T]
	mainbox    *chaselev.Deque[T]
	workDone   chan struct{}
	workWg     sync.WaitGroup

	numSleeping atomic.Int32
	wakeup      chan struct{}

	numWorkers int
	rngs       []PaddedRand

	do func(item T, workerID int)
}

// NewWorkerPool returns a pool sized for numWorkers. numWorkers <= 0 is
// clamped to 1. The pool is created in a stopped state; call Start to spawn
// the worker goroutines.
func NewWorkerPool[T any](numWorkers int) *WorkerPool[T] {
	if numWorkers <= 0 {
		numWorkers = 1
	}
	p := &WorkerPool[T]{
		workQueues: make([]*chaselev.Deque[T], numWorkers),
		mainbox:    chaselev.New[T](),
		workDone:   make(chan struct{}),
		wakeup:     make(chan struct{}, numWorkers),
		numWorkers: numWorkers,
		rngs:       make([]PaddedRand, numWorkers),
	}
	for i := range p.rngs {
		p.rngs[i].Rng = rand.New(rand.NewSource(int64(i)))
		p.workQueues[i] = chaselev.New[T]()
	}
	return p
}

// NumWorkers returns the configured worker count.
func (p *WorkerPool[T]) NumWorkers() int { return p.numWorkers }

// WorkerRand returns worker i's RNG. Useful when the caller needs a
// per-worker random source for unrelated purposes (e.g., a randomized
// algorithmic strategy) and wants to avoid allocating its own slice.
//
// Panics with a descriptive message if i is outside [0, NumWorkers()): the
// per-worker RNG is intended to be accessed only with a valid workerID, and
// silently returning a different worker's RNG (or nil) would violate the
// no-false-sharing contract of PaddedRand.
func (p *WorkerPool[T]) WorkerRand(i int) *rand.Rand {
	if i < 0 || i >= p.numWorkers {
		panic(fmt.Sprintf("WorkerPool.WorkerRand: workerID %d out of range [0, %d)", i, p.numWorkers))
	}
	return p.rngs[i].Rng
}

// Push enqueues an item.
//   - workerID >= 0 routes to workQueues[workerID]; only the goroutine
//     running worker workerID may call Push with this argument.
//   - workerID < 0 routes to the mainbox; intended for the main goroutine
//     dispatching from outside the pool (initial roots and downstream batches).
//
// Each Push increments an internal WaitGroup that is decremented after the
// item's DoWork callback returns. Wait blocks until the counter reaches zero.
func (p *WorkerPool[T]) Push(item T, workerID int) {
	p.workWg.Add(1)
	if workerID >= 0 {
		p.workQueues[workerID].Push(item)
	} else {
		p.mainbox.Push(item)
	}
	// Wake one sleeper if any. Sequential consistency between this atomic
	// Load and the thief's atomic Add (in the sleep path) closes the race:
	// if a thief Adds after our Load returns 0, its post-Add re-check of all
	// queues will see this Push (also a sequentially consistent atomic on
	// the deque's bottom), so it won't sleep. If our Load returns >0 a
	// thief is already (or about to be) blocked; the non-blocking send
	// delivers a wake — surplus signals beyond capacity drop because
	// cascading wakes from woken thieves will re-signal.
	if p.numSleeping.Load() > 0 {
		select {
		case p.wakeup <- struct{}{}:
		default:
		}
	}
}

// Start spawns numWorkers goroutines and stores do as the per-item callback.
// Each worker runs Pop on its own deque, Steals from peers and the mainbox,
// and blocks on wakeup when idle. Workers exit when Stop closes workDone.
//
// Start may only be called once on a pool.
func (p *WorkerPool[T]) Start(do func(item T, workerID int)) {
	p.do = do
	for i := range p.numWorkers {
		go p.workerLoop(i)
	}
}

// Wait blocks until every Push has been balanced by a completed DoWork.
func (p *WorkerPool[T]) Wait() { p.workWg.Wait() }

// Stop signals all workers to exit and unblocks any sleeping workers. Wait
// should be called before Stop to ensure all in-flight work has completed.
func (p *WorkerPool[T]) Stop() { close(p.workDone) }

func (p *WorkerPool[T]) workerLoop(i int) {
	defer func() {
		if r := recover(); r != nil {
			// Final safety net for panics inside the loop itself (e.g.,
			// chaselev Pop/Steal). Per-item panics from p.do are caught in
			// runWorkItem so they never reach here.
			fmt.Fprintf(os.Stderr, "WorkerPool: worker %d loop panicked: %v\n", i, r)
		}
	}()
	rng := p.rngs[i].Rng
	ownQueue := p.workQueues[i]
	// K = 2 light steal rounds before sleeping. Each round scans up to
	// numWorkers-1 victims plus the mainbox. Two rounds covers cases where
	// a producer is mid-Push during the first scan.
	const stealRoundsBeforeSleep = 2
	emptyRounds := 0
	for {
		select {
		case _, ok := <-p.workDone:
			if !ok {
				return
			}
		default:
		}

		if work, ok := ownQueue.Pop(); ok {
			p.runWorkItem(work, i)
			emptyRounds = 0
			continue
		}

		stolen := false
		if p.numWorkers > 1 {
			for attempt := 0; attempt < p.numWorkers-1; attempt++ {
				victim := rng.Intn(p.numWorkers)
				if victim == i {
					continue
				}
				if work, ok := p.workQueues[victim].Steal(); ok {
					p.runWorkItem(work, i)
					stolen = true
					emptyRounds = 0
					break
				}
			}
		}
		if stolen {
			continue
		}

		if work, ok := p.mainbox.Steal(); ok {
			p.runWorkItem(work, i)
			emptyRounds = 0
			continue
		}

		emptyRounds++
		if emptyRounds < stealRoundsBeforeSleep {
			runtime.Gosched()
			continue
		}

		// Register intent to sleep BEFORE re-checking queues. The atomic
		// Add provides a sequentially-consistent barrier: any producer
		// whose Push happens-before our subsequent re-check will be
		// visible; any producer whose Push happens-after our Add will see
		// numSleeping > 0 and signal.
		p.numSleeping.Add(1)
		anyWork := !p.mainbox.IsEmpty()
		if !anyWork {
			for q := 0; q < p.numWorkers; q++ {
				if !p.workQueues[q].IsEmpty() {
					anyWork = true
					break
				}
			}
		}
		if anyWork {
			p.numSleeping.Add(-1)
			emptyRounds = 0
			continue
		}

		select {
		case <-p.wakeup:
		case _, ok := <-p.workDone:
			p.numSleeping.Add(-1)
			if !ok {
				return
			}
			emptyRounds = 0
			continue
		}
		p.numSleeping.Add(-1)
		emptyRounds = 0
	}
}

// runWorkItem invokes the per-item callback for one work item. It
// guarantees workWg.Done() runs even if p.do panics, so a buggy work
// item cannot leak the internal WaitGroup (which would hang Wait
// forever) and cannot kill the worker goroutine — the panic is logged
// and the steal loop resumes on the next iteration.
//
// Defers fire LIFO, so the recover deferred SECOND runs FIRST; it
// catches any panic from p.do, then the workWg.Done deferred FIRST runs,
// balancing the Add(1) recorded when this item was Push'd.
func (p *WorkerPool[T]) runWorkItem(item T, workerID int) {
	defer p.workWg.Done()
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr,
				"WorkerPool: worker %d: work item panicked: %v\n", workerID, r)
		}
	}()
	p.do(item, workerID)
}
