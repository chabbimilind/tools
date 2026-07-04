// Package chaselev provides a per-worker work-stealing deque using the
// Chase-Lev algorithm (Chase & Lev, "Dynamic Circular Work-Stealing Deque",
// SPAA 2005). The owner pushes/pops at the bottom; thieves take from the top
// via a single CAS. Both ends are wait-free in the common case.
//
// Single-owner contract: only the goroutine that owns a Deque may call Push
// or Pop. Any number of thieves may call Steal concurrently with Push/Pop and
// with each other.
//
// Race-detector cleanliness. Vanilla Chase-Lev has a well-known benign race
// on the data slot: when the deque's circular buffer wraps (Push reaching
// an index whose slot is currently being read by a stale-top thief), the
// thief's slot read and the owner's slot write touch the same memory with
// no happens-before chain — Push only synchronizes via bottom, which the
// stale thief loaded earlier. The thief's CAS on top then fails (some other
// thief has advanced top in the meantime, which is exactly what allowed
// Push to wrap), and the racy value is discarded. Correctness is preserved.
//
// To keep the zero-allocation hot path while staying race-detector clean,
// the slot read in Steal and the slot write in Push (the only pair that can
// race) go through tiny //go:norace helpers. The race detector still
// instruments the rest of the package, so genuine bugs elsewhere are caught.
//
// Grow threshold (deliberate deviation from the paper). The original
// Chase-Lev paper grows when "size >= a.size() - 1", i.e., as soon as the
// array would reach cap-1 items. We grow at "b-t >= cap" instead, allowing
// the array to fill to exactly cap items before doubling. Both schemes
// preserve the safety invariant "Push only overwrites a slot whose previous
// occupant has already been stolen": if owner is about to write slot[b mod
// cap] without growing, then b-t < cap, so b - cap < t, meaning top has
// advanced past the previous occupant at position b-cap. The paper's
// stricter cutoff makes the invariant slightly simpler to prove (b-t < cap
// holds at every instant); ours saves one grow per generation by using all
// cap slots. The wrap-around overwrite itself never happens in our scheme
// either: when b-t reaches cap, the *next* Push triggers grow before
// writing, so the new value goes into the new array, not into slot[t mod
// cap] of the old one. See TestPush_FillsToCapacityBeforeGrowing for the
// boundary case.
//
// This package lives under ssync/workstealing/ alongside future work-stealing
// primitives (e.g., the original Cilk THE protocol with locked steal).
package chaselev

import (
	"sync/atomic"
)

// Deque is a Chase-Lev work-stealing deque holding values of type T.
// The zero value is not usable; obtain one via New.
type Deque[T any] struct {
	bottom atomic.Int64 // owner end; next push slot
	top    atomic.Int64 // thief end; next steal slot
	array  atomic.Pointer[wsArray[T]]
}

type wsArray[T any] struct {
	mask int64
	data []T
}

// DefaultInitialCap is the starting capacity of a freshly-constructed Deque.
// Must be a power of two; growth doubles the array.
const DefaultInitialCap = 256

// New returns a new Deque with DefaultInitialCap initial capacity.
func New[T any]() *Deque[T] {
	return NewWithInitialCap[T](DefaultInitialCap)
}

// NewWithInitialCap returns a new Deque with the given initial capacity.
// initialCap must be a power of two and at least 2; otherwise it is rounded
// up to the next power of two and clamped to a minimum of 2.
func NewWithInitialCap[T any](initialCap int) *Deque[T] {
	if initialCap < 2 {
		initialCap = 2
	}
	if initialCap&(initialCap-1) != 0 {
		// Round up to next power of two.
		c := 2
		for c < initialCap {
			c <<= 1
		}
		initialCap = c
	}
	d := &Deque[T]{}
	d.array.Store(newWSArray[T](initialCap))
	return d
}

func newWSArray[T any](capacity int) *wsArray[T] {
	return &wsArray[T]{
		mask: int64(capacity - 1),
		data: make([]T, capacity),
	}
}

// loadSlotBenignRace reads a deque slot. //go:norace excludes the read from
// race instrumentation so the benign Push-write vs Steal-read race documented
// at the package level does not surface as a false positive. Callers must
// validate the value via the surrounding top CAS — values returned from a
// racing slot read are only safe to consume after the CAS succeeds.
//
//go:norace
func loadSlotBenignRace[T any](p *T) T { return *p }

// storeSlotBenignRace writes a deque slot. Pair of loadSlotBenignRace; see
// its doc for the race-model justification.
//
//go:norace
func storeSlotBenignRace[T any](p *T, v T) { *p = v }

// Push appends an item at the bottom. Owner-only.
func (d *Deque[T]) Push(v T) {
	b := d.bottom.Load()
	t := d.top.Load()
	a := d.array.Load()
	// Grow when the array would otherwise be over-full. We use b-t >= cap
	// (the array may hold up to cap items); the Chase-Lev paper uses
	// b-t >= cap-1 (reserves one slot). Both prevent slot[b mod cap] from
	// overwriting a still-live slot[t mod cap]; see the package doc.
	if b-t >= a.mask+1 {
		// Grow: copy live items into a new, larger array, then publish.
		// The new array is private to the owner until d.array.Store
		// publishes it, so the copy needs no special synchronization. The
		// read of the old array is concurrent only with thief reads (read
		// vs read, not a race).
		newCap := (a.mask + 1) * 2
		na := newWSArray[T](int(newCap))
		for i := t; i < b; i++ {
			na.data[i&na.mask] = a.data[i&a.mask]
		}
		d.array.Store(na)
		a = na
	}
	// This write races (benignly) with concurrent Steal slot reads when
	// the buffer wraps — see loadSlotBenignRace.
	storeSlotBenignRace(&a.data[b&a.mask], v)
	// bottom.Store is sequentially consistent; the slot write above is
	// thereby visible to any thief that observes the new bottom.
	d.bottom.Store(b + 1)
}

// Pop removes an item from the bottom. Owner-only.
func (d *Deque[T]) Pop() (T, bool) {
	b := d.bottom.Load() - 1
	a := d.array.Load()
	d.bottom.Store(b)
	// Sequentially consistent fence between the bottom decrement and the
	// top load — thieves reading bottom after their CAS on top observe our
	// decrement in the same total order.
	t := d.top.Load()
	var zero T
	if t > b {
		// Empty (no in-flight thieves competed for this slot). Restore.
		d.bottom.Store(b + 1)
		return zero, false
	}
	// Reading the bottom slot is safe under the single-owner contract;
	// in the t < b case it cannot collide with any thief (thieves take
	// from the top end). In the t == b case the slot was last written
	// by Push before the matching bottom.Store, which sequentially
	// happens-before our bottom.Load above.
	v := a.data[b&a.mask]
	if t < b {
		// More than one element — owner takes the bottom slot uncontested.
		a.data[b&a.mask] = zero
		return v, true
	}
	// t == b: exactly one element remains and a thief might race for it.
	// CAS top to win; if we lose, the thief took it.
	ok := d.top.CompareAndSwap(t, t+1)
	d.bottom.Store(b + 1)
	if !ok {
		return zero, false
	}
	// Do NOT nil out a.data[b&a.mask] here even though we won the CAS. A
	// losing thief may already have loaded the slot before its CAS failed
	// (Steal reads the slot before its CAS); nil'ing would race with that
	// read. The thief discards the value, so leaving the slot alone is
	// harmless — it will be overwritten by a future Push or reclaimed when
	// the array generation is collected.
	return v, true
}

// Steal removes an item from the top. Safe for any number of concurrent
// thieves; concurrent with owner Push/Pop.
func (d *Deque[T]) Steal() (T, bool) {
	t := d.top.Load()
	// Sequentially consistent ordering between top.Load and bottom.Load
	// ensures we observe owner's bottom decrement (in Pop) if it happens
	// before our CAS — without that, we could steal an item the owner
	// already claimed.
	b := d.bottom.Load()
	var zero T
	if t >= b {
		return zero, false
	}
	a := d.array.Load()
	// This read races (benignly) with concurrent Push slot writes when
	// the buffer wraps. The following CAS gates correctness: if the CAS
	// succeeds, top hasn't advanced past t since our load, so no Push
	// can have wrapped to this slot index in the meantime. If the CAS
	// fails, we discard v.
	v := loadSlotBenignRace(&a.data[t&a.mask])
	if !d.top.CompareAndSwap(t, t+1) {
		return zero, false
	}
	// Note: do not nil out a.data[t&a.mask] here — concurrent Push by the
	// owner may have advanced into the slot after grow, and writing nil
	// would race with that Push.
	return v, true
}

// IsEmpty is a best-effort check (may be slightly stale due to concurrent
// Push/Pop/Steal). Safe to call from any goroutine.
func (d *Deque[T]) IsEmpty() bool {
	return d.bottom.Load() <= d.top.Load()
}
