package chaselev

import (
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPushPop_LIFO(t *testing.T) {
	d := New[int]()
	for i := 0; i < 100; i++ {
		d.Push(i)
	}
	for i := 99; i >= 0; i-- {
		v, ok := d.Pop()
		require.True(t, ok, "Pop should succeed for i=%d", i)
		require.Equal(t, i, v)
	}
	_, ok := d.Pop()
	require.False(t, ok, "Pop on empty deque should fail")
}

func TestPopOnEmpty(t *testing.T) {
	d := New[int]()
	_, ok := d.Pop()
	require.False(t, ok)
	require.True(t, d.IsEmpty())
}

func TestStealOnEmpty(t *testing.T) {
	d := New[int]()
	_, ok := d.Steal()
	require.False(t, ok)
}

func TestSteal_FIFO(t *testing.T) {
	d := New[int]()
	for i := 0; i < 100; i++ {
		d.Push(i)
	}
	for i := 0; i < 100; i++ {
		v, ok := d.Steal()
		require.True(t, ok, "Steal should succeed for i=%d", i)
		require.Equal(t, i, v)
	}
	_, ok := d.Steal()
	require.False(t, ok)
}

func TestPushPopGrowth(t *testing.T) {
	// Force several growth events.
	d := NewWithInitialCap[int](2)
	const N = 10_000
	for i := 0; i < N; i++ {
		d.Push(i)
	}
	for i := N - 1; i >= 0; i-- {
		v, ok := d.Pop()
		require.True(t, ok)
		require.Equal(t, i, v)
	}
}

func TestStealGrowth(t *testing.T) {
	d := NewWithInitialCap[int](2)
	const N = 10_000
	for i := 0; i < N; i++ {
		d.Push(i)
	}
	for i := 0; i < N; i++ {
		v, ok := d.Steal()
		require.True(t, ok)
		require.Equal(t, i, v)
	}
}

func TestNewWithInitialCap_RoundsUp(t *testing.T) {
	// 17 should round up to 32.
	d := NewWithInitialCap[int](17)
	a := d.array.Load()
	require.EqualValues(t, 31, a.mask, "cap-1 should be 31 (cap=32)")
}

// posTag returns a distinctive non-zero tag for logical position p. We
// avoid the natural numbers 0,1,2,… in boundary tests so that a "slot
// returns zero" bug or a "slot returns the wrong position's value" bug
// shows up as a tag mismatch rather than coincidentally matching the
// expected value.
func posTag(p int) int { return 0x0C0DE000 | p }

// TestPush_FillsToCapacityBeforeGrowing documents and verifies the
// deliberate deviation from the Chase-Lev paper's grow threshold. The
// paper grows when "size >= cap-1" (reserves one slot); we grow when
// "size >= cap" (uses all cap slots). Both prevent slot[t mod cap] from
// being overwritten while position t is still live, because the next Push
// always grows before the wrap-around write.
//
// Concretely: with cap=4, pushing 4 items must NOT grow (paper would have
// grown after the 3rd push). Pushing a 5th must trigger grow before write.
// Every item must round-trip to its original distinctive tag (so a silent
// slot-aliasing bug is visible as a tag mismatch).
func TestPush_FillsToCapacityBeforeGrowing(t *testing.T) {
	d := NewWithInitialCap[int](4)
	require.EqualValues(t, 3, d.array.Load().mask, "expected cap=4 (mask=3)")

	// Push exactly cap items. Our scheme does not grow.
	for i := 0; i < 4; i++ {
		d.Push(posTag(i))
	}
	require.EqualValues(t, 3, d.array.Load().mask,
		"array must NOT grow at b-t==cap-1 — paper grows here, we don't")

	// Push one more. b-t now equals cap, so this push must grow first.
	d.Push(posTag(4))
	require.Greater(t, d.array.Load().mask, int64(3),
		"array must grow when b-t reaches cap; otherwise slot[b mod cap] "+
			"would alias slot[t mod cap] and overwrite a live item")

	// LIFO pop must yield the distinctive tags 4,3,2,1,0 in reverse —
	// any slot corruption (zero, off-by-one index, stale generation)
	// surfaces as a tag mismatch.
	for i := 4; i >= 0; i-- {
		v, ok := d.Pop()
		require.True(t, ok, "pop %d failed", i)
		require.Equalf(t, posTag(i), v,
			"position %d: got %#x, want %#x — slot aliasing or grow lost data?",
			i, v, posTag(i))
	}
	_, ok := d.Pop()
	require.False(t, ok, "deque should be empty")
}

// TestPushWrapAround_AfterStealsFreeSlots verifies the safety invariant
// directly: when Steals have advanced top past some position p, Push may
// later write to slot[p mod cap] without growing — the slot is "free"
// because position p's value has already left the deque. This is the
// scenario that allows our >= cap grow threshold to be safe.
func TestPushWrapAround_AfterStealsFreeSlots(t *testing.T) {
	d := NewWithInitialCap[int](4)

	// Fill to cap items at positions 0..3 with distinctive tags.
	for i := 0; i < 4; i++ {
		d.Push(posTag(i))
	}

	// Steal 3 items in FIFO order; top advances 0 -> 3. Each stolen value
	// must be the tag originally placed at that position — otherwise the
	// wrap-around scheme has corrupted the head.
	for i := 0; i < 3; i++ {
		v, ok := d.Steal()
		require.True(t, ok)
		require.Equalf(t, posTag(i), v,
			"steal at position %d returned %#x, want %#x", i, v, posTag(i))
	}

	initialMask := d.array.Load().mask

	// Push 3 more (positions 4,5,6). In our scheme these three pushes
	// happen WITHOUT growing because each wraps to a slot whose previous
	// occupant was just stolen. The writes go into:
	//   slot[4 & 3]=0  (vacated by steal of position 0)
	//   slot[5 & 3]=1  (vacated by steal of position 1)
	//   slot[6 & 3]=2  (vacated by steal of position 2)
	for i := 4; i <= 6; i++ {
		d.Push(posTag(i))
	}
	require.EqualValues(t, initialMask, d.array.Load().mask,
		"wrap-around pushes should not grow when previous occupants were stolen")

	// Deque now holds positions 3,4,5,6. LIFO pop yields tags for
	// positions 6,5,4,3 — verifies no slot got crossed during wrap.
	for _, p := range []int{6, 5, 4, 3} {
		v, ok := d.Pop()
		require.True(t, ok)
		require.Equalf(t, posTag(p), v,
			"LIFO pop at position %d returned %#x, want %#x", p, v, posTag(p))
	}
	_, ok := d.Pop()
	require.False(t, ok)
}

// TestSteal_AtCapacityBoundary verifies Steal returns correct values when
// the array is at exactly b-t==cap — the "fully packed" state that our
// scheme allows but the paper's never reaches.
func TestSteal_AtCapacityBoundary(t *testing.T) {
	d := NewWithInitialCap[int](4)
	for i := 0; i < 4; i++ {
		d.Push(posTag(i))
	}
	require.EqualValues(t, 3, d.array.Load().mask,
		"should still be at initial cap=4 with all slots used")

	for i := 0; i < 4; i++ {
		v, ok := d.Steal()
		require.True(t, ok, "steal %d failed at boundary", i)
		require.Equalf(t, posTag(i), v,
			"steal at position %d returned %#x, want %#x", i, v, posTag(i))
	}
	_, ok := d.Steal()
	require.False(t, ok)
	require.True(t, d.IsEmpty())
}

// TestChoreographed_CircularBufferEdges walks the deque through a
// hand-tuned sequence designed to hit every subtle boundary in the
// circular buffer:
//   - Phase 1: fill to exactly cap items (paper grows here, we don't)
//   - Phase 2: wrap-without-grow cycle — alternate Steal+Push so each
//     push reuses a slot the matching steal just vacated
//   - Phase 3: push past cap, force a grow while wrapped positions
//     occupy non-contiguous slots — verify the grow correctly remaps
//     every live position into the new array
//   - Phase 4: drain LIFO, asserting tags for non-trivial positions
//     (numerically and via a model)
//   - Phase 5: re-use the deque post-drain to confirm no stale values
//     leak through from earlier generations
//
// Every Push uses a distinctive tag = 0x0C0DE000 | position; the test
// asserts each Pop/Steal returns the EXACT tag for the position it
// claims to be returning. This catches slot aliasing, torn writes,
// stale array reads, off-by-one indexing, and grow-time index errors —
// any one of which would silently return the wrong position's value in
// a less stringent test.
func TestChoreographed_CircularBufferEdges(t *testing.T) {
	const cap = 4
	d := NewWithInitialCap[int](cap)
	initialMask := d.array.Load().mask
	require.EqualValues(t, cap-1, initialMask, "initial mask should be cap-1")

	// Model: live[i] is the tag at deque position (top+i). live[0] is the
	// thief side (next Steal), live[len-1] is the owner side (next Pop).
	var live []int
	nextPos := 0 // monotonically assigns positions/tags to pushes

	push := func() {
		d.Push(posTag(nextPos))
		live = append(live, posTag(nextPos))
		nextPos++
	}
	expectSteal := func(label string) {
		require.NotEmptyf(t, live, "%s: model says empty but tried steal", label)
		want := live[0]
		v, ok := d.Steal()
		require.Truef(t, ok, "%s: steal returned !ok", label)
		require.Equalf(t, want, v,
			"%s: stolen tag %#x; expected %#x — slot corruption?", label, v, want)
		live = live[1:]
	}
	expectPop := func(label string) {
		require.NotEmptyf(t, live, "%s: model says empty but tried pop", label)
		want := live[len(live)-1]
		v, ok := d.Pop()
		require.Truef(t, ok, "%s: pop returned !ok", label)
		require.Equalf(t, want, v,
			"%s: popped tag %#x; expected %#x — slot corruption?", label, v, want)
		live = live[:len(live)-1]
	}
	assertNoGrow := func(label string) {
		require.EqualValuesf(t, initialMask, d.array.Load().mask,
			"%s: array grew unexpectedly (mask now %d, was %d)",
			label, d.array.Load().mask, initialMask)
	}

	// --- Phase 1: fill to exactly cap items. Paper would have grown
	//     after the (cap-1)-th push; our scheme keeps the same array.
	for i := 0; i < cap; i++ {
		push()
	}
	assertNoGrow("phase1: filled to cap items")
	require.Len(t, live, cap)

	// --- Phase 2: cap-1 wrap-without-grow cycles. Each iteration steals
	//     the FIFO front and pushes a new tag onto the back; bottom
	//     advances faster than top, so b-t stays at cap throughout, but
	//     never EXCEEDS cap (because we steal first, then push). The
	//     pushes wrap into slots just vacated by the steals.
	//     Slot occupancy walk (cap=4):
	//       initial:        slot 0=pos0, 1=pos1, 2=pos2, 3=pos3
	//       steal pos 0:    slot 0=stale, others unchanged
	//       push pos 4:     slot 4&3=0 → slot 0=pos4 (wrap into freed)
	//       steal pos 1:    slot 1 freed
	//       push pos 5:     slot 5&3=1 → slot 1=pos5
	//       steal pos 2:    slot 2 freed
	//       push pos 6:     slot 6&3=2 → slot 2=pos6
	//     Final occupancy: slot 0=pos4, 1=pos5, 2=pos6, 3=pos3.
	for i := 0; i < cap-1; i++ {
		expectSteal("phase2: steal head")
		push()
		assertNoGrow("phase2: push wraps into freed slot, no grow")
	}
	require.Len(t, live, cap, "phase2 maintains exactly cap items")

	// --- Phase 3: one more push — b-t is already cap, so this push MUST
	//     grow before writing. The grow has to correctly copy four live
	//     items that sit at NON-CONTIGUOUS slot indices in the old array
	//     (positions 3,4,5,6 at slots 3,0,1,2 respectively). A naive copy
	//     loop that assumes contiguous t..b-1 slot indices would corrupt
	//     the new array; our `for i := t; i < b; i++ { na[i&new] = a[i&old] }`
	//     handles it correctly because both index maps use mod.
	push()
	require.Greater(t, d.array.Load().mask, initialMask,
		"phase3: push at b-t==cap must trigger grow")
	require.Equal(t, int64(cap*2-1), d.array.Load().mask,
		"phase3: grow should exactly double cap")
	require.Len(t, live, cap+1, "phase3 added one item across grow")

	// --- Phase 4: drain everything LIFO. The deque holds (from oldest
	//     to newest) positions 3, 4, 5, 6, 7. Pop yields 7,6,5,4,3 —
	//     any wrap-around miscopy in Phase 3 would surface here.
	for len(live) > 0 {
		expectPop("phase4: LIFO drain")
	}
	_, ok := d.Pop()
	require.False(t, ok, "phase4: deque should be empty after draining all items")
	_, ok = d.Steal()
	require.False(t, ok, "phase4: steal on empty deque should fail")
	require.True(t, d.IsEmpty(), "phase4: IsEmpty should agree")

	// --- Phase 5: re-use the deque post-drain. Push a fresh batch and
	//     drain via Steal. Any stale value lingering in a slot from
	//     earlier phases would surface as a tag mismatch (the new tags
	//     use larger position numbers, distinct from any earlier tag).
	for i := 0; i < cap*3; i++ {
		push()
	}
	for len(live) > 0 {
		expectSteal("phase5: drain via steal after re-use")
	}
}

func TestNewWithInitialCap_Min(t *testing.T) {
	d := NewWithInitialCap[int](1)
	a := d.array.Load()
	require.EqualValues(t, 1, a.mask, "cap-1 should be 1 (cap=2)")
}

// TestConcurrent_OwnerVsManyThieves stresses the canonical Chase-Lev
// guarantees: every pushed item is consumed exactly once by either Pop or
// Steal, and the deque is empty at the end.
func TestConcurrent_OwnerVsManyThieves(t *testing.T) {
	const (
		nThieves = 8
		nItems   = 100_000
	)
	d := New[int]()
	consumed := make([]atomic.Bool, nItems)

	var wg sync.WaitGroup

	// Owner: pushes nItems, intermixed with Pops.
	wg.Add(1)
	ownerDone := make(chan struct{})
	go func() {
		defer wg.Done()
		for i := 0; i < nItems; i++ {
			d.Push(i)
			// Occasionally Pop, simulating LIFO discipline.
			if i%4 == 0 {
				if v, ok := d.Pop(); ok {
					if v < 0 || v >= nItems {
						t.Errorf("Pop returned out-of-range %d", v)
						return
					}
					if consumed[v].Swap(true) {
						t.Errorf("Pop returned duplicate %d", v)
						return
					}
				}
			}
		}
		// Drain remaining via Pop.
		for {
			v, ok := d.Pop()
			if !ok {
				break
			}
			if v < 0 || v >= nItems {
				t.Errorf("Pop returned out-of-range %d", v)
				return
			}
			if consumed[v].Swap(true) {
				t.Errorf("Pop returned duplicate %d", v)
				return
			}
		}
		close(ownerDone)
	}()

	// Thieves: keep stealing until owner is done AND deque is empty.
	for i := 0; i < nThieves; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				v, ok := d.Steal()
				if ok {
					if v < 0 || v >= nItems {
						t.Errorf("Steal returned out-of-range %d", v)
						return
					}
					if consumed[v].Swap(true) {
						t.Errorf("Steal returned duplicate %d", v)
						return
					}
					continue
				}
				select {
				case <-ownerDone:
					if d.IsEmpty() {
						return
					}
				default:
				}
				runtime.Gosched()
			}
		}()
	}

	wg.Wait()
	for i := 0; i < nItems; i++ {
		if !consumed[i].Load() {
			t.Fatalf("item %d was never consumed", i)
		}
	}
	require.True(t, d.IsEmpty())
}

// TestConcurrent_RandomOps stresses the deque with random Push/Pop on the
// owner and random Steal on thieves. Outcome property: every Pushed value is
// consumed exactly once, and final deque is empty.
func TestConcurrent_RandomOps(t *testing.T) {
	const (
		nThieves = 16
		nItems   = 500_000
	)
	d := NewWithInitialCap[int](2) // small to force growth
	consumed := make([]atomic.Bool, nItems)

	var wg sync.WaitGroup
	ownerDone := make(chan struct{})

	// Owner: mostly Push, occasional Pop.
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewSource(1))
		for i := 0; i < nItems; i++ {
			d.Push(i)
			if rng.Intn(8) == 0 {
				if v, ok := d.Pop(); ok {
					if v < 0 || v >= nItems || consumed[v].Swap(true) {
						t.Errorf("invalid Pop value %d (i=%d)", v, i)
						return
					}
				}
			}
		}
		// Drain.
		for {
			v, ok := d.Pop()
			if !ok {
				break
			}
			if v < 0 || v >= nItems || consumed[v].Swap(true) {
				t.Errorf("invalid drain Pop %d", v)
				return
			}
		}
		close(ownerDone)
	}()

	for i := 0; i < nThieves; i++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(seed)))
			for {
				v, ok := d.Steal()
				if ok {
					if v < 0 || v >= nItems || consumed[v].Swap(true) {
						t.Errorf("invalid Steal value %d", v)
						return
					}
					continue
				}
				select {
				case <-ownerDone:
					if d.IsEmpty() {
						return
					}
				default:
				}
				if rng.Intn(4) == 0 {
					runtime.Gosched()
				}
			}
		}(i + 1)
	}

	wg.Wait()
	missing := 0
	for i := 0; i < nItems; i++ {
		if !consumed[i].Load() {
			missing++
		}
	}
	require.Zero(t, missing, "items not consumed")
	require.True(t, d.IsEmpty())
}
