package utils

// Concurrent typeutils.Map
//
// Most of the code is borrowed, but the core map data structure used is an
// xsync.Map

import (
	"bytes"
	"fmt"
	"go/types"
	"sync"
	"sync/atomic"

	"github.com/puzpuzpuz/xsync/v4"
	"golang.org/x/tools/go/types/typeutil"
)

// TypeMap is a hash-table-based mapping from types (types.Type) to
// arbitrary values.  The concrete types that implement
// the Type interface are pointers.  Since they are not canonicalized,
// == cannot be used to check for equivalence, and thus we cannot
// simply use a Go map.
//
// Just as with map[K]V, a nil *TypeMap is a valid empty map.
//
// Read-only map operations ([TypeMap.At], [TypeMap.Len], and so on) may
// safely be called concurrently.
//
// TODO(adonovan): deprecate in favor of https://go.dev/issues/69420
// and 69559, if the latter proposals for a generic hash-map type and
// a types.Hash function are accepted.
type TypeMap struct {
	table  *xsync.Map[uint32, *bucket]
	once   sync.Once
	length atomic.Int64 // number of map entries
}

func (m *TypeMap) getTable() *xsync.Map[uint32, *bucket] {
	m.once.Do(func() {
		m.table = xsync.NewMap[uint32, *bucket]()
	})
	return m.table
}

var theHasher typeutil.Hasher

func hash(t types.Type) uint32 {
	return theHasher.Hash(t)
}

// entry is an entry (key/value association) in a hash bucket.
type entry struct {
	key   types.Type
	value any
}

type bucket struct {
	entries []entry
	lk      sync.RWMutex
}

// SetHasher has no effect.
//
// It is a relic of an optimization that is no longer profitable. Do
// not use [Hasher], [MakeHasher], or [SetHasher] in new code.
func (m *TypeMap) SetHasher(typeutil.Hasher) {}

// Delete removes the entry with the given key, if any.
// It returns true if the entry was found.
func (m *TypeMap) Delete(key types.Type) bool {
	if m != nil {
		hash := hash(key)
		if bucket, ok := m.getTable().Load(hash); ok {
			bucket.lk.Lock()
			defer bucket.lk.Unlock()

			for i, e := range bucket.entries {
				if e.key != nil && types.Identical(key, e.key) {
					// We can't compact the bucket as it
					// would disturb iterators.
					bucket.entries[i] = entry{}
					m.length.Add(-1)
					return true
				}
			}
		}
	}
	return false
}

// At returns the map entry for the given key.
// The result is nil if the entry is not present.
func (m *TypeMap) At(key types.Type) any {
	if m == nil {
		return nil
	}
	if bucket, ok := m.getTable().Load(hash(key)); ok {
		bucket.lk.RLock()
		defer bucket.lk.RUnlock()
		for _, e := range bucket.entries {
			if e.key != nil && types.Identical(key, e.key) {
				return e.value
			}
		}
	}
	return nil
}

// Set sets the map entry for key to val,
// and returns the previous entry, if any.
func (m *TypeMap) Set(key types.Type, value any) (prev any) {
	hash := hash(key)
	tbl := m.getTable()
	b, _ := tbl.LoadOrStore(hash, &bucket{})

	b.lk.Lock()
	defer b.lk.Unlock()

	var hole *entry
	for i, e := range b.entries {
		if e.key == nil {
			hole = &b.entries[i]
		} else if types.Identical(key, e.key) {
			prev = e.value
			b.entries[i].value = value
			return
		}
	}

	if hole != nil {
		*hole = entry{key, value} // overwrite deleted entry
	} else {
		b.entries = append(b.entries, entry{key, value})
	}

	m.length.Add(1)

	return
}

// SetNoLock performs Set without acquiring the bucket writer lock.
// The caller MUST ensure exclusive access to this key, typically by
// being the goroutine that won the LoadOrStore race for this key's
// concreteTypeInfo/interfaceTypeInfo.
func (m *TypeMap) SetNoLock(key types.Type, value any) (prev any) {
	hash := hash(key)
	tbl := m.getTable()
	b, _ := tbl.LoadOrStore(hash, &bucket{})

	var hole *entry
	for i, e := range b.entries {
		if e.key == nil {
			hole = &b.entries[i]
		} else if types.Identical(key, e.key) {
			prev = e.value
			b.entries[i].value = value
			return
		}
	}

	if hole != nil {
		*hole = entry{key, value} // overwrite deleted entry
	} else {
		b.entries = append(b.entries, entry{key, value})
	}

	m.length.Add(1)

	return
}

// LoadOrStore performs an atomic load-or-store operation over a type map
func (m *TypeMap) LoadOrStore(key types.Type, value any) (any, bool) {
	h := hash(key)
	tbl := m.getTable()

	// Fast path: check if bucket exists first
	if bucket, ok := tbl.Load(h); ok {
		// Try with read lock first
		bucket.lk.RLock()
		for _, e := range bucket.entries {
			if e.key != nil && types.Identical(key, e.key) {
				bucket.lk.RUnlock()
				return e.value, true
			}
		}
		bucket.lk.RUnlock()

		// Key not found, acquire write lock and add entry
		bucket.lk.Lock()
		defer bucket.lk.Unlock()

		// Double-check after acquiring write lock
		for _, e := range bucket.entries {
			if e.key != nil && types.Identical(key, e.key) {
				return e.value, true
			}
		}

		// Add the new entry
		var hole *entry
		for i, e := range bucket.entries {
			if e.key == nil {
				hole = &bucket.entries[i]
			} else if types.Identical(key, e.key) {
				bucket.entries[i].value = value
			}
		}
		if hole != nil {
			*hole = entry{key, value} // overwrite deleted entry
		} else {
			bucket.entries = append(bucket.entries, entry{key, value})
		}

		m.length.Add(1)
		return value, false
	}

	// Slow path: bucket doesn't exist, create new one
	newBucket := &bucket{entries: []entry{{key, value}}}
	actual, loaded := tbl.LoadOrStore(h, newBucket)
	if loaded {
		// Another goroutine created the bucket, retry with the existing bucket
		actual.lk.Lock()
		defer actual.lk.Unlock()

		// Check if key was added by another goroutine
		for _, e := range actual.entries {
			if e.key != nil && types.Identical(key, e.key) {
				return e.value, true
			}
		}

		// Add the new entry
		actual.entries = append(actual.entries, entry{key, value})
		m.length.Add(1)
		return value, false
	}

	// New bucket was created successfully
	m.length.Add(1)
	return value, false
}

// Len returns the number of map entries.
func (m *TypeMap) Len() int {
	if m != nil {
		return int(m.length.Load())
	}
	return 0
}

// Iterate calls function f on each entry in the map in unspecified order.
//
// If f should mutate the map, Iterate provides the same guarantees as
// Go maps: if f deletes a map entry that Iterate has not yet reached,
// f will not be invoked for it, but if f inserts a map entry that
// Iterate has not yet reached, whether or not f will be invoked for
// it is unspecified.
func (m *TypeMap) Iterate(f func(key types.Type, value any)) {
	if m != nil {
		m.getTable().Range(func(_ uint32, bucket *bucket) bool {
			bucket.lk.RLock()
			defer bucket.lk.RUnlock()
			for _, e := range bucket.entries {
				// If the key is nil, then it is a hole. Holes are ignored.
				if e.key != nil {
					f(e.key, e.value)
				}
			}
			return true
		})
	}
}

// Keys returns a new slice containing the set of map keys.
// The order is unspecified.
func (m *TypeMap) Keys() []types.Type {
	keys := make([]types.Type, 0, m.Len())
	m.Iterate(func(key types.Type, _ any) {
		keys = append(keys, key)
	})
	return keys
}

func (m *TypeMap) toString(values bool) string {
	if m == nil {
		return "{}"
	}
	var buf bytes.Buffer
	fmt.Fprint(&buf, "{")
	sep := ""
	m.Iterate(func(key types.Type, value any) {
		fmt.Fprint(&buf, sep)
		sep = ", "
		fmt.Fprint(&buf, key)
		if values {
			fmt.Fprintf(&buf, ": %q", value)
		}
	})
	fmt.Fprint(&buf, "}")
	return buf.String()
}

// String returns a string representation of the map's entries.
// Values are printed using fmt.Sprintf("%v", v).
// Order is unspecified.
func (m *TypeMap) String() string {
	return m.toString(true)
}

// KeysString returns a string representation of the map's key set.
// Order is unspecified.
func (m *TypeMap) KeysString() string {
	return m.toString(false)
}
