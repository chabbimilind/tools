package utils

import (
	"fmt"
	"go/types"
	"hash/crc32"
)

// Fingerprint returns a bitmask with one bit set per method id,
// enabling 'implements' to quickly reject most candidates.
func Fingerprint(mset *types.MethodSet) uint64 {
	var space [128]byte // stack buffer; 128 bytes covers long Uber package paths in unexported method Ids
	var mask uint64
	for i := 0; i < mset.Len(); i++ {
		method := mset.At(i).Obj()
		sig := method.Type().(*types.Signature)
		sum := crc32.ChecksumIEEE(fmt.Appendf(space[:0], "%s/%d/%d",
			method.Id(),
			sig.Params().Len(),
			sig.Results().Len()))
		mask |= 1 << (sum % 64)
	}
	return mask
}
