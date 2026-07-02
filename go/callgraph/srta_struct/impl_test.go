package srta_struct

import (
	"testing"

	"golang.org/x/tools/go/callgraph/rtalib/rtatest"
)

func TestCorrectness(t *testing.T) {
	rtatest.AssertMatchesStdlib(t, New())
}
