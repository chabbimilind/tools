package srta

import (
	"testing"

	"golang.org/x/tools/go/callgraph/internal/rtautil/rtatest"
)

func TestCorrectness(t *testing.T) {
	rtatest.AssertMatchesStdlib(t, New())
}
