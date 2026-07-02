package prta_naive

import (
	"testing"

	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/rtalib/rtatest"
)

func TestCorrectness(t *testing.T) {
	rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4))
}
