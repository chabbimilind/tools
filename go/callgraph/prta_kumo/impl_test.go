package prta_kumo

import (
	"testing"

	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/rtalib/rtatest"
)

func TestCorrectness(t *testing.T) {
	t.Run("RarestMethodByConcreteType", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4))
	})
	t.Run("RarestMethodByInterfaceType", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4), rtalib.WithMethodSelectionStrategy(rtalib.RarestMethodByInterfaceType))
	})
	t.Run("RandomMethodStrategy", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithNumWorkers(4), rtalib.WithMethodSelectionStrategy(rtalib.RandomMethodStrategy))
	})
}

func TestResultMethods(t *testing.T) {
	_, entry := rtatest.LoadProgramAndRoots(t)
	result := New().Analyze(entry, true, rtalib.WithNumWorkers(4)).(*Result)

	_ = result.MCBucketSizePercentiles()
	_ = result.MIBucketSizePercentiles()
	_ = result.InterfaceMethodCountBuckets()
}
