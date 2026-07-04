package srta_kumo

import (
	"testing"

	rtalib "golang.org/x/tools/go/callgraph/internal/rtautil"
	"golang.org/x/tools/go/callgraph/internal/rtautil/rtatest"
)

func TestCorrectness(t *testing.T) {
	t.Run("RarestMethodByConcreteType", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New())
	})
	t.Run("RarestMethodByInterfaceType", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithMethodSelectionStrategy(rtalib.RarestMethodByInterfaceType))
	})
	t.Run("RandomMethodStrategy", func(t *testing.T) {
		rtatest.AssertMatchesStdlib(t, New(), rtalib.WithMethodSelectionStrategy(rtalib.RandomMethodStrategy))
	})
}

func TestResultMethods(t *testing.T) {
	_, entry := rtatest.LoadProgramAndRoots(t)
	result := New().Analyze(entry, true).(*Result)

	_ = result.MCBucketSizePercentiles()
	_ = result.MIBucketSizePercentiles()
	_ = result.InterfaceMethodCountBuckets()
	_ = result.MethodsPerConcretePercentiles()
	_ = result.MethodsPerInterfacePercentiles()
	_ = result.ConcreteMethodCountBuckets()
}
