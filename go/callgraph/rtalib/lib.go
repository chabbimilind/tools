package rta

import (
	"sort"

	"golang.org/x/tools/go/callgraph"
	rtapkg "golang.org/x/tools/go/callgraph/rta"
	"golang.org/x/tools/go/callgraph/rtalib/utils"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types/typeutil"
)

// Result defines methods to access RTA analysis results.
// Different implementations can use different internal representations
// (e.g., sync.Map for parallel, regular map for sequential) and convert
// on demand via Materialize().
type Result interface {
	// GetCallGraph returns the discovered callgraph.
	// It does not include edges for calls made via reflection.
	GetCallGraph() *callgraph.Graph

	// GetReachable returns the set of reachable functions and methods.
	// This includes exported methods of runtime types, since
	// they may be accessed via reflection.
	// The value indicates whether the function is address-taken.
	GetReachable() map[*ssa.Function]struct{ AddrTaken bool }

	// GetRuntimeTypes returns the set of types that are needed at
	// runtime, for interfaces or reflection.
	//
	// The value indicates whether the type is inaccessible to reflection.
	// Consider:
	//     type A struct{B}
	//     fmt.Println(new(A))
	// Types *A, A and B are accessible to reflection, but the unnamed
	// type struct{B} is not.
	GetRuntimeTypes() typeutil.Map

	// Materialize converts the result to the standard golang.org/x/tools/go/callgraph/rta.Result.
	// This allows implementations to use optimal internal representations
	// (e.g., sync.Map for parallel implementations) and convert on demand.
	Materialize() *rtapkg.Result
}

// MethodSelectionStrategy controls how the MI index selects which method
// to index an interface under.
type MethodSelectionStrategy int

const (
	// RarestMethodByConcreteType selects the method with the fewest concrete types in MC (default).
	RarestMethodByConcreteType MethodSelectionStrategy = iota
	// RarestMethodByInterfaceType selects the method used by the fewest interfaces.
	RarestMethodByInterfaceType
	// RandomMethodStrategy selects a random method (ablation baseline).
	RandomMethodStrategy
)

// AnalyzeOption configures optional Analyze behavior.
type AnalyzeOption func(*AnalyzeConfig)

// AnalyzeConfig holds optional configuration for Analyze.
type AnalyzeConfig struct {
	NumWorkers              int
	MethodSelectionStrategy MethodSelectionStrategy
}

// DefaultAnalyzeConfig returns an AnalyzeConfig with defaults applied, then opts.
func DefaultAnalyzeConfig(opts ...AnalyzeOption) AnalyzeConfig {
	cfg := AnalyzeConfig{NumWorkers: 1, MethodSelectionStrategy: RarestMethodByConcreteType}
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// WithNumWorkers sets the number of workers for parallel RTA implementations.
// n must be at least 1.
func WithNumWorkers(n int) AnalyzeOption {
	return func(c *AnalyzeConfig) {
		if n < 1 {
			n = 1
		}
		c.NumWorkers = n
	}
}

// WithMethodSelectionStrategy sets the MI method selection strategy.
func WithMethodSelectionStrategy(s MethodSelectionStrategy) AnalyzeOption {
	return func(c *AnalyzeConfig) { c.MethodSelectionStrategy = s }
}

// RTA defines the contract for Rapid Type Analysis implementations.
type RTA interface {
	// Analyze performs Rapid Type Analysis starting at the specified root functions.
	// It returns nil if no roots were specified.
	//
	// The root functions must be one or more entrypoints (main and init functions)
	// of a complete SSA program, with function bodies for all dependencies,
	// constructed with the [ssa.InstantiateGenerics] mode flag.
	//
	// If buildCallGraph is true, Result.CallGraph will contain a call graph;
	// otherwise, only the other fields (reachable functions) are populated.
	Analyze(roots []*ssa.Function, buildCallGraph bool, opts ...AnalyzeOption) Result
}

// ImplementsCounter is an optional interface that Result implementations
// can satisfy to report the number of implements checks performed during analysis.
type ImplementsCounter interface {
	ImplementsCallCount() int
}

// TypeCounter reports discovered concrete and interface types.
type TypeCounter interface {
	NumConcreteTypes() int
	NumInterfaceTypes() int
}

// ImplementsSuccessFailCounter reports success/fail breakdown of implements checks.
type ImplementsSuccessFailCounter interface {
	ImplementsSuccessCount() int
	ImplementsFailCount() int
}

// ImplementsSourceCounter reports checks by origin (from interfaces() vs implementations()).
type ImplementsSourceCounter interface {
	ChecksFromInterfaces() int
	ChecksFromImplementations() int
}

// ImplementsDirectionCounter reports per-direction success/fail breakdown.
type ImplementsDirectionCounter interface {
	FailsFromInterfaces() int
	SuccessFromInterfaces() int
	FailsFromImplementations() int
	SuccessFromImplementations() int
}

// CallSiteCounter reports call site type breakdown.
type CallSiteCounter interface {
	StaticCallSites() int
	IndirectCallSites() int
	InvokeCallSites() int
	TotalCallSites() int
}

// InvokeLookupCounter reports invoke callgraph edges attached at the two
// algorithm sites that produce them: the visitInvoke path (a new invoke
// site arrives and is matched against currently-known concrete impls)
// and the addRuntimeType path (a new concrete type arrives and is
// matched against currently-known invoke sites). Together these count
// all invoke-edge attachments; the breakdown lets us assess the impact
// of optimizations that reduce per-site / per-type work (cached cmethod
// lists, per-method invoke-site indexing) by re-measuring at the same
// two call sites without changing the metric definition.
type InvokeLookupCounter interface {
	InvokeLookupsFromVisitInvoke() int
	InvokeLookupsFromAddRuntimeType() int
}

// Percentiles holds standard percentile values.
type Percentiles struct {
	P50, P90, P95, P99, P100 int
}

// IndexStats reports MC/MI bucket size percentiles and interface method-count distribution.
type IndexStats interface {
	MCBucketSizePercentiles() Percentiles
	MIBucketSizePercentiles() Percentiles
	// InterfaceMethodCountBuckets returns the fraction of interfaces in each
	// method-count bucket: [1, 2, 4, 8, 16, 32, 64, >64].
	InterfaceMethodCountBuckets() [8]float64
}

// MethodStats reports methods-per-type percentiles.
type MethodStats interface {
	MethodsPerConcretePercentiles() Percentiles
	MethodsPerInterfacePercentiles() Percentiles
	// ConcreteMethodCountBuckets returns the fraction of concrete types in each
	// method-count bucket: [1, 2, 4, 8, 16, 32, 64, >64].
	ConcreteMethodCountBuckets() [8]float64
}

// DedupTiming reports pre/post dedup timing and memory.
type DedupTiming interface {
	PreDedupDurationMs() float64
	PostDedupDurationMs() float64
	PreDedupHeapDeltaMB() float64
	PostDedupHeapDeltaMB() float64
	DuplicatedEdgeCount() int
}

// BaseMetrics holds all scalar metric fields that Result types expose.
// Embed in a flavor's Result to inherit getter methods.
// Unused fields stay zero; consumers access via type assertions.
type BaseMetrics struct {
	ImplementsCallCountVal             int
	ImplementsSuccessCountVal          int
	ImplementsFailCountVal             int
	NumConcreteTypesVal                int
	NumInterfaceTypesVal               int
	ChecksFromInterfacesVal            int
	ChecksFromImplementationsVal       int
	FailsFromInterfacesVal             int
	SuccessFromInterfacesVal           int
	FailsFromImplementationsVal        int
	SuccessFromImplementationsVal      int
	StaticCallSitesVal                 int
	IndirectCallSitesVal               int
	InvokeCallSitesVal                 int
	PreDedupDurationMsVal              float64
	PostDedupDurationMsVal             float64
	PreDedupHeapDeltaMBVal             float64
	PostDedupHeapDeltaMBVal            float64
	DuplicatedEdgeCountVal             int
	InvokeLookupsFromVisitInvokeVal    int
	InvokeLookupsFromAddRuntimeTypeVal int
}

func (m *BaseMetrics) ImplementsCallCount() int        { return m.ImplementsCallCountVal }
func (m *BaseMetrics) ImplementsSuccessCount() int     { return m.ImplementsSuccessCountVal }
func (m *BaseMetrics) ImplementsFailCount() int        { return m.ImplementsFailCountVal }
func (m *BaseMetrics) NumConcreteTypes() int           { return m.NumConcreteTypesVal }
func (m *BaseMetrics) NumInterfaceTypes() int          { return m.NumInterfaceTypesVal }
func (m *BaseMetrics) ChecksFromInterfaces() int       { return m.ChecksFromInterfacesVal }
func (m *BaseMetrics) ChecksFromImplementations() int  { return m.ChecksFromImplementationsVal }
func (m *BaseMetrics) FailsFromInterfaces() int        { return m.FailsFromInterfacesVal }
func (m *BaseMetrics) SuccessFromInterfaces() int      { return m.SuccessFromInterfacesVal }
func (m *BaseMetrics) FailsFromImplementations() int   { return m.FailsFromImplementationsVal }
func (m *BaseMetrics) SuccessFromImplementations() int { return m.SuccessFromImplementationsVal }
func (m *BaseMetrics) StaticCallSites() int            { return m.StaticCallSitesVal }
func (m *BaseMetrics) IndirectCallSites() int          { return m.IndirectCallSitesVal }
func (m *BaseMetrics) InvokeCallSites() int            { return m.InvokeCallSitesVal }

// InvokeLookupsFromVisitInvoke returns the number of invoke callgraph
// edges attached during visitInvoke processing. Optimized flavors with
// per-method cached cmethod lists still count every edge attached here,
// so this number stays comparable across flavors even when the per-edge
// CPU cost differs.
func (m *BaseMetrics) InvokeLookupsFromVisitInvoke() int { return m.InvokeLookupsFromVisitInvokeVal }

// InvokeLookupsFromAddRuntimeType returns the number of invoke callgraph
// edges attached during addRuntimeType processing (a new concrete type
// becoming reachable attaches edges from itself to every existing
// invoke site of every interface it implements).
func (m *BaseMetrics) InvokeLookupsFromAddRuntimeType() int {
	return m.InvokeLookupsFromAddRuntimeTypeVal
}
func (m *BaseMetrics) TotalCallSites() int {
	return m.StaticCallSitesVal + m.IndirectCallSitesVal + m.InvokeCallSitesVal
}
func (m *BaseMetrics) PreDedupDurationMs() float64   { return m.PreDedupDurationMsVal }
func (m *BaseMetrics) PostDedupDurationMs() float64  { return m.PostDedupDurationMsVal }
func (m *BaseMetrics) PreDedupHeapDeltaMB() float64  { return m.PreDedupHeapDeltaMBVal }
func (m *BaseMetrics) PostDedupHeapDeltaMB() float64 { return m.PostDedupHeapDeltaMBVal }
func (m *BaseMetrics) DuplicatedEdgeCount() int      { return m.DuplicatedEdgeCountVal }

// WorkerCounters holds per-worker counters for parallel RTA flavors.
// Embed in a flavor's rta working-state struct. Unused counters stay zero.
type WorkerCounters struct {
	ImplementsCallCounts                  utils.WorkerCounter
	ImplementsSuccessCounts               utils.WorkerCounter
	ImplementsFailCounts                  utils.WorkerCounter
	ChecksFromInterfacesCounts            utils.WorkerCounter
	ChecksFromImplementationsCounts       utils.WorkerCounter
	FailsFromInterfacesCounts             utils.WorkerCounter
	SuccessFromInterfacesCounts           utils.WorkerCounter
	FailsFromImplementationsCounts        utils.WorkerCounter
	SuccessFromImplementationsCounts      utils.WorkerCounter
	StaticCallSiteCounts                  utils.WorkerCounter
	IndirectCallSiteCounts                utils.WorkerCounter
	InvokeCallSiteCounts                  utils.WorkerCounter
	DuplicatedEdgeCounts                  utils.WorkerCounter
	InvokeLookupsFromVisitInvokeCounts    utils.WorkerCounter
	InvokeLookupsFromAddRuntimeTypeCounts utils.WorkerCounter
}

// NewWorkerCounters creates a WorkerCounters with all counters sized for numWorkers.
func NewWorkerCounters(numWorkers int) WorkerCounters {
	return WorkerCounters{
		ImplementsCallCounts:                  utils.NewWorkerCounter(numWorkers),
		ImplementsSuccessCounts:               utils.NewWorkerCounter(numWorkers),
		ImplementsFailCounts:                  utils.NewWorkerCounter(numWorkers),
		ChecksFromInterfacesCounts:            utils.NewWorkerCounter(numWorkers),
		ChecksFromImplementationsCounts:       utils.NewWorkerCounter(numWorkers),
		FailsFromInterfacesCounts:             utils.NewWorkerCounter(numWorkers),
		SuccessFromInterfacesCounts:           utils.NewWorkerCounter(numWorkers),
		FailsFromImplementationsCounts:        utils.NewWorkerCounter(numWorkers),
		SuccessFromImplementationsCounts:      utils.NewWorkerCounter(numWorkers),
		StaticCallSiteCounts:                  utils.NewWorkerCounter(numWorkers),
		IndirectCallSiteCounts:                utils.NewWorkerCounter(numWorkers),
		InvokeCallSiteCounts:                  utils.NewWorkerCounter(numWorkers),
		DuplicatedEdgeCounts:                  utils.NewWorkerCounter(numWorkers),
		InvokeLookupsFromVisitInvokeCounts:    utils.NewWorkerCounter(numWorkers),
		InvokeLookupsFromAddRuntimeTypeCounts: utils.NewWorkerCounter(numWorkers),
	}
}

// MergeInto sums all per-worker counters and writes the totals into m.
func (wc *WorkerCounters) MergeInto(m *BaseMetrics) {
	m.ImplementsCallCountVal = wc.ImplementsCallCounts.Sum()
	m.ImplementsSuccessCountVal = wc.ImplementsSuccessCounts.Sum()
	m.ImplementsFailCountVal = wc.ImplementsFailCounts.Sum()
	m.ChecksFromInterfacesVal = wc.ChecksFromInterfacesCounts.Sum()
	m.ChecksFromImplementationsVal = wc.ChecksFromImplementationsCounts.Sum()
	m.FailsFromInterfacesVal = wc.FailsFromInterfacesCounts.Sum()
	m.SuccessFromInterfacesVal = wc.SuccessFromInterfacesCounts.Sum()
	m.FailsFromImplementationsVal = wc.FailsFromImplementationsCounts.Sum()
	m.SuccessFromImplementationsVal = wc.SuccessFromImplementationsCounts.Sum()
	m.StaticCallSitesVal = wc.StaticCallSiteCounts.Sum()
	m.IndirectCallSitesVal = wc.IndirectCallSiteCounts.Sum()
	m.InvokeCallSitesVal = wc.InvokeCallSiteCounts.Sum()
	m.DuplicatedEdgeCountVal = wc.DuplicatedEdgeCounts.Sum()
	m.InvokeLookupsFromVisitInvokeVal = wc.InvokeLookupsFromVisitInvokeCounts.Sum()
	m.InvokeLookupsFromAddRuntimeTypeVal = wc.InvokeLookupsFromAddRuntimeTypeCounts.Sum()
}

// ComputePercentiles computes P50, P90, P95, P99, P100 from a slice of ints.
// The input slice is sorted in place. Returns zero Percentiles if data is empty.
func ComputePercentiles(data []int) Percentiles {
	if len(data) == 0 {
		return Percentiles{}
	}
	sort.Ints(data)
	n := len(data)
	percentile := func(p float64) int {
		idx := int(p / 100.0 * float64(n-1))
		if idx >= n {
			idx = n - 1
		}
		return data[idx]
	}
	return Percentiles{
		P50:  percentile(50),
		P90:  percentile(90),
		P95:  percentile(95),
		P99:  percentile(99),
		P100: data[n-1],
	}
}
