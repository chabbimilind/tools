package srta_kumo_random

import (
	rtalib "golang.org/x/tools/go/callgraph/rtalib"
	"golang.org/x/tools/go/callgraph/srta_kumo"
	"golang.org/x/tools/go/ssa"
)

// RTA delegates to srta_kumo with RandomMethodStrategy.
type RTA struct {
	inner *srta_kumo.RTA
}

// New creates a new RTA instance that uses random method selection.
func New() *RTA {
	return &RTA{inner: srta_kumo.New()}
}

// Analyze performs RTA with random MI method selection strategy.
func (a *RTA) Analyze(roots []*ssa.Function, buildCallGraph bool, opts ...rtalib.AnalyzeOption) rtalib.Result {
	opts = append(opts, rtalib.WithMethodSelectionStrategy(rtalib.RandomMethodStrategy))
	return a.inner.Analyze(roots, buildCallGraph, opts...)
}
