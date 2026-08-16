package gorgonnx

import (
	"errors"

	"github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Graph is the top structure that should be compatible with
//    backend.ComputationGraph
// It holds a gorgonia.ExprGraph that is populated on the first call to the
// Run() method
type Graph struct {
	g         *weightedDirectedGraph
	exprgraph *gorgonia.ExprGraph
	m         gorgonia.VM
	vmType    string // the VM kind g.m was compiled as, so a kind change recompiles
	roots     []int64
	groups    [][]*Node // a reference of all the nodes that belongs to a group
}

// SetVM used by the backend
// A call to this method do not call the PopulateExprgraph method
// it is the responsibility of the caller to call it before
/*
func (g *Graph) SetVM(vm gorgonia.VM) {
	g.m = vm
}
*/

// GetExprGraph returns the gorgonia graph; if the graph is nil, it populates the graph before returing it
func (g *Graph) GetExprGraph() (*gorgonia.ExprGraph, error) {
	var err error
	if g.exprgraph == nil {
		err = g.PopulateExprgraph()
	}
	return g.exprgraph, err
}

// ApplyOperation to fulfill the onnx.Backend interface
func (g *Graph) ApplyOperation(o onnx.Operation, ns ...graph.Node) error {
	nodes := make([]*Node, len(ns))
	for i, n := range ns {
		n.(*Node).operation = &o
		nodes[i] = n.(*Node)
	}
	g.groups = append(g.groups, nodes)
	return nil
}

// Run the graph. It populate the underlying exprgraph if the graph is nil
func (g *Graph) Run() error {
	return g.RunWithVM("tape")
}

// prepare builds the exprgraph and VM the bound inputs require, without
// executing. It reports whether the VM is newly compiled, which a LispMachine
// must not have Reset called on before its first run.
func (g *Graph) prepare(vmType string) (fresh bool, err error) {
	// A new input shape invalidates the built graph: the old shape is baked
	// into every node derived from it.
	if g.exprgraph != nil && g.inputShapeChanged() {
		g.Reset()
	}

	if g.exprgraph == nil {
		if err := g.PopulateExprgraph(); err != nil {
			return false, err
		}
		// A VM held from before is compiled against nodes that no longer exist.
		g.closeVM()
	}

	// Validate before touching the cached VM: an unknown name must not cost
	// the caller the compiled program it already had.
	switch vmType {
	case "tape", "lisp":
	default:
		return false, errors.New("unknown VM type: " + vmType + " (use 'lisp' or 'tape')")
	}

	// The compiled program depends only on the exprgraph, not on the values
	// bound into it, so it is kept and reused; compiling costs about as much
	// as executing. Everything that discards the exprgraph must therefore
	// also drop the VM.
	if g.m != nil && (g.vmType != vmType || !vmReusable(vmType)) {
		g.closeVM()
	}
	if g.m == nil {
		switch vmType {
		case "tape":
			g.m = gorgonia.NewTapeMachine(g.exprgraph)
		case "lisp":
			g.m = gorgonia.NewLispMachine(g.exprgraph, gorgonia.ExecuteFwdOnly())
		}
		g.vmType = vmType

		return true, nil
	}

	return false, nil
}

// vmReusable reports whether a compiled VM of this kind can be rewound and run
// again.
//
// Only the tape machine can. lispMachine.Reset rewinds the backward pass --
// it sets fwd to the last node, not the first -- while the forward loop runs
// while fwd < len(sorted). A reused LispMachine therefore re-executes only its
// final node and returns the previous run's values for everything else, with
// no error. So lisp gets a fresh VM per run, as it did before VM caching.
func vmReusable(vmType string) bool {
	return vmType == "tape"
}

// RunWithVM runs the graph with the specified VM type ("lisp" or "tape").
// This is primarily for testing to compare VM behaviors.
func (g *Graph) RunWithVM(vmType string) error {
	fresh, err := g.prepare(vmType)
	if err != nil {
		return err
	}
	if !fresh {
		// Rewind the previous run. A just-compiled VM is already rewound.
		g.m.Reset()
	}

	err = g.m.RunAll()
	if err != nil {
		return err
	}

	// Now sets the output tensor
	for i := 0; i < len(g.roots); i++ {
		root := g.Node(g.roots[i]).(*Node)
		if root.gorgoniaNode == nil {
			return errors.New("root node is nil")
		}
		v := root.gorgoniaNode.Value()
		if t, ok := v.(tensor.Tensor); ok {
			root.t = t
			continue
		}
		// Some ops (e.g. gorgonia's builtin Lt/Gt on 0-d operands) legitimately
		// return a boxed gorgonia.Scalar rather than a tensor.Tensor when the
		// node's static type is scalar. Wrap it as a 0-d tensor so it can still
		// be surfaced as a root output.
		if s, ok := v.(gorgonia.Scalar); ok {
			root.t = tensor.New(tensor.FromScalar(s.Data()))
			continue
		}
		return errors.New("root node is not a tensor")
	}
	return nil
}

// inputShapeChanged reports whether any leaf carries a tensor the built graph
// can no longer accept. Operation nodes are skipped: their shapes follow from
// their inputs.
func (g *Graph) inputShapeChanged() bool {
	it := g.g.Nodes()
	for it.Next() {
		n := it.Node().(*Node)
		if n.operation == nil && n.shapeChanged() {
			return true
		}
	}

	return false
}

// closeVM releases the cached VM, if any. Safe to call when none is held.
func (g *Graph) closeVM() {
	if g.m != nil {
		g.m.Close()
		g.m = nil
		g.vmType = ""
	}
}

// Close releases the compiled VM, which is retained between runs. The graph
// stays usable: the next Run recompiles.
//
// This is not the graph's bulk -- the VM is a few MiB against ~150 MiB of
// intermediates held by the operation nodes. Reset frees those, but measured
// worse: they are reused in place, so returning them only to reallocate them
// churns the heap.
func (g *Graph) Close() {
	g.closeVM()
}

// Reset clears the gorgonia execution graph, allowing the model to be
// rebuilt with different input shapes (e.g., different batch sizes).
// Call this before SetInput when changing batch dimensions.
func (g *Graph) Reset() {
	// Clear all gorgoniaNode references and intermediate tensor values
	it := g.g.Nodes()
	for it.Next() {
		n := it.Node().(*Node)
		n.gorgoniaNode = nil
		// Clear tensor values for operation nodes (not inputs/constants)
		// Input tensors will be set again via SetInput before next Run().
		// Drop the provenance flag with the value it describes: the rebuild
		// re-runs the operator, which decides afresh whether its result is a
		// constant, and a flag left over from the previous build could vouch
		// for a tensor this one never produced.
		if n.operation != nil {
			n.t = nil
			n.constant = false
		}
	}
	// Clear the exprgraph - it will be rebuilt on next Run()
	g.exprgraph = nil
	g.closeVM()
}

// PopulateExprgraph creates the underlynig graph by walking the current graph
func (g *Graph) PopulateExprgraph() error {
	g.exprgraph = gorgonia.NewGraph()
	// Find the root nodes
	// TODO make it more efficient
	g.roots = make([]int64, 0)
	it := g.g.Nodes()
	for it.Next() {
		n := it.Node()
		if g.g.To(n.ID()).Len() == 0 {
			g.roots = append(g.roots, n.ID())
		}
	}
	return g.populateExprgraph()
}
