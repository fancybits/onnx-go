package gorgonnx

import (
	"errors"

	"github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Graph is the top structure that should be compatible with
//    backend.ComputationGraph
// It holds a gorgonia.ExprGraph that is populated on the first call to the
// Run() method
type Graph struct {
	g         *simple.WeightedDirectedGraph
	exprgraph *gorgonia.ExprGraph
	m         gorgonia.VM
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

// RunWithVM runs the graph with the specified VM type ("lisp" or "tape").
// This is primarily for testing to compare VM behaviors.
func (g *Graph) RunWithVM(vmType string) error {
	if g.exprgraph == nil {
		err := g.PopulateExprgraph()
		if err != nil {
			return err
		}
	}

	// Create VM based on type
	switch vmType {
	case "tape":
		g.m = gorgonia.NewTapeMachine(g.exprgraph)
	case "lisp":
		// Use LispMachine for inference.
		// TapeMachine has a register reuse optimization that can cause incorrect results
		// when operations share memory in complex graphs.
		// LispMachine executes operations as it traverses the graph without register pooling.
		g.m = gorgonia.NewLispMachine(g.exprgraph, gorgonia.ExecuteFwdOnly())
	default:
		return errors.New("unknown VM type: " + vmType + " (use 'lisp' or 'tape')")
	}
	defer g.m.Close()

	err := g.m.RunAll()
	if err != nil {
		return err
	}

	// Now sets the output tensor
	for i := 0; i < len(g.roots); i++ {
		root := g.Node(g.roots[i]).(*Node)
		var ok bool
		if root.gorgoniaNode == nil {
			return errors.New("root node is nil")
		}
		root.t, ok = root.gorgoniaNode.Value().(tensor.Tensor)
		if !ok {
			return errors.New("root node is not a tensor")
		}
	}
	return nil
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
		// Input tensors will be set again via SetInput before next Run()
		if n.operation != nil {
			n.t = nil
		}
	}
	// Clear the exprgraph - it will be rebuilt on next Run()
	g.exprgraph = nil
	if g.m != nil {
		g.m.Close()
		g.m = nil
	}
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
