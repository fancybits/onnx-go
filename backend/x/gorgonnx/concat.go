package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type concat struct {
	axis int
}

func init() {
	register("Concat", newConcat)
}

func newConcat() operator {
	return &concat{}
}

func (a *concat) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)

	// Filter out nil nodes and collect valid ones. valid stays index-aligned
	// with nodes so that a filtered-out child cannot shift the provenance and
	// tensor lookups below onto the wrong input.
	var nodes []*gorgonia.Node
	var valid []*Node
	for i := 0; i < len(children); i++ {
		if children[i].gorgoniaNode != nil {
			nodes = append(nodes, children[i].gorgoniaNode)
			valid = append(valid, children[i])
		}
	}

	if len(nodes) == 0 {
		return errors.New("concat: no valid input nodes")
	}

	if len(nodes) == 1 {
		// Single input - just pass through. The alias carries the input's value
		// and provenance with it, as Identity's does.
		n.gorgoniaNode = nodes[0]
		n.t = valid[0].t
		n.constant = valid[0].constant
		return nil
	}

	// Fold only when every input is a genuine compile-time constant, so that
	// shape computations still have their values at graph construction time.
	// Carrying a tensor is not evidence of one: gorgonia binds a value to every
	// leaf, and a graph input's may already hold this run's data, which folding
	// would bake in for every later run.
	if allConstInputs(valid) {
		result, err := concatTensors(a.axis, valid)
		if err != nil {
			return fmt.Errorf("concat constants: %w", err)
		}
		// Set both t (for immediate access in shape computations) and gorgoniaNode (for graph execution)
		n.t = result
		n.constant = true
		n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("concat_const")))
		return nil
	}

	// gorgonia.Concat panics on an empty operand (an all-zero-size dense) —
	// the same bug concatTensors above works around for the all-constant
	// path. Here at least one input is not a proven constant, so the value
	// cannot be folded away at build time, but the *shape* of every input is
	// still static and known now: filter out any input whose concat axis is
	// zero before handing the rest to gorgonia. This only drops operands
	// that contribute no elements to the result, so it cannot change what
	// the op computes, and it runs after the constant-provenance check
	// above, so it cannot turn a non-constant concat into a constant one.
	axis := a.axis
	if axis < 0 {
		axis += len(nodes[0].Shape())
	}
	var symNodes []*gorgonia.Node
	for _, gn := range nodes {
		shape := gn.Shape()
		if axis < len(shape) && shape[axis] == 0 {
			continue
		}
		symNodes = append(symNodes, gn)
	}

	if len(symNodes) == 0 {
		// Every input is empty along the concat axis: nothing to
		// concatenate. Match concatTensors' degenerate case.
		n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph,
			tensor.New(tensor.WithShape(0), tensor.Of(nodes[0].Dtype())),
			gorgonia.WithName(getUniqNodeName("concat_empty")))
		return nil
	}

	var err error
	n.gorgoniaNode, err = gorgonia.Concat(axis, symNodes...)
	return err
}

// concatTensors concatenates constant tensors along the specified axis. Every
// child must carry compile-time constant provenance; callers establish that.
func concatTensors(axis int, children []*Node) (tensor.Tensor, error) {
	if len(children) == 0 {
		return nil, errors.New("concat: no tensors to concatenate")
	}

	var tensors []tensor.Tensor
	var dtype tensor.Dtype
	for i, child := range children {
		t := constTensorFromNode(child)
		if t == nil {
			return nil, fmt.Errorf("concat: input %d is not a constant tensor", i)
		}
		if dtype == (tensor.Dtype{}) {
			dtype = t.Dtype()
		}
		// Filter out empty tensors - workaround for tensor.Concat bug
		// https://github.com/gorgonia/tensor panics on empty tensors
		if t.Shape().TotalSize() > 0 {
			tensors = append(tensors, t)
		}
	}

	// Handle negative axis
	if len(tensors) > 0 {
		firstShape := tensors[0].Shape()
		if axis < 0 {
			axis = len(firstShape) + axis
		}
	}

	// Handle edge cases after filtering
	if len(tensors) == 0 {
		return tensor.New(tensor.WithShape(0), tensor.Of(dtype)), nil
	}
	if len(tensors) == 1 {
		return tensors[0], nil
	}

	return tensor.Concat(axis, tensors[0], tensors[1:]...)
}

func (a *concat) init(o onnx.Operation) error {
	axis, ok := o.Attributes["axis"]
	if !ok {
		return errors.New("concat: expected axis attribute is not found")
	}
	err := errors.New("axis in not an int")
	if axis, ok := axis.(int64); ok {
		a.axis = int(axis)
		err = nil
	}
	return err
}
