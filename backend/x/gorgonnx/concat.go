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

	// Filter out nil nodes and collect valid ones
	var nodes []*gorgonia.Node
	for i := 0; i < len(children); i++ {
		if children[i].gorgoniaNode != nil {
			nodes = append(nodes, children[i].gorgoniaNode)
		}
	}

	if len(nodes) == 0 {
		return errors.New("concat: no valid input nodes")
	}

	if len(nodes) == 1 {
		// Single input - just pass through
		n.gorgoniaNode = nodes[0]
		return nil
	}

	// Check if all inputs are constants - if so, perform concat immediately
	// This is important for shape computations that need values at graph construction time
	allConstants := true
	for i, node := range nodes {
		// Check both gorgoniaNode.Value() and the node's t field
		hasValue := node.Value() != nil || children[i].t != nil
		if !hasValue {
			allConstants = false
			break
		}
	}

	if allConstants {
		result, err := concatTensors(a.axis, nodes, children)
		if err != nil {
			return fmt.Errorf("concat constants: %w", err)
		}
		// Set both t (for immediate access in shape computations) and gorgoniaNode (for graph execution)
		n.t = result
		n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("concat_const")))
		return nil
	}

	var err error
	n.gorgoniaNode, err = gorgonia.Concat(a.axis, nodes...)
	return err
}

// concatTensors concatenates constant tensors along the specified axis
func concatTensors(axis int, nodes []*gorgonia.Node, children []*Node) (tensor.Tensor, error) {
	if len(nodes) == 0 {
		return nil, errors.New("concat: no tensors to concatenate")
	}

	var tensors []tensor.Tensor
	var dtype tensor.Dtype
	for i, node := range nodes {
		// Try gorgoniaNode.Value() first, then fall back to node's t field
		var t tensor.Tensor
		var ok bool
		if node.Value() != nil {
			t, ok = node.Value().(tensor.Tensor)
		}
		if !ok && i < len(children) && children[i].t != nil {
			t = children[i].t
			ok = true
		}
		if !ok {
			return nil, fmt.Errorf("concat: input %d is not a tensor", i)
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
