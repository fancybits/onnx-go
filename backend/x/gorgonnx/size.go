package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size

func init() {
	register("Size", newSize)
}

func newSize() operator {
	return &sizeOp{}
}

type sizeOp struct{}

func (s *sizeOp) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	if err := checkCondition(children, 1); err != nil {
		return err
	}

	input := children[0]

	// Get the shape of the input
	var shape tensor.Shape
	if input.gorgoniaNode != nil {
		shape = input.gorgoniaNode.Shape()
	} else if input.t != nil {
		shape = input.t.Shape()
	}

	// Calculate total size
	totalSize := int64(shape.TotalSize())

	// Create a 0-dimensional tensor (scalar) with the size
	// ONNX Size returns a tensor with shape (1,) containing int64
	result := tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{totalSize}))

	// Set both t (for immediate access in shape computations) and gorgoniaNode (for graph execution)
	n.t = result
	n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("size")))
	return nil
}

func (s *sizeOp) init(o onnx.Operation) error {
	return nil
}
