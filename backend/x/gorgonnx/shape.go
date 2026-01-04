package gorgonnx

import (
	"errors"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	register("Shape", func() operator { return new(shape) })
}

type shape struct{}

func (*shape) apply(graph *Graph, nodes ...*Node) error {
	if len(nodes) != 1 {
		return errors.New("wrong number of input nodes")
	}
	children := getOrderedChildren(graph.g, nodes[0])
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}
	shapeSlice := children[0].gorgoniaNode.Shape()
	// ONNX expects shapes to be int64 tensors
	s := make([]int64, len(shapeSlice))
	for i, v := range shapeSlice {
		s[i] = int64(v)
	}
	t := tensor.New(tensor.WithShape(len(s)), tensor.WithBacking(s))
	// Set both t (for immediate access in shape computations) and gorgoniaNode (for graph execution)
	nodes[0].t = t
	nodes[0].gorgoniaNode = gorgonia.NodeFromAny(graph.exprgraph, t, gorgonia.WithName(getUniqNodeName("shape")))

	return nil
}

func (*shape) init(onnx.Operation) error {
	return nil
}
