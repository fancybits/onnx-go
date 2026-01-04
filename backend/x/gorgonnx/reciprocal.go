package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal

func init() {
	register("Reciprocal", newReciprocal)
}

type reciprocal struct{}

func newReciprocal() operator {
	return &reciprocal{}
}

func (r *reciprocal) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}

	n.gorgoniaNode, err = gorgonia.Inverse(children[0].gorgoniaNode)
	return err
}

func (r *reciprocal) init(o onnx.Operation) error {
	return nil
}
