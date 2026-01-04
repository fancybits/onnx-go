package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater

func init() {
	register("Greater", newGreater)
}

type greater struct{}

func newGreater() operator {
	return &greater{}
}

func (g *greater) apply(gg *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(gg.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	// Handle broadcasting if needed
	a, b, err := broadcast(children[0], children[1])
	if err != nil {
		return err
	}

	// retSame=false means the result is a boolean tensor
	n.gorgoniaNode, err = gorgonia.Gt(a, b, false)
	return err
}

func (g *greater) init(o onnx.Operation) error {
	return nil
}
