package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less

func init() {
	register("Less", newLess)
}

type less struct{}

func newLess() operator {
	return &less{}
}

func (l *less) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
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
	n.gorgoniaNode, err = gorgonia.Lt(a, b, false)
	return err
}

func (l *less) init(o onnx.Operation) error {
	return nil
}
