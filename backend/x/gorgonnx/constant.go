package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type constant struct {
	value *gorgonia.Node
	t     tensor.Tensor // store the tensor for immediate access
}

func init() {
	register("Constant", newConstant)
}

func newConstant() operator {
	return &constant{}
}

func (a *constant) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	if a.value == nil {
		return fmt.Errorf("constant: value is nil (no value attribute found)")
	}
	// Set both gorgoniaNode and t for immediate access in shape computations
	n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, a.t, gorgonia.WithName(getUniqNodeName("constant")))
	n.t = a.t
	// The value comes from the node's own attribute: a compile-time constant
	// on the same footing as an initializer.
	n.MarkConst()
	return nil
}

func (a *constant) init(o onnx.Operation) error {
	val, ok := o.Attributes["value"]
	if !ok {
		return fmt.Errorf("constant: missing value attribute")
	}
	// Store the tensor for immediate access
	if t, ok := val.(tensor.Tensor); ok {
		a.t = t
		a.value = gorgonia.NewConstant(t)
	} else {
		return fmt.Errorf("constant: value attribute is not a tensor, got %T", val)
	}
	return nil
}
