package gorgonnx

import (
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#And

func init() {
	register("And", newAnd)
}

type andOp struct{}

func (a *andOp) Arity() int { return 2 }

func (a *andOp) Type() hm.Type {
	t := hm.TypeVariable('a')
	return hm.NewFnType(t, t, t)
}

func (a *andOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("and: infershape failed, nil shape")
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (a *andOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("and: expected 2 inputs, got %d", len(inputs))
	}
	x, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("and: only dense tensors are supported")
	}
	y, ok := inputs[1].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("and: only dense tensors are supported")
	}
	if x.Dtype() != tensor.Bool || y.Dtype() != tensor.Bool {
		return nil, fmt.Errorf("and: expected bool tensors, got %v and %v", x.Dtype(), y.Dtype())
	}
	if !x.Shape().Eq(y.Shape()) {
		return nil, fmt.Errorf("and: shape mismatch %v vs %v", x.Shape(), y.Shape())
	}

	xd, xIsScalar := andBoolData(x)
	yd, yIsScalar := andBoolData(y)
	if len(xd) != len(yd) {
		return nil, fmt.Errorf("and: data length mismatch %d vs %d", len(xd), len(yd))
	}

	result := make([]bool, len(xd))
	for i := range xd {
		result[i] = xd[i] && yd[i]
	}

	if xIsScalar && yIsScalar {
		return tensor.New(tensor.FromScalar(result[0])), nil
	}
	return tensor.New(tensor.WithShape(x.Shape().Clone()...), tensor.WithBacking(result)), nil
}

// andBoolData extracts the bool backing slice of a dense tensor, handling
// both 0-d scalar tensors (whose Data() returns a bare bool) and n-d
// tensors (whose Data() returns a []bool).
func andBoolData(t *tensor.Dense) ([]bool, bool) {
	switch d := t.Data().(type) {
	case bool:
		return []bool{d}, true
	case []bool:
		return d, false
	default:
		return nil, false
	}
}

func (a *andOp) ReturnsPtr() bool     { return false }
func (a *andOp) CallsExtern() bool    { return false }
func (a *andOp) OverwritesInput() int { return -1 }

func (a *andOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("and"))
}

func (a *andOp) Hashcode() uint32 {
	h := fnv.New32a()
	a.WriteHash(h)
	return h.Sum32()
}

func (a *andOp) String() string { return "And" }

type and struct{}

func newAnd() operator {
	return &and{}
}

func (a *and) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	// broadcast() cannot reliably reshape two 0-d scalar operands, so skip
	// it when the shapes already match (including the 0-d/0-d case).
	var x, y *gorgonia.Node
	if children[0].gorgoniaNode.Shape().Eq(children[1].gorgoniaNode.Shape()) {
		x, y = children[0].gorgoniaNode, children[1].gorgoniaNode
	} else {
		x, y, err = broadcast(children[0], children[1])
		if err != nil {
			return err
		}
	}

	n.gorgoniaNode, err = gorgonia.ApplyOp(&andOp{}, x, y)
	return err
}

func (a *and) init(o onnx.Operation) error {
	return nil
}
