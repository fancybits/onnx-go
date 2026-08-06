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

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not

func init() {
	register("Not", newNot)
}

type notOp struct{}

func (n *notOp) Arity() int { return 1 }

func (n *notOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (n *notOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("not: infershape failed, nil shape")
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (n *notOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("not: expected 1 input, got %d", len(inputs))
	}

	t, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("not: only dense tensors are supported")
	}

	// Note: ONNX spec says Not only supports bool, but real models use float/int
	// where 0 = false (becomes true) and non-zero = true (becomes false)
	switch t.Dtype() {
	case tensor.Bool:
		return notResult(t, func(v bool) bool { return !v })
	case tensor.Int8:
		return notResult(t, func(v int8) bool { return v == 0 })
	case tensor.Int32:
		return notResult(t, func(v int32) bool { return v == 0 })
	case tensor.Int64:
		return notResult(t, func(v int64) bool { return v == 0 })
	case tensor.Float32:
		return notResult(t, func(v float32) bool { return v == 0 })
	case tensor.Float64:
		return notResult(t, func(v float64) bool { return v == 0 })
	default:
		return nil, fmt.Errorf("not: unsupported dtype %v", t.Dtype())
	}
}

// numericData extracts the T-typed backing of a dense tensor, handling
// both 0-d scalar tensors (Data() returns a bare T) and n-d tensors
// (Data() returns a []T) - mirroring andBoolData in and.go for numeric
// types.
func numericData[T any](t *tensor.Dense) ([]T, bool, error) {
	switch d := t.Data().(type) {
	case T:
		return []T{d}, true, nil
	case []T:
		return d, false, nil
	default:
		return nil, false, fmt.Errorf("expected %T data, got %T", *new(T), d)
	}
}

// notResult extracts T-typed data from t - handling both 0-d scalar tensors
// (Data() returns a bare T) and n-d tensors (Data() returns a []T), via
// numericData above - applies isFalsy elementwise, and preserves a
// 0-d output shape when the input was a scalar (mirroring andOp).
func notResult[T any](t *tensor.Dense, isFalsy func(T) bool) (gorgonia.Value, error) {
	data, isScalar, err := numericData[T](t)
	if err != nil {
		return nil, fmt.Errorf("not: %v", err)
	}
	result := make([]bool, len(data))
	for i, v := range data {
		result[i] = isFalsy(v)
	}
	if isScalar {
		return tensor.New(tensor.FromScalar(result[0])), nil
	}
	return tensor.New(tensor.WithShape(t.Shape().Clone()...), tensor.WithBacking(result)), nil
}

func (n *notOp) ReturnsPtr() bool     { return false }
func (n *notOp) CallsExtern() bool    { return false }
func (n *notOp) OverwritesInput() int { return -1 }

func (n *notOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("not"))
}

func (n *notOp) Hashcode() uint32 {
	h := fnv.New32a()
	n.WriteHash(h)
	return h.Sum32()
}

func (n *notOp) String() string { return "Not" }

type not struct{}

func newNot() operator {
	return &not{}
}

func (n *not) apply(g *Graph, ns ...*Node) error {
	node := ns[0]
	children := getOrderedChildren(g.g, node)
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}

	op := &notOp{}
	node.gorgoniaNode, err = gorgonia.ApplyOp(op, children[0].gorgoniaNode)
	return err
}

func (n *not) init(o onnx.Operation) error {
	return nil
}
