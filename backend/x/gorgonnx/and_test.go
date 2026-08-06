package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func applyBinaryOp(t *testing.T, opName string, a, b tensor.Tensor) tensor.Tensor {
	t.Helper()
	g := NewGraph()
	in1 := g.NewNode()
	g.AddNode(in1)
	in2 := g.NewNode()
	g.AddNode(in2)
	out := g.NewNode()
	g.AddNode(out)
	g.SetWeightedEdge(g.NewWeightedEdge(out, in1, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(out, in2, 1))
	if err := in1.(*Node).SetTensor(a); err != nil {
		t.Fatal(err)
	}
	if err := in2.(*Node).SetTensor(b); err != nil {
		t.Fatal(err)
	}
	if err := g.ApplyOperation(onnx.Operation{Name: opName}, out); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}
	return out.(*Node).GetTensor()
}

func TestAnd(t *testing.T) {
	a := tensor.New(tensor.WithShape(4), tensor.WithBacking([]bool{true, true, false, false}))
	b := tensor.New(tensor.WithShape(4), tensor.WithBacking([]bool{true, false, true, false}))
	got := applyBinaryOp(t, "And", a, b)
	assert.Equal(t, []bool{true, false, false, false}, got.Data())
}

func TestAndScalar(t *testing.T) {
	// The bigru artifact ANDs 0-d bool scalars (loop cond bookkeeping).
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{true}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{true}))
	got := applyBinaryOp(t, "And", a, b)
	assert.Equal(t, true, got.Data())
}
