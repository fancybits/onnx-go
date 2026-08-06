package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

// applyUnaryOp runs a single-input operation and returns the computed tensor.
func applyUnaryOp(t *testing.T, opName string, a tensor.Tensor) tensor.Tensor {
	t.Helper()
	g := NewGraph()
	in := g.NewNode()
	g.AddNode(in)
	out := g.NewNode()
	g.AddNode(out)
	g.SetWeightedEdge(g.NewWeightedEdge(out, in, 0))
	if err := in.(*Node).SetTensor(a); err != nil {
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

func TestNot(t *testing.T) {
	a := tensor.New(tensor.WithShape(4), tensor.WithBacking([]bool{true, true, false, false}))
	got := applyUnaryOp(t, "Not", a)
	assert.Equal(t, []bool{false, false, true, true}, got.Data())
}

func TestNotScalar(t *testing.T) {
	// The bigru artifact's loop-cond bookkeeping feeds 0-d bool scalars
	// into Not. A 0-d Dense's Data() returns a bare bool rather than a
	// []bool, which the naive []bool cast would panic/error on.
	trueScalar := tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{true}))
	got := applyUnaryOp(t, "Not", trueScalar)
	assert.Equal(t, false, got.Data())
	assert.Equal(t, 0, got.Shape().Dims())

	falseScalar := tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{false}))
	got = applyUnaryOp(t, "Not", falseScalar)
	assert.Equal(t, true, got.Data())
	assert.Equal(t, 0, got.Shape().Dims())
}
