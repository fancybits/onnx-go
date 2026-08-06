package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// splitHarness builds: input(tensor) -> Split -> N outputs. Returns output tensors.
func splitHarness(t *testing.T, in tensor.Tensor, axis int64, numOutputs int) ([]tensor.Tensor, error) {
	t.Helper()
	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	if err := input.(*Node).SetTensor(in); err != nil {
		t.Fatal(err)
	}
	outs := make([]graph.Node, numOutputs)
	for i := range outs {
		o := g.NewNode()
		g.AddNode(o)
		g.SetWeightedEdge(g.NewWeightedEdge(o, input, 0))
		outs[i] = o
	}
	err := g.ApplyOperation(
		onnx.Operation{Name: "Split", Attributes: map[string]interface{}{"axis": axis}},
		outs...,
	)
	if err != nil {
		return nil, err
	}
	if err := g.Run(); err != nil {
		return nil, err
	}
	result := make([]tensor.Tensor, numOutputs)
	for i, n := range outs {
		result[i] = n.(*Node).GetTensor()
	}
	return result, nil
}

func TestSplitAxisNeg1(t *testing.T) {
	// [2,6] split into 3 along last axis -> 3 x [2,2]
	in := tensor.New(tensor.WithShape(2, 6), tensor.WithBacking([]float32{
		0, 1, 2, 3, 4, 5,
		6, 7, 8, 9, 10, 11,
	}))
	outs, err := splitHarness(t, in, -1, 3)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, []int{2, 2}, []int(outs[0].Shape()))
	assert.Equal(t, []float32{0, 1, 6, 7}, outs[0].Data())
	assert.Equal(t, []float32{2, 3, 8, 9}, outs[1].Data())
	assert.Equal(t, []float32{4, 5, 10, 11}, outs[2].Data())
}

func TestSplitSizeOneAxis0(t *testing.T) {
	// [2,3] split into 2 along axis 0 -> 2 x [1,3]; exercises the size-1 dim
	// case that gorgonia.Slice would squeeze (the artifact splits [2,192] this way)
	in := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}))
	outs, err := splitHarness(t, in, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, []int{1, 3}, []int(outs[0].Shape()))
	assert.Equal(t, []float32{1, 2, 3}, outs[0].Data())
	assert.Equal(t, []float32{4, 5, 6}, outs[1].Data())
}

func TestSplitNonDivisible(t *testing.T) {
	in := tensor.New(tensor.WithShape(2, 4), tensor.WithBacking(make([]float32, 8)))
	_, err := splitHarness(t, in, -1, 3)
	assert.Error(t, err)
}

// TestSplitDynamicInput exercises Split fed by an Add node (not a constant),
// so the operator cannot take the constant fast path and must go through the
// custom sliceOp. This mirrors how Split is used on real dynamic graphs.
func TestSplitDynamicInput(t *testing.T) {
	x := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1}))
	y := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{0, 1, 2, 3, 4, 5}))

	g := NewGraph()
	xNode := g.NewNode()
	g.AddNode(xNode)
	if err := xNode.(*Node).SetTensor(x); err != nil {
		t.Fatal(err)
	}
	yNode := g.NewNode()
	g.AddNode(yNode)
	if err := yNode.(*Node).SetTensor(y); err != nil {
		t.Fatal(err)
	}

	addOut := g.NewNode()
	g.AddNode(addOut)
	g.SetWeightedEdge(g.NewWeightedEdge(addOut, xNode, 0))
	g.SetWeightedEdge(g.NewWeightedEdge(addOut, yNode, 1))

	numOutputs := 2
	outs := make([]graph.Node, numOutputs)
	for i := range outs {
		o := g.NewNode()
		g.AddNode(o)
		g.SetWeightedEdge(g.NewWeightedEdge(o, addOut, 0))
		outs[i] = o
	}

	if err := g.ApplyOperation(onnx.Operation{Name: "Add"}, addOut); err != nil {
		t.Fatal(err)
	}
	if err := g.ApplyOperation(
		onnx.Operation{Name: "Split", Attributes: map[string]interface{}{"axis": int64(0)}},
		outs...,
	); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}

	// Equivalent constant split, computed independently, for comparison:
	// x+y = [1,2,3, 4,5,6] split on axis 0 into 2 x [1,3]
	assert.Equal(t, []int{1, 3}, []int(outs[0].(*Node).GetTensor().Shape()))
	assert.Equal(t, []float32{1, 2, 3}, outs[0].(*Node).GetTensor().Data())
	assert.Equal(t, []float32{4, 5, 6}, outs[1].(*Node).GetTensor().Data())
}
