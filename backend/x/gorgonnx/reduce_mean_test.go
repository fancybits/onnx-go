package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReduceMean_Keepdims(t *testing.T) {
	// Input: 2x3 tensor
	inputT := tensor.New(
		tensor.WithShape(2, 3),
		tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}),
	)
	// Mean along axis 1 with keepdims: [[2], [5]]
	expectedOutput := []float32{2, 5}

	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)

	err := g.ApplyOperation(onnx.Operation{
		Name: "ReduceMean",
		Attributes: map[string]interface{}{
			"axes":     []int64{1},
			"keepdims": int64(1),
		},
	}, output)
	if err != nil {
		t.Fatal(err)
	}

	err = g.Run()
	if err != nil {
		t.Fatal(err)
	}

	outputT := output.(*Node).GetTensor()
	assert.Equal(t, tensor.Shape{2, 1}, outputT.Shape())
	assert.InDeltaSlice(t, expectedOutput, outputT.Data(), 1e-6)
}

func TestReduceMean_NoKeepdims(t *testing.T) {
	// Input: 2x3 tensor
	inputT := tensor.New(
		tensor.WithShape(2, 3),
		tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}),
	)
	// Mean along axis 1 without keepdims: [2, 5]
	expectedOutput := []float32{2, 5}

	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)

	err := g.ApplyOperation(onnx.Operation{
		Name: "ReduceMean",
		Attributes: map[string]interface{}{
			"axes":     []int64{1},
			"keepdims": int64(0),
		},
	}, output)
	if err != nil {
		t.Fatal(err)
	}

	err = g.Run()
	if err != nil {
		t.Fatal(err)
	}

	outputT := output.(*Node).GetTensor()
	assert.Equal(t, tensor.Shape{2}, outputT.Shape())
	assert.InDeltaSlice(t, expectedOutput, outputT.Data(), 1e-6)
}
