package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestGlobalMaxPool(t *testing.T) {
	// Input: 1x1x2x2 tensor (batch=1, channel=1, 2x2 spatial)
	inputT := tensor.New(
		tensor.WithShape(1, 1, 2, 2),
		tensor.WithBacking([]float32{1, 2, 3, 4}),
	)
	// Expected: max of [1,2,3,4] = 4, shaped as 1x1x1x1
	expectedOutput := []float32{4}

	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)

	err := g.ApplyOperation(onnx.Operation{
		Name:       "GlobalMaxPool",
		Attributes: nil,
	}, output)
	if err != nil {
		t.Fatal(err)
	}

	err = g.Run()
	if err != nil {
		t.Fatal(err)
	}

	outputT := output.(*Node).GetTensor()
	assert.Equal(t, tensor.Shape{1, 1, 1, 1}, outputT.Shape())
	assert.InDeltaSlice(t, expectedOutput, outputT.Data(), 1e-6)
}

func TestGlobalMaxPool_MultiChannel(t *testing.T) {
	// Input: 1x2x2x2 tensor (batch=1, 2 channels, 2x2 spatial)
	inputT := tensor.New(
		tensor.WithShape(1, 2, 2, 2),
		tensor.WithBacking([]float32{
			// Channel 0: max = 4
			1, 2, 3, 4,
			// Channel 1: max = 8
			5, 6, 7, 8,
		}),
	)
	expectedOutput := []float32{4, 8}

	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)

	err := g.ApplyOperation(onnx.Operation{
		Name:       "GlobalMaxPool",
		Attributes: nil,
	}, output)
	if err != nil {
		t.Fatal(err)
	}

	err = g.Run()
	if err != nil {
		t.Fatal(err)
	}

	outputT := output.(*Node).GetTensor()
	assert.Equal(t, tensor.Shape{1, 2, 1, 1}, outputT.Shape())
	assert.InDeltaSlice(t, expectedOutput, outputT.Data(), 1e-6)
}
