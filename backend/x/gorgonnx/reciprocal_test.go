package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestReciprocal(t *testing.T) {
	inputT := tensor.New(
		tensor.WithShape(4),
		tensor.WithBacking([]float32{1, 2, 4, 8}),
	)
	expectedOutput := []float32{1, 0.5, 0.25, 0.125}

	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)

	err := g.ApplyOperation(onnx.Operation{
		Name:       "Reciprocal",
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
	assert.InDeltaSlice(t, expectedOutput, outputT.Data(), 1e-6, "the two tensors should be equal.")
}
