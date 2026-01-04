package gorgonnx

import (
	"sort"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum

func init() {
	register("ReduceSum", newReduceSum)
}

type reduceSum struct {
	axes     []int
	keepdims bool
}

func newReduceSum() operator {
	return &reduceSum{
		keepdims: true, // default
	}
}

func (r *reduceSum) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)

	// ReduceSum can have 1 input (old opset, axes as attribute) or 2 inputs (new opset, axes as tensor)
	if len(children) < 1 || len(children) > 2 {
		return &onnx.ErrNotImplemented{
			Operator: "ReduceSum",
			Message:  "expected 1 or 2 inputs",
		}
	}

	input := children[0].gorgoniaNode
	inputShape := input.Shape()

	// Get axes - either from attribute or from second input tensor
	axes := r.axes
	if len(children) == 2 && children[1].gorgoniaNode != nil {
		// New opset: axes comes from second input tensor
		axesTensor := children[1].gorgoniaNode.Value()
		if axesTensor != nil {
			axes = nil // Clear attribute axes
			axesInt64 := tensorToInt64Slice(axesTensor.(*tensor.Dense))
			axes = make([]int, len(axesInt64))
			for i, v := range axesInt64 {
				axes[i] = int(v)
			}
		}
	}
	if len(axes) == 0 {
		axes = make([]int, len(inputShape))
		for i := range axes {
			axes[i] = i
		}
	}

	// Normalize negative axes
	for i := range axes {
		if axes[i] < 0 {
			axes[i] = len(inputShape) + axes[i]
		}
	}

	// Sort axes in descending order so we reduce from highest to lowest
	// This prevents index shifting issues
	sort.Sort(sort.Reverse(sort.IntSlice(axes)))

	var err error
	result := input
	for _, axis := range axes {
		result, err = gorgonia.Sum(result, axis)
		if err != nil {
			return err
		}
	}

	// If keepdims is true, we need to insert 1-sized dimensions back
	if r.keepdims {
		// Build output shape with 1s in reduced positions
		outputShape := make([]int, len(inputShape))
		for i := range outputShape {
			outputShape[i] = inputShape[i]
		}
		for _, axis := range axes {
			normalizedAxis := axis
			if normalizedAxis < 0 {
				normalizedAxis = len(inputShape) + normalizedAxis
			}
			outputShape[normalizedAxis] = 1
		}
		if len(axes) == 0 {
			// All axes reduced (shouldn't happen since we fill them above)
			for i := range outputShape {
				outputShape[i] = 1
			}
		}
		result, err = gorgonia.Reshape(result, outputShape)
		if err != nil {
			return err
		}
	}

	n.gorgoniaNode = result
	return nil
}

func (r *reduceSum) init(o onnx.Operation) error {
	// Parse axes attribute
	if axes, ok := o.Attributes["axes"]; ok {
		if axesSlice, ok := axes.([]int64); ok {
			r.axes = make([]int, len(axesSlice))
			for i, v := range axesSlice {
				r.axes[i] = int(v)
			}
		}
	}

	// Parse keepdims attribute (default is 1/true)
	if keepdims, ok := o.Attributes["keepdims"]; ok {
		if kd, ok := keepdims.(int64); ok {
			r.keepdims = kd != 0
		}
	}

	return nil
}
