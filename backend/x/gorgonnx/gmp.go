package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool

func init() {
	register("GlobalMaxPool", newGMP)
}

func newGMP() operator {
	return &gmp{}
}

type gmp struct{}

func (g *gmp) apply(gg *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(gg.g, n)
	if err := checkCondition(children, 1); err != nil {
		return err
	}

	input := children[0].gorgoniaNode
	shape := input.Shape()

	// GlobalMaxPool: max over all spatial dimensions (all dims except batch and channel)
	// Input shape: (N, C, D1, D2, ..., Dn)
	// Output shape: (N, C, 1, 1, ..., 1)
	if len(shape) < 3 {
		return fmt.Errorf("GlobalMaxPool: input must have at least 3 dimensions (N, C, spatial...)")
	}

	// Reduce over spatial dimensions (from dim 2 onwards)
	spatialAxes := make([]int, len(shape)-2)
	for i := range spatialAxes {
		spatialAxes[i] = i + 2
	}

	var err error
	result := input
	// Apply max reduction over each spatial axis, in reverse order to preserve indices
	for i := len(spatialAxes) - 1; i >= 0; i-- {
		result, err = gorgonia.Max(result, spatialAxes[i])
		if err != nil {
			return err
		}
	}

	// Reshape to add back the 1-sized spatial dimensions
	outputShape := make([]int, len(shape))
	copy(outputShape, shape)
	for i := 2; i < len(outputShape); i++ {
		outputShape[i] = 1
	}

	n.gorgoniaNode, err = gorgonia.Reshape(result, outputShape)
	return err
}

func (*gmp) init(o onnx.Operation) error {
	return nil
}
