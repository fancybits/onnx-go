package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

func init() {
	register("GlobalAveragePool", newGAP)
}

func newGAP() operator {
	return &gap{}
}

type gap struct{}

func (g *gap) apply(gg *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(gg.g, n)
	if len(children) != 1 {
		return errors.New("GlobalAveragePool: bad arity")
	}

	input := children[0].gorgoniaNode
	dims := input.Dims()
	inputShape := input.Shape()

	// GlobalAveragePool pools over all spatial dimensions
	// For 4D input (N, C, H, W): pool over H, W (axes 2, 3) -> output (N, C, 1, 1)
	// For 3D input (N, C, L): pool over L (axis 2) -> output (N, C, 1)
	// For 2D input (N, C): pool over C (axis 1) -> output (N, 1)

	switch dims {
	case 4:
		// Use Gorgonia's built-in for 4D
		var err error
		n.gorgoniaNode, err = gorgonia.GlobalAveragePool2D(input)
		return err
	case 3:
		// For 3D, compute mean over the last axis (axis 2)
		// Mean reduces the dimension, so we need to reshape to add it back
		meanNode, err := gorgonia.Mean(input, 2)
		if err != nil {
			return err
		}
		// Reshape from (N, C) to (N, C, 1)
		newShape := []int{inputShape[0], inputShape[1], 1}
		n.gorgoniaNode, err = gorgonia.Reshape(meanNode, newShape)
		return err
	case 2:
		// For 2D, compute mean over the last axis (axis 1)
		meanNode, err := gorgonia.Mean(input, 1)
		if err != nil {
			return err
		}
		// Reshape from (N,) to (N, 1)
		newShape := []int{inputShape[0], 1}
		n.gorgoniaNode, err = gorgonia.Reshape(meanNode, newShape)
		return err
	default:
		return fmt.Errorf("GlobalAveragePool: unsupported input dimensions %d", dims)
	}
}

func (*gap) init(onnx.Operation) error {
	return nil
}
