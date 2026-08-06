package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// SPEC: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze

type unsqueeze struct {
	Axes []int64 // Used for older opsets where axes is an attribute
}

func init() {
	register("Unsqueeze", newUnsqueeze)
}
func newUnsqueeze() operator {
	return &unsqueeze{}
}

func (a *unsqueeze) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)

	// Unsqueeze can have 1 input (old opset, axes as attribute) or 2 inputs (new opset, axes as input)
	if len(children) < 1 || len(children) > 2 {
		return fmt.Errorf("unsqueeze: expected 1 or 2 inputs, got %d", len(children))
	}

	// Get the data tensor
	if children[0].gorgoniaNode == nil {
		return fmt.Errorf("unsqueeze: data input is nil")
	}
	tensor := children[0].gorgoniaNode

	// Get axes - either from second input (new opset) or attribute (old opset)
	// axes determines the output shape, so it has to be read at build time
	// whatever its provenance
	var axes []int64
	if len(children) == 2 {
		if axesTensor := staticTensorFromNode(children[1]); axesTensor != nil {
			axes = tensorToInt64Slice(axesTensor)
		}
	}
	if len(axes) == 0 {
		if len(a.Axes) > 0 {
			// Old opset: axes is an attribute
			axes = a.Axes
		} else {
			return fmt.Errorf("unsqueeze: axes not provided as attribute or input")
		}
	}

	// Handle negative axes
	inputDims := tensor.Dims()
	outputDims := inputDims + len(axes)
	for i := range axes {
		if axes[i] < 0 {
			axes[i] = int64(outputDims) + axes[i]
		}
	}

	dims := make([]int, outputDims)
	for k := range dims {
		dims[k] = -1
	}
	for _, v := range axes {
		if v >= 0 && int(v) < len(dims) {
			dims[v] = 1
		}
	}

	// Fill in the remaining dimensions from the input tensor
	var inputIdx int
	for k := range dims {
		if dims[k] == -1 {
			if inputIdx < inputDims {
				dims[k] = tensor.Shape()[inputIdx]
				inputIdx++
			} else {
				dims[k] = 1
			}
		}
	}

	var err error
	n.gorgoniaNode, err = gorgonia.Reshape(tensor, dims)
	return err
}

func (a *unsqueeze) init(o onnx.Operation) error {
	// axes attribute is optional in newer opsets (13+)
	// where it's passed as the second input instead
	axes, ok := o.Attributes["axes"].([]int64)
	if ok {
		a.Axes = axes
	}
	// If axes is not an attribute, it will be read from the second input in apply()
	return nil
}
