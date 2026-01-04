package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape

func init() {
	register("ConstantOfShape", newConstantOfShape)
}

func newConstantOfShape() operator {
	return &constantOfShape{}
}

type constantOfShape struct {
	value tensor.Tensor // The value to fill the output tensor with (default: 0.0 float32)
}

func (c *constantOfShape) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	if err := checkCondition(children, 1); err != nil {
		return err
	}

	// Get shape from input tensor
	shapeNode := children[0]
	shapeTensor := getTensorFromNode(shapeNode)
	if shapeTensor == nil {
		return fmt.Errorf("constantofshape: shape input must be a constant tensor")
	}

	// Convert shape tensor to []int
	outputShape := tensorToIntSlice(shapeTensor)

	// Create output tensor filled with the constant value
	var result *tensor.Dense
	if c.value == nil {
		// Default: float32 zeros
		result = tensor.New(tensor.WithShape(outputShape...), tensor.Of(tensor.Float32))
		result.Zero()
	} else {
		// Use the specified value
		dtype := c.value.Dtype()
		result = tensor.New(tensor.WithShape(outputShape...), tensor.Of(dtype))

		// Fill with the constant value
		totalSize := result.Shape().TotalSize()
		switch dtype {
		case tensor.Float32:
			val := c.value.Data().([]float32)[0]
			data := make([]float32, totalSize)
			for i := range data {
				data[i] = val
			}
			result = tensor.New(tensor.WithShape(outputShape...), tensor.WithBacking(data))
		case tensor.Float64:
			val := c.value.Data().([]float64)[0]
			data := make([]float64, totalSize)
			for i := range data {
				data[i] = val
			}
			result = tensor.New(tensor.WithShape(outputShape...), tensor.WithBacking(data))
		case tensor.Int32:
			val := c.value.Data().([]int32)[0]
			data := make([]int32, totalSize)
			for i := range data {
				data[i] = val
			}
			result = tensor.New(tensor.WithShape(outputShape...), tensor.WithBacking(data))
		case tensor.Int64:
			val := c.value.Data().([]int64)[0]
			data := make([]int64, totalSize)
			for i := range data {
				data[i] = val
			}
			result = tensor.New(tensor.WithShape(outputShape...), tensor.WithBacking(data))
		case tensor.Int:
			val := c.value.Data().([]int)[0]
			data := make([]int, totalSize)
			for i := range data {
				data[i] = val
			}
			result = tensor.New(tensor.WithShape(outputShape...), tensor.WithBacking(data))
		case tensor.Bool:
			val := c.value.Data().([]bool)[0]
			data := make([]bool, totalSize)
			for i := range data {
				data[i] = val
			}
			result = tensor.New(tensor.WithShape(outputShape...), tensor.WithBacking(data))
		default:
			return fmt.Errorf("constantofshape: unsupported dtype %v", dtype)
		}
	}

	n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("constantofshape")))
	return nil
}

func (c *constantOfShape) init(o onnx.Operation) error {
	if value, ok := o.Attributes["value"]; ok {
		if t, ok := value.(tensor.Tensor); ok {
			c.value = t
		} else {
			return fmt.Errorf("constantofshape: value attribute is not a tensor")
		}
	}
	// If no value attribute, c.value remains nil and we'll use float32 zeros
	return nil
}
