package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Specifications: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze

type squeeze struct {
	Axes []int64
}

func init() {
	register("Squeeze", newSqueeze)
}
func newSqueeze() operator {
	return &squeeze{}
}

func (a *squeeze) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)

	// ONNX opset 13+ has axes as a second input rather than attribute
	if len(children) == 2 {
		// Get axes from second input
		// axes determines the output shape, so it has to be read at build time
		// whatever its provenance
		axesNode := children[1]
		var axesTensor interface{}
		if t := staticTensorFromNode(axesNode); t != nil {
			axesTensor = t.Data()
		}
		if axesTensor != nil {
			switch axes := axesTensor.(type) {
			case []int64:
				a.Axes = axes
			case int64:
				a.Axes = []int64{axes}
			case []int32:
				a.Axes = make([]int64, len(axes))
				for i, v := range axes {
					a.Axes[i] = int64(v)
				}
			case int32:
				a.Axes = []int64{int64(axes)}
			}
		}
	} else if len(children) != 1 {
		return fmt.Errorf("squeeze: expected 1 or 2 children, got %d", len(children))
	}

	tensor := children[0].gorgoniaNode
	numAxes := len(a.Axes)
	shape := tensor.Shape()
	var dims []int

	if numAxes == 0 {
		// According to the spec, we have to squeeze all axes of single dimensions
		for _, v := range shape {
			if v == 1 {
				numAxes++
			}
		}
		// Scalar, we need to keep at least 1 axe
		if numAxes == tensor.Dims() {
			dims = []int{1}
		} else {
			dims = make([]int, tensor.Dims()-numAxes)
			index := 0
			for _, v := range shape {
				if v != 1 {
					dims[index] = v
					index++
				}
			}
		}
	} else {
		// Axes to squeeze are specified in the Axes parameter
		dims = make([]int, tensor.Dims()-numAxes)
		// Make a mask with the axes to remove
		mask := make([]bool, tensor.Dims())
		for _, v := range a.Axes {
			// Handle negative axes
			if v < 0 {
				v = int64(tensor.Dims()) + v
			}
			if v >= 0 && v < int64(tensor.Dims()) {
				mask[v] = true
			}
		}
		// If an axis is selected with shape entry not equal to one, an error is raised.
		index := 0
		for k, v := range shape {
			if mask[k] {
				if v != 1 {
					return fmt.Errorf("Unable to squeeze an axis whose shape entry is not 1 (got %v instead)", v)
				}
				continue
			}
			dims[index] = v
			index++
		}
	}

	var err error
	n.gorgoniaNode, err = gorgonia.Reshape(tensor, dims)

	return err
}

func (a *squeeze) init(o onnx.Operation) error {
	if o.Attributes == nil {
		// Use the default Axes attribute
		a.Axes = []int64{}
		return nil
	}

	axes := o.Attributes["axes"]
	if axes == nil {
		// The Axes attribute is optional
		a.Axes = []int64{}
		return nil
	}

	var ok bool
	a.Axes, ok = axes.([]int64)
	if !ok {
		return errors.New("squeeze: axes in not an []int64")
	}
	return nil
}
