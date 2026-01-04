package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type reshape struct {
	toShape tensor.Shape
}

func init() {
	register("Reshape", newReshape)
}

func newReshape() operator {
	return &reshape{}
}

func (a *reshape) inferShape(requiredShape interface{}, targetShape tensor.Shape) error {
	var toShape tensor.Shape
	data := requiredShape
	if to, ok := data.(int64); ok {
		data = []int64{to}
	}
	if to, ok := data.([]int64); ok {
		// Calculate total size of input tensor
		dimSize := targetShape.TotalSize()

		// Build output shape, handling 0 and -1 special values
		toShape = make([]int, len(to))
		inferIdx := -1 // index of the -1 dimension to infer
		knownProduct := 1

		for i := 0; i < len(to); i++ {
			toShape[i] = int(to[i])
			if toShape[i] == 0 {
				// 0 means copy from input shape at same index
				if i < len(targetShape) {
					toShape[i] = targetShape[i]
				} else {
					toShape[i] = 1
				}
			}
			if toShape[i] == -1 {
				inferIdx = i
			} else {
				knownProduct *= toShape[i]
			}
		}

		// Infer the -1 dimension
		if inferIdx >= 0 {
			if knownProduct == 0 {
				return fmt.Errorf("Cannot reshape: known product is 0")
			}
			toShape[inferIdx] = dimSize / knownProduct
		}
	} else {
		return fmt.Errorf("Cannot reshape, bad output shape %#v", requiredShape)
	}
	a.toShape = toShape
	return nil
}

func (a *reshape) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	// Get the target shape from child node - try gorgoniaNode.Value() first, fallback to t
	var shapeData interface{}
	if children[1].gorgoniaNode != nil && children[1].gorgoniaNode.Value() != nil {
		shapeData = children[1].gorgoniaNode.Value().Data()
	} else if children[1].t != nil {
		shapeData = children[1].t.Data()
	} else {
		return fmt.Errorf("reshape: shape input has no value")
	}

	err = a.inferShape(shapeData, children[0].gorgoniaNode.Shape())
	if err != nil {
		return err
	}

	n.gorgoniaNode, err = gorgonia.Reshape(children[0].gorgoniaNode, a.toShape)

	return err
}

func (a *reshape) init(o onnx.Operation) error {
	return nil
}
