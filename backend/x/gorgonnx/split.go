package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
//
// Equal-split form only (opset 13+): axis attribute, N outputs, no
// split-sizes input, no pre-opset-13 split attribute. Any other form fails
// loudly at graph-build time.

func init() {
	register("Split", newSplit)
}

type split struct {
	axis int
}

func newSplit() operator {
	return &split{}
}

func (s *split) init(o onnx.Operation) error {
	s.axis = 0
	if v, ok := o.Attributes["axis"]; ok {
		i, ok := v.(int64)
		if !ok {
			return fmt.Errorf("split: axis is not an int64")
		}
		s.axis = int(i)
	}
	if _, ok := o.Attributes["split"]; ok {
		return &onnx.ErrNotImplemented{
			Operator:      "Split",
			AttributeName: "split",
			Message:       "split: pre-opset-13 variable-size split (split attribute) is not supported, only equal-split is",
		}
	}
	return nil
}

func (s *split) apply(g *Graph, ns ...*Node) error {
	// All outputs share an identical edge to the single data input, so any
	// of the output nodes yields the same children slice.
	children := getOrderedChildren(g.g, ns[0])
	if len(children) != 1 {
		return &onnx.ErrNotImplemented{
			Operator: "Split",
			Message:  fmt.Sprintf("split: variable-size split (split-sizes input) is not supported, only equal-split is (want 1 input, have %d)", len(children)),
		}
	}
	if err := checkForNil(children); err != nil {
		return err
	}
	data := children[0]

	var shape tensor.Shape
	if t := constTensorFromNode(data); t != nil {
		shape = t.Shape()
	} else {
		shape = data.gorgoniaNode.Shape()
	}
	axis := s.axis
	if axis < 0 {
		axis += len(shape)
	}
	if axis < 0 || axis >= len(shape) {
		return fmt.Errorf("split: axis %d out of range for rank %d", s.axis, len(shape))
	}
	n := len(ns)
	if shape[axis]%n != 0 {
		return fmt.Errorf("split: dimension %d (size %d) not divisible by %d outputs", axis, shape[axis], n)
	}
	size := shape[axis] / n

	// Fold only a genuine constant. A node carrying a tensor is not evidence
	// of one: a graph input carries a placeholder, which SetInput may already
	// have filled with this run's data before the graph is built, and folding
	// that bakes it in for every later run. Anything else goes down the
	// symbolic path, which computes the same result at run time.
	var dataTensor tensor.Tensor
	if isConstNode(data) {
		dataTensor = data.t
	}
	for i, out := range ns {
		starts := []int64{int64(i * size)}
		ends := []int64{int64((i + 1) * size)}
		axes := []int64{int64(axis)}
		steps := []int64{1}
		if dataTensor != nil {
			dataD, ok := dataTensor.(*tensor.Dense)
			if !ok {
				return fmt.Errorf("split: data must be a dense tensor")
			}
			result, err := doSlice(dataD, starts, ends, axes, steps)
			if err != nil {
				return fmt.Errorf("split constant: %w", err)
			}
			out.t = result
			out.MarkConst()
			out.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("split_const")))
		} else {
			op := &sliceOp{starts: starts, ends: ends, axes: axes, steps: steps}
			var err error
			out.gorgoniaNode, err = gorgonia.ApplyOp(op, data.gorgoniaNode)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
