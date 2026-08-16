package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceSum computes sum(x) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum

func init() {
	register("ReduceSum", newReducer("ReduceSum", nil, gorgonia.Sum, nil))
}
