package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceMax computes max(x) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax

func init() {
	register("ReduceMax", newReducer("ReduceMax", nil, gorgonia.Max, nil))
}
