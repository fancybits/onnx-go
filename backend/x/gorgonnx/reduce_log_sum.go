package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceLogSum computes log(sum(x)) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum

func init() {
	register("ReduceLogSum", newReducer("ReduceLogSum", nil, gorgonia.Sum, gorgonia.Log))
}
