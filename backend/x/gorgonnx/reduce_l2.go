package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceL2 computes sqrt(sum(x²)) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2

func init() {
	register("ReduceL2", newReducer("ReduceL2", gorgonia.Square, gorgonia.Sum, gorgonia.Sqrt))
}
