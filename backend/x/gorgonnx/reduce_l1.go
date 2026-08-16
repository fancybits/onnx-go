package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceL1 computes sum(|x|) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1

func init() {
	register("ReduceL1", newReducer("ReduceL1", gorgonia.Abs, gorgonia.Sum, nil))
}
