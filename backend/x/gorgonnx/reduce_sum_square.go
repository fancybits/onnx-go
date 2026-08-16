package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceSumSquare computes sum(x²) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare

func init() {
	register("ReduceSumSquare", newReducer("ReduceSumSquare", gorgonia.Square, gorgonia.Sum, nil))
}
