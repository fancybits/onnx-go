package gorgonnx

import (
	"gorgonia.org/gorgonia"
)

// ReduceMean computes mean(x) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean

func init() {
	register("ReduceMean", newReducer("ReduceMean", nil, gorgonia.Mean, nil))
}
