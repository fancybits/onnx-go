package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// ReduceLogSumExp computes log(sum(exp(x))) over the selected axes.
// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp

func init() {
	register("ReduceLogSumExp", newLogSumExp)
}

// logSumExp does not go through the shared pre/reduce/post pipeline, because
// evaluating exp(x) directly is not usable: float32 exp overflows above ~88
// and underflows to zero below ~-104, so the naive form returns NaN for the
// log-domain magnitudes this operator exists to handle -- log(sum(exp([-120,
// -121, -122]))) is about -119.6, not NaN.
//
// Shifting by the per-axis maximum first keeps every exponent in range and is
// exact: log(sum(exp(x))) == m + log(sum(exp(x-m))) for any m.
type logSumExp struct {
	reducer
}

func newLogSumExp() operator {
	return &logSumExp{
		reducer: reducer{
			name:     "ReduceLogSumExp",
			keepdims: true,
		},
	}
}

func (r *logSumExp) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)

	if len(children) < 1 || len(children) > 2 {
		return &onnx.ErrNotImplemented{
			Operator: r.name,
			Message:  "expected 1 or 2 inputs",
		}
	}

	input := children[0].gorgoniaNode
	inputShape := input.Shape()

	axes, err := r.resolveAxes(children, len(inputShape))
	if err != nil {
		return err
	}
	if len(axes) == 0 {
		n.gorgoniaNode = input

		return nil
	}

	// The maximum along the reduced axes, in both shapes: collapsed to add
	// back at the end, and with the axes kept so it broadcasts against input.
	maxCollapsed := input
	for _, axis := range axes {
		if maxCollapsed, err = gorgonia.Max(maxCollapsed, axis); err != nil {
			return err
		}
	}
	maxKept, err := gorgonia.Reshape(maxCollapsed, keptShape(inputShape, axes))
	if err != nil {
		return err
	}

	// Materialize the max to the input's full shape rather than broadcasting
	// it. gorgonia's BroadcastPattern is a single byte split into two nibbles,
	// so it addresses at most 4 dimensions per operand (bcAllowableAxes); an
	// axis index of 4 silently sets a bit belonging to the other operand, and
	// 8 or above shifts out entirely. Tiling has no such limit, and the shapes
	// then match exactly, so the subtraction needs no broadcast at all.
	repeats := make([]int64, len(inputShape))
	for i := range repeats {
		repeats[i] = 1
	}
	for _, axis := range axes {
		repeats[axis] = int64(inputShape[axis])
	}
	maxFull, err := gorgonia.ApplyOp(&tileOp{repeats: repeats}, maxKept)
	if err != nil {
		return err
	}

	shifted, err := gorgonia.Sub(input, maxFull)
	if err != nil {
		return err
	}

	result, err := gorgonia.Exp(shifted)
	if err != nil {
		return err
	}
	for _, axis := range axes {
		if result, err = gorgonia.Sum(result, axis); err != nil {
			return err
		}
	}
	if result, err = gorgonia.Log(result); err != nil {
		return err
	}
	if result, err = gorgonia.Add(result, maxCollapsed); err != nil {
		return err
	}

	if r.keepdims {
		if result, err = gorgonia.Reshape(result, keptShape(inputShape, axes)); err != nil {
			return err
		}
	}

	n.gorgoniaNode = result

	return nil
}
