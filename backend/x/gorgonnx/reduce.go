package gorgonnx

import (
	"fmt"
	"sort"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// Shared machinery for the ONNX reduction family. Every member selects axes
// the same way and honours keepdims the same way, differing only in what it
// does to each element before reducing, which reduction it applies, and what
// it does to the result. Each operator registers itself from its own file.

// unaryFn is an elementwise transform applied before or after the reduction.
type unaryFn func(*gorgonia.Node) (*gorgonia.Node, error)

// reduceFn collapses a node along one axis.
type reduceFn func(*gorgonia.Node, ...int) (*gorgonia.Node, error)

type reducer struct {
	// name is the ONNX operator, used only for error messages.
	name string
	// pre and post are optional; nil means identity.
	pre    unaryFn
	reduce reduceFn
	post   unaryFn

	axes     []int
	keepdims bool
	// noopWithEmptyAxes makes an empty axis selection an identity rather than
	// "reduce everything" (ReduceSum-13 and Reduce*-18).
	noopWithEmptyAxes bool
}

// newReducer builds the constructor `register` wants.
func newReducer(name string, pre unaryFn, reduce reduceFn, post unaryFn) func() operator {
	return func() operator {
		return &reducer{
			name:     name,
			pre:      pre,
			reduce:   reduce,
			post:     post,
			keepdims: true, // ONNX default
		}
	}
}

func (r *reducer) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)

	// One input in the older opset, where axes is an attribute; two in the
	// newer one, where it is a tensor.
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

	// An empty selection with noop_with_empty_axes set means "do nothing".
	if len(axes) == 0 {
		n.gorgoniaNode = input

		return nil
	}

	result := input
	if r.pre != nil {
		if result, err = r.pre(result); err != nil {
			return err
		}
	}

	// Descending, so reducing one axis does not shift the index of the next.
	for _, axis := range axes {
		if result, err = r.reduce(result, axis); err != nil {
			return err
		}
	}

	if r.post != nil {
		if result, err = r.post(result); err != nil {
			return err
		}
	}

	if r.keepdims {
		if result, err = gorgonia.Reshape(result, keptShape(inputShape, axes)); err != nil {
			return err
		}
	}

	n.gorgoniaNode = result

	return nil
}

// keptShape is the input shape with every reduced axis set to 1, which is what
// keepdims asks for.
func keptShape(inputShape []int, axes []int) []int {
	out := make([]int, len(inputShape))
	copy(out, inputShape)
	for _, axis := range axes {
		out[axis] = 1
	}

	return out
}

// resolveAxes returns the axes to reduce, normalized to non-negative and
// sorted descending. An empty selection means every axis, which is what ONNX
// specifies when the attribute is absent.
func (r *reducer) resolveAxes(children []*Node, rank int) ([]int, error) {
	axes := r.axes

	if len(children) == 2 {
		// Newer opset: axes arrives as a tensor. It decides which dimensions
		// collapse and so fixes the output shape, which means it has to be
		// readable at build time whatever its provenance.
		axesTensor := staticTensorFromNode(children[1])
		if axesTensor == nil {
			// The selection decides the output shape, so it has to be known
			// when the graph is built. Falling back to "every axis" here would
			// quietly turn a partial reduction into a full collapse.
			return nil, &onnx.ErrNotImplemented{
				Operator: r.name,
				Message:  "axes input is not resolvable at graph build time",
			}
		}
		raw := tensorToInt64Slice(axesTensor)
		axes = make([]int, len(raw))
		for i, v := range raw {
			axes[i] = int(v)
		}
	}

	if len(axes) == 0 {
		if r.noopWithEmptyAxes {
			return nil, nil
		}
		axes = make([]int, rank)
		for i := range axes {
			axes[i] = i
		}
	}

	normalized := make([]int, len(axes))
	for i, axis := range axes {
		if axis < 0 {
			axis += rank
		}
		if axis < 0 || axis >= rank {
			// Deliberately not ErrNotImplemented: the loader and the test
			// harness both read that as "backend does not support this
			// operator" and skip, which would bury a malformed model.
			return nil, fmt.Errorf("%s: axis %d out of range for rank %d", r.name, axes[i], rank)
		}
		normalized[i] = axis
	}
	sort.Sort(sort.Reverse(sort.IntSlice(normalized)))

	return normalized, nil
}

func (r *reducer) init(o onnx.Operation) error {
	if axes, ok := o.Attributes["axes"]; ok {
		axesSlice, ok := axes.([]int64)
		if !ok {
			return &onnx.ErrNotImplemented{
				Operator:       r.name,
				AttributeName:  "axes",
				AttributeValue: axes,
				Message:        "expected a list of ints",
			}
		}
		r.axes = make([]int, len(axesSlice))
		for i, v := range axesSlice {
			r.axes[i] = int(v)
		}
	}

	if noop, ok := o.Attributes["noop_with_empty_axes"]; ok {
		v, ok := noop.(int64)
		if !ok {
			return &onnx.ErrNotImplemented{
				Operator:       r.name,
				AttributeName:  "noop_with_empty_axes",
				AttributeValue: noop,
				Message:        "expected an int",
			}
		}
		r.noopWithEmptyAxes = v != 0
	}

	if keepdims, ok := o.Attributes["keepdims"]; ok {
		kd, ok := keepdims.(int64)
		if !ok {
			return &onnx.ErrNotImplemented{
				Operator:       r.name,
				AttributeName:  "keepdims",
				AttributeValue: keepdims,
				Message:        "expected an int",
			}
		}
		r.keepdims = kd != 0
	}

	return nil
}
