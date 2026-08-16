package gorgonnx

import (
	"math"
	"testing"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/tensor"
)

// applyReduce runs a single reduction operator over one input and returns the
// output node, so behaviour the ONNX fixtures do not exercise can be checked
// directly.
func applyReduce(t *testing.T, op string, attrs map[string]interface{}, in tensor.Tensor) (tensor.Tensor, error) {
	t.Helper()
	g := NewGraph()
	x := g.NewNode()
	g.AddNode(x)
	out := g.NewNode()
	g.AddNode(out)
	g.SetWeightedEdge(g.NewWeightedEdge(out, x, 0))
	if err := x.(*Node).SetTensor(in); err != nil {
		t.Fatal(err)
	}
	if err := g.ApplyOperation(onnx.Operation{Name: op, Attributes: attrs}, out); err != nil {
		return nil, err
	}
	if err := g.Run(); err != nil {
		return nil, err
	}

	return out.(*Node).GetTensor(), nil
}

// TestReduceLogSumExpStability covers the range the conformance fixtures miss.
// They all use small values, where the naive exp/sum/log form happens to work;
// float32 exp overflows above ~88 and underflows below ~-104, which is exactly
// the log-domain territory this operator exists for.
func TestReduceLogSumExpStability(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input []float32
		want  float64
	}{
		{"small values", []float32{1, 2, 3}, 3.4076059},
		{"large positive, exp would overflow", []float32{90, 91, 92}, 92.4076059},
		{"large negative, exp would underflow", []float32{-120, -121, -122}, -119.5923941},
		{"one dominant term", []float32{89, 1, 1}, 89.0},
		{"mixed signs", []float32{-50, 0, 50}, 50.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := applyReduce(t, "ReduceLogSumExp",
				map[string]interface{}{"keepdims": int64(0)},
				tensor.New(tensor.WithShape(len(tc.input)), tensor.WithBacking(tc.input)))
			if err != nil {
				t.Fatal(err)
			}
			v := float64(firstFloat32(t, got))
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("got %v, want %v -- the shift by the per-axis max is missing", v, tc.want)
			}
			if math.Abs(v-tc.want) > 1e-3 {
				t.Errorf("got %v, want %v", v, tc.want)
			}
		})
	}
}

// TestReduceLogSumExpHighRank guards the rank limit. The shift needs the
// per-axis max at the input's shape, and gorgonia's BroadcastPattern is a
// single byte with four bits per operand -- an axis index of 4 silently sets a
// bit belonging to the other operand. Tiling the max instead has no such
// limit. Every ONNX fixture for this operator is rank 3 or less.
func TestReduceLogSumExpHighRank(t *testing.T) {
	// (2,2,2,2,3): rank 5, beyond gorgonia's 4-dimension broadcast pattern.
	shape := []int{2, 2, 2, 2, 3}
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%7) - 3
	}

	for _, tc := range []struct {
		name string
		axes []int64
	}{
		{"axis 4 (past the pattern limit)", []int64{4}},
		{"axis 0", []int64{0}},
		{"axes 1 and 3", []int64{1, 3}},
		{"default (all axes)", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			attrs := map[string]interface{}{"keepdims": int64(1)}
			if tc.axes != nil {
				attrs["axes"] = tc.axes
			}
			in := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(append([]float32(nil), data...)))
			got, err := applyReduce(t, "ReduceLogSumExp", attrs, in)
			if err != nil {
				t.Fatalf("rank-5 ReduceLogSumExp failed: %v", err)
			}
			// Spot-check the full reduction against a float64 reference.
			if tc.axes == nil {
				var sum float64
				for _, v := range data {
					sum += math.Exp(float64(v))
				}
				want := math.Log(sum)
				g := float64(firstFloat32(t, got))
				if math.Abs(g-want) > 1e-3 {
					t.Errorf("got %v, want %v", g, want)
				} else {
					t.Logf("full reduction = %v (reference %v)", g, want)
				}
			} else {
				t.Logf("ok, output shape %v", got.Shape())
			}
		})
	}
}

// TestReduceNoopWithEmptyAxes: with the flag set and no axes selected the
// operator is an identity, not a full collapse.
func TestReduceNoopWithEmptyAxes(t *testing.T) {
	in := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}))

	got, err := applyReduce(t, "ReduceSum",
		map[string]interface{}{"keepdims": int64(1), "noop_with_empty_axes": int64(1)}, in)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Shape().Eq(tensor.Shape{2, 3}) {
		t.Fatalf("shape %v, want (2,3): the input should pass through untouched", got.Shape())
	}

	// Without the flag, the same empty selection reduces everything.
	collapsed, err := applyReduce(t, "ReduceSum", map[string]interface{}{"keepdims": int64(1)}, in)
	if err != nil {
		t.Fatal(err)
	}
	if !collapsed.Shape().Eq(tensor.Shape{1, 1}) {
		t.Errorf("shape %v, want (1,1) without noop_with_empty_axes", collapsed.Shape())
	}
}

// TestReduceAxisOutOfRange: a malformed axis must be reported as an ordinary
// error. ErrNotImplemented would be read by the loader and the test harness as
// "this backend does not support the operator" and skipped, hiding it.
func TestReduceAxisOutOfRange(t *testing.T) {
	in := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}))

	_, err := applyReduce(t, "ReduceSum",
		map[string]interface{}{"axes": []int64{5}}, in)
	if err == nil {
		t.Fatal("expected an error for an out-of-range axis")
	}
	if _, ok := err.(*onnx.ErrNotImplemented); ok {
		t.Errorf("axis validation reported as ErrNotImplemented, which callers skip: %v", err)
	}
}

// TestReduceNegativeAxes checks the negative-axis normalization the fixtures
// only cover for a couple of the operators.
func TestReduceNegativeAxes(t *testing.T) {
	in := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}))

	neg, err := applyReduce(t, "ReduceSum",
		map[string]interface{}{"axes": []int64{-1}, "keepdims": int64(0)}, in)
	if err != nil {
		t.Fatal(err)
	}
	pos, err := applyReduce(t, "ReduceSum",
		map[string]interface{}{"axes": []int64{1}, "keepdims": int64(0)}, in)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := neg.Data(), pos.Data(); !tensorDataEqual(got, want) {
		t.Errorf("axis -1 gave %v, axis 1 gave %v; they address the same axis", got, want)
	}
}

// firstFloat32 reads the leading element, tolerating a fully reduced tensor
// whose Data() is a bare scalar rather than a slice.
func firstFloat32(t *testing.T, x tensor.Tensor) float32 {
	t.Helper()
	switch d := x.Data().(type) {
	case []float32:
		return d[0]
	case float32:
		return d
	default:
		t.Fatalf("unexpected output type %T", x.Data())
		return 0
	}
}

func tensorDataEqual(a, b interface{}) bool {
	af, aok := a.([]float32)
	bf, bok := b.([]float32)
	if !aok || !bok || len(af) != len(bf) {
		return false
	}
	for i := range af {
		if af[i] != bf[i] {
			return false
		}
	}

	return true
}
