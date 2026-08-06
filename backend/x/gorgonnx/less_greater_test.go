package gorgonnx

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLessScalarFloat32(t *testing.T) {
	// The bigru artifact's while-loop condition bookkeeping compares 0-d
	// float32 scalars. gorgonia.Lt used to panic on this
	// ("scalarBinOp.Do() - Unhandled Scalar Type not yet implemented for
	// float32"); the fork now normalizes 0-d Dense operands in
	// scalarBinOp.Do, and this test pins that fix through the gorgonnx
	// path.
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{1}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{2}))
	got := applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{2}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{1}))
	got = applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, false, got.Data())
}

func TestLessScalarFloat64(t *testing.T) {
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))
	got := applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))
	got = applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, false, got.Data())
}

func TestLessScalarInt32(t *testing.T) {
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{1}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{2}))
	got := applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{2}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{1}))
	got = applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, false, got.Data())
}

func TestLessScalarInt64(t *testing.T) {
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{1}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{2}))
	got := applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{2}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{1}))
	got = applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, false, got.Data())
}

func TestLessVector(t *testing.T) {
	a := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{1, 2, 3, 4}))
	b := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{2, 2, 2, 2}))
	got := applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, []bool{true, false, false, false}, got.Data())
}

func TestLessBroadcastScalarVsVector(t *testing.T) {
	a := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{1, 2, 3, 4}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{3}))
	got := applyBinaryOp(t, "Less", a, b)
	assert.Equal(t, []bool{true, true, false, false}, got.Data())
}

func TestGreaterScalarFloat32(t *testing.T) {
	// Same scalar case as Less: gorgonia.Gt now handles 0-d tensors via
	// the fork's scalarBinOp.Do fix, pinned by this test.
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{2}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{1}))
	got := applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{2}))
	got = applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, false, got.Data())
}

func TestGreaterScalarFloat64(t *testing.T) {
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))
	got := applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]float64{2}))
	got = applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, false, got.Data())
}

func TestGreaterScalarInt32(t *testing.T) {
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{2}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{1}))
	got := applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]int32{2}))
	got = applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, false, got.Data())
}

func TestGreaterScalarInt64(t *testing.T) {
	a := tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{2}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{1}))
	got := applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, true, got.Data())

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{2}))
	got = applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, false, got.Data())
}

func TestGreaterVector(t *testing.T) {
	a := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{1, 2, 3, 4}))
	b := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{2, 2, 2, 2}))
	got := applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, []bool{false, false, true, true}, got.Data())
}

func TestGreaterBroadcastScalarVsVector(t *testing.T) {
	a := tensor.New(tensor.WithShape(4), tensor.WithBacking([]float32{1, 2, 3, 4}))
	b := tensor.New(tensor.WithShape(), tensor.WithBacking([]float32{3}))
	got := applyBinaryOp(t, "Greater", a, b)
	assert.Equal(t, []bool{false, false, false, true}, got.Data())
}
