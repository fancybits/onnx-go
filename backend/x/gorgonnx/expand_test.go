package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// expandBoolModel expands a bool graph input to a larger shape via an
// initializer target shape, so the data operand cannot take the constant
// fast path and must go symbolic.
func expandBoolModel() *ir.ModelProto {
	return g2model(&ir.GraphProto{
		Name:  "expand_bool",
		Input: []*ir.ValueInfoProto{provenanceValueInfo("x", ir.TensorProto_BOOL, 1)},
		Initializer: []*ir.TensorProto{
			{Name: "newshape", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{3}},
		},
		Node: []*ir.NodeProto{
			{OpType: "Expand", Input: []string{"x", "newshape"}, Output: []string{"expanded"}},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("expanded", ir.TensorProto_BOOL, 3)},
	})
}

// Expand's runtime/symbolic path broadcasts via a Hadamard product against a
// ones tensor, which only typechecks for numeric dtypes. A bool graph input
// bound at run time cannot take the constant fast path, so it used to reach
// gorgonia's multiply and fail deep in the tape with an opaque typeclass
// error ("Type bool is not a member of Number"). It must instead fail
// loudly and specifically at build time.
func TestExpandRuntimeBoolFailsLoudly(t *testing.T) {
	backend, model := buildProvenanceModel(t, expandBoolModel())
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]bool{true}))))

	err := backend.Run()
	require.Error(t, err, "runtime bool expansion is not supported and must fail, not silently misbehave")
	assert.Contains(t, err.Error(), "expand: runtime bool expansion is not supported (symbolic path is numeric-only)")
}

// The same bool input against a constant target shape can still take the
// constant fast path when the data itself is a genuine compile-time
// constant (an initializer, not a graph input) — that path is unaffected by
// the new symbolic-path guard.
func TestExpandConstantBoolStillWorks(t *testing.T) {
	m := g2model(&ir.GraphProto{
		Name: "expand_bool_const",
		Initializer: []*ir.TensorProto{
			{Name: "x", DataType: int32(ir.TensorProto_BOOL), Dims: []int64{1}, Int32Data: []int32{1}},
			{Name: "newshape", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{3}},
		},
		Node: []*ir.NodeProto{
			{OpType: "Expand", Input: []string{"x", "newshape"}, Output: []string{"expanded"}},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("expanded", ir.TensorProto_BOOL, 3)},
	})

	backend, model := buildProvenanceModel(t, m)
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []bool{true, true, true}, outs[0].Data())
}
