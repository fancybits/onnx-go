package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// Concat's empty-tensor workaround lives in concatTensors, the all-constant
// path. A graph input reaches the symbolic path instead (gorgonia.Concat),
// which panics on an empty operand. A runtime float32[0] graph input
// concatenated with a float32[1] initializer must still compute the
// initializer's element, not crash.
func TestConcatSymbolicWithEmptyGraphInput(t *testing.T) {
	m := g2model(&ir.GraphProto{
		Name:  "concat_empty_mixed",
		Input: []*ir.ValueInfoProto{provenanceValueInfo("x", ir.TensorProto_FLOAT, 0)},
		Initializer: []*ir.TensorProto{
			{Name: "y", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{42}},
		},
		Node: []*ir.NodeProto{
			{
				OpType: "Concat", Input: []string{"x", "y"}, Output: []string{"joined"},
				Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
			},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_FLOAT, 1)},
	})

	backend, model := buildProvenanceModel(t, m)
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(0), tensor.WithBacking([]float32{}))))

	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []float32{42}, outs[0].Data())
}

// When every operand on the symbolic path is empty along the concat axis,
// there is nothing to concatenate. This must be handled degenerately rather
// than reaching gorgonia.Concat with no operands at all.
func TestConcatSymbolicAllEmpty(t *testing.T) {
	m := g2model(&ir.GraphProto{
		Name: "concat_empty_all",
		Input: []*ir.ValueInfoProto{
			provenanceValueInfo("x", ir.TensorProto_FLOAT, 0),
			provenanceValueInfo("y", ir.TensorProto_FLOAT, 0),
		},
		Node: []*ir.NodeProto{
			{
				OpType: "Concat", Input: []string{"x", "y"}, Output: []string{"joined"},
				Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
			},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_FLOAT, 0)},
	})

	backend, model := buildProvenanceModel(t, m)
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(0), tensor.WithBacking([]float32{}))))
	require.NoError(t, model.SetInput(1, tensor.New(
		tensor.WithShape(0), tensor.WithBacking([]float32{}))))

	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, 0, outs[0].Shape().TotalSize())
}
