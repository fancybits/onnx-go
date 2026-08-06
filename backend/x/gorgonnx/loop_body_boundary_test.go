package gorgonnx

import (
	"testing"

	"github.com/gogo/protobuf/proto"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// loopOmittedOptionalInputBody slices the carried value with the optional
// "axes" input omitted. ONNX spells an omitted optional input as the empty
// name, which is a reference to nothing — not to a value in the enclosing
// scope.
func loopOmittedOptionalInputBody() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "body_omitted_optional",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 2),
		},
		Initializer: []*ir.TensorProto{
			{Name: "starts", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{0}},
			{Name: "ends", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{2}},
			{Name: "steps", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{1}},
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Slice", Input: []string{"acc", "starts", "ends", "", "steps"}, Output: []string{"sliced"}},
			{OpType: "Add", Input: []string{"sliced", "outer"}, Output: []string{"sum"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 2),
		},
	}
}

func loopOmittedOptionalInputModel() *ir.ModelProto {
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name: "main",
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{3}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "acc0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{2}, FloatData: []float32{0, 0}},
				{Name: "outer", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{2}, FloatData: []float32{1, 1}},
			},
			Node: []*ir.NodeProto{
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "acc0"},
					Output: []string{"final"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: loopOmittedOptionalInputBody()},
					},
				},
			},
			Output: []*ir.ValueInfoProto{loopTestF32ValueInfo("final", 2)},
		},
	}
}

// A body node with an omitted optional input must run exactly as the same node
// does at the top level: the empty name must not be mistaken for a value
// captured from the enclosing scope.
func TestLoopBodyWithOmittedOptionalInput(t *testing.T) {
	raw, err := proto.Marshal(loopOmittedOptionalInputModel())
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []float32{3, 3}, outs[0].Data(), "0 + 3*1 on each element")
}

// loopOuterNamedOutputModel has a body scan output that names an enclosing
// scope value directly, with no Identity in between and no body node
// referencing it. It is a use of that value just as a node input would be.
func loopOuterNamedOutputModel() *ir.ModelProto {
	body := &ir.GraphProto{
		Name: "body_outer_output",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Add", Input: []string{"acc", "outer"}, Output: []string{"sum"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 1),
			loopTestF32ValueInfo("extra", 1),
		},
	}
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name: "main",
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{3}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "acc0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{1}},
				{Name: "outer", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{2}},
				{Name: "extra", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{9}},
			},
			Node: []*ir.NodeProto{
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "acc0"},
					Output: []string{"final", "scan"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: body},
					},
				},
			},
			Output: []*ir.ValueInfoProto{
				loopTestF32ValueInfo("final", 1),
				loopTestF32ValueInfo("scan", 3, 1),
			},
		},
	}
}

// A body output naming an enclosing-scope value must carry that value, not the
// zero-filled placeholder its own declaration created.
func TestLoopBodyOutputNamingOuterValue(t *testing.T) {
	raw, err := proto.Marshal(loopOuterNamedOutputModel())
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 2)
	assert.Equal(t, []float32{7}, outs[0].Data(), "1 + 3*2")
	assert.Equal(t, []int{3, 1}, []int(outs[1].Shape()))
	assert.Equal(t, []float32{9, 9, 9}, outs[1].Data(),
		"scan output naming an outer value must carry that value")
}
