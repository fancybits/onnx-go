package gorgonnx

import (
	"testing"

	"github.com/gogo/protobuf/proto"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// loopCaptureChainModel puts the producer of a captured value behind a chain of
// operations, so that whether it has been built by the time the Loop group is
// reached depends on the order the walk happens to visit groups in — a captured
// value carries no edge from the Loop node to its producer, so the walk's
// readiness check cannot see the dependency.
//
//	x -> Identity -> a -> Identity -> b
//	Loop(M, cond, acc0) { body: sum = acc + b }
//
// The body captures "b", but the Loop node's only edges are to M, cond and
// acc0. With the groups in this order the walk applies "a", then — because
// deleting a group shifts the slice under the loop index — skips "b" and
// reaches the Loop in the same pass, with "b" still unbuilt.
func loopCaptureChainModel(captured string) *ir.ModelProto {
	body := &ir.GraphProto{
		Name: "body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Add", Input: []string{"acc", captured}, Output: []string{"sum"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 1),
		},
	}
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name: "main",
			Initializer: []*ir.TensorProto{
				{Name: "x", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{5}},
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{3}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "acc0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{1}},
			},
			Node: []*ir.NodeProto{
				{OpType: "Identity", Input: []string{"x"}, Output: []string{"a"}},
				{OpType: "Identity", Input: []string{"a"}, Output: []string{"b"}},
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "acc0"},
					Output: []string{"final"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: body},
					},
				},
			},
			ValueInfo: []*ir.ValueInfoProto{
				loopTestF32ValueInfo("a", 1),
				loopTestF32ValueInfo("b", 1),
			},
			Output: []*ir.ValueInfoProto{loopTestF32ValueInfo("final", 1)},
		},
	}
}

// A Loop whose captured value is produced later in the walk must be deferred to
// a later pass, not rejected: the model is valid, only the visit order is
// unlucky.
func TestLoopCapturedValueBuiltAfterLoopGroup(t *testing.T) {
	raw, err := proto.Marshal(loopCaptureChainModel("b"))
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []float32{16}, outs[0].Data(), "acc0 + 3*b = 1 + 3*5")
}

// Deferring an unresolvable capture must not turn a broken model into a hang or
// an opaque failure: once no pass can make progress, the build must fail naming
// the loop and the value it could not find.
func TestLoopCapturedValueNeverResolves(t *testing.T) {
	raw, err := proto.Marshal(loopCaptureChainModel("nowhere"))
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	err = model.UnmarshalBinary(raw)
	if err == nil {
		err = backend.Run()
	}
	require.Error(t, err)
	assert.Contains(t, err.Error(), "loop:")
	assert.Contains(t, err.Error(), `"nowhere"`)
	assert.NotContains(t, err.Error(), "infinite loop")
}
