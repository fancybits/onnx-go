package gorgonnx

import (
	"testing"

	"github.com/gogo/protobuf/proto"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

// A graph input carries a tensor before the graph is ever built, and SetInput
// may already have filled it with this run's data. That makes it
// indistinguishable from an initializer by value alone, so a trip count taken
// from one would be silently baked into the unroll — the model would then
// ignore every later value of that input. It must be rejected on provenance.
func TestLoopTripCountFromBoundGraphInputRejected(t *testing.T) {
	raw, err := proto.Marshal(loopTestModelWithInputTripCount())
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]int64{3}))))

	err = backend.Run()
	require.Error(t, err, "a trip count bound into a graph input must not be baked into the unroll")
	assert.Contains(t, err.Error(), "loop:")
	assert.Contains(t, err.Error(), "constant")
}

// A trip count produced by a Constant node is a genuine compile-time constant
// and must still be accepted.
func TestLoopTripCountFromConstantNode(t *testing.T) {
	m := loopTestConstantModel(3)
	inits := m.Graph.Initializer
	m.Graph.Initializer = nil
	for _, init := range inits {
		if init.Name == "M" {
			continue
		}
		m.Graph.Initializer = append(m.Graph.Initializer, init)
	}
	m.Graph.Node = append([]*ir.NodeProto{
		{
			Name:   "trip_count",
			OpType: "Constant",
			Output: []string{"M"},
			Attribute: []*ir.AttributeProto{
				{
					Name: "value",
					Type: ir.AttributeProto_TENSOR,
					T: &ir.TensorProto{
						DataType: int32(ir.TensorProto_INT64),
						Dims:     []int64{1},
						Int64Data: []int64{
							3,
						},
					},
				},
			},
		},
	}, m.Graph.Node...)

	raw, err := proto.Marshal(m)
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []float32{7}, outs[0].Data(), "1 + 3*2")
}

// loopCarriedFromInputModel feeds the loop's carried initial value straight
// from a graph input, and routes it through Split+Concat inside the body — a
// pair whose constant fast path folds at build time. Folding the input's bound
// tensor would bake the first run's data into the graph, so every later run
// with a different input would return the first run's answer.
//
//	acc0 (graph input) -> Loop { acc -> Split -> Concat -> Add(outer) -> sum }
func loopCarriedFromInputModel() *ir.ModelProto {
	body := &ir.GraphProto{
		Name: "body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 2),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Split", Input: []string{"acc"}, Output: []string{"a_lo", "a_hi"}},
			{
				OpType: "Concat", Input: []string{"a_lo", "a_hi"}, Output: []string{"acc_re"},
				Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
			},
			{OpType: "Add", Input: []string{"acc_re", "outer"}, Output: []string{"sum"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 2),
		},
	}
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name:  "main",
			Input: []*ir.ValueInfoProto{loopTestF32ValueInfo("acc0", 2)},
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{3}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "outer", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{2}, FloatData: []float32{2, 2}},
			},
			Node: []*ir.NodeProto{
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "acc0"},
					Output: []string{"final"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: body},
					},
				},
			},
			Output: []*ir.ValueInfoProto{loopTestF32ValueInfo("final", 2)},
		},
	}
}

func TestLoopCarriedValueFromGraphInputIsNotBaked(t *testing.T) {
	raw, err := proto.Marshal(loopCarriedFromInputModel())
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(2), tensor.WithBacking([]float32{1, 1}))))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []float32{7, 7}, outs[0].Data(), "1 + 3*2")

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(2), tensor.WithBacking([]float32{10, 10}))))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []float32{16, 16}, outs[0].Data(),
		"the second run must use the second input, not the value folded on the first")
}

// The same hazard without any Loop: Split must not fold a graph input's bound
// tensor into the graph.
func TestSplitDoesNotFoldGraphInput(t *testing.T) {
	m := &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name:  "main",
			Input: []*ir.ValueInfoProto{loopTestF32ValueInfo("x", 4)},
			Node: []*ir.NodeProto{
				{OpType: "Split", Input: []string{"x"}, Output: []string{"lo", "hi"}},
			},
			Output: []*ir.ValueInfoProto{
				loopTestF32ValueInfo("lo", 2),
				loopTestF32ValueInfo("hi", 2),
			},
		},
	}
	raw, err := proto.Marshal(m)
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(4), tensor.WithBacking([]float32{1, 2, 3, 4}))))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 2}, outs[0].Data())
	assert.Equal(t, []float32{3, 4}, outs[1].Data())

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(4), tensor.WithBacking([]float32{5, 6, 7, 8}))))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []float32{5, 6}, outs[0].Data(), "second run must not return the first run's folded slice")
	assert.Equal(t, []float32{7, 8}, outs[1].Data(), "second run must not return the first run's folded slice")
}
