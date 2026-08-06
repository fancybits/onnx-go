package gorgonnx

import (
	"testing"

	"github.com/gogo/protobuf/proto"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// loopTestF32ValueInfo builds a float32 tensor value_info with static dims.
func loopTestF32ValueInfo(name string, dims ...int64) *ir.ValueInfoProto {
	shape := &ir.TensorShapeProto{}
	for _, d := range dims {
		shape.Dim = append(shape.Dim, &ir.TensorShapeProto_Dimension{
			Value: &ir.TensorShapeProto_Dimension_DimValue{DimValue: d},
		})
	}
	return &ir.ValueInfoProto{
		Name: name,
		Type: &ir.TypeProto{
			Value: &ir.TypeProto_TensorType{
				TensorType: &ir.TypeProto_Tensor{
					ElemType: int32(ir.TensorProto_FLOAT),
					Shape:    shape,
				},
			},
		},
	}
}

// loopTestBody is the Loop body:
//
//	(iter, cond_in, acc, keep) -> (cond_out, sum, keep, scan_out)
//
// "acc" accumulates the outer-scope constant "outer" (a captured value),
// "keep" is a pass-through carried value returned unchanged, and "scan_out"
// records the running accumulator on every iteration.
func loopTestBody() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 1),
			loopTestF32ValueInfo("keep", 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Add", Input: []string{"acc", "outer"}, Output: []string{"sum"}},
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Identity", Input: []string{"sum"}, Output: []string{"scan_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 1),
			// Pass-through carried value: an output that names a body input.
			loopTestF32ValueInfo("keep", 1),
			loopTestF32ValueInfo("scan_out", 1),
		},
	}
}

// loopTestModel returns a model whose only node is a Loop with a
// compile-time-constant trip count supplied as an initializer.
func loopTestModel(tripCount int64) *ir.ModelProto {
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name: "main",
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{tripCount}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "acc0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{1}},
				{Name: "keep0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{42}},
				{Name: "outer", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{2}},
			},
			Node: []*ir.NodeProto{
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "acc0", "keep0"},
					Output: []string{"final", "keep_final", "scan"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: loopTestBody()},
					},
				},
			},
			Output: []*ir.ValueInfoProto{
				loopTestF32ValueInfo("final", 1),
				loopTestF32ValueInfo("keep_final", 1),
				loopTestF32ValueInfo("scan", 3, 1),
			},
		},
	}
}

// loopTestModelWithInputTripCount returns the same model, but with the trip
// count supplied as a graph *input* rather than an initializer — i.e. a value
// that is not known at graph-build time.
func loopTestModelWithInputTripCount() *ir.ModelProto {
	m := loopTestModel(3)
	inits := m.Graph.Initializer
	m.Graph.Initializer = nil
	for _, init := range inits {
		if init.Name == "M" {
			continue
		}
		m.Graph.Initializer = append(m.Graph.Initializer, init)
	}
	m.Graph.Input = []*ir.ValueInfoProto{
		{
			Name: "M",
			Type: &ir.TypeProto{
				Value: &ir.TypeProto_TensorType{
					TensorType: &ir.TypeProto_Tensor{
						ElemType: int32(ir.TensorProto_INT64),
						Shape: &ir.TensorShapeProto{
							Dim: []*ir.TensorShapeProto_Dimension{
								{Value: &ir.TensorShapeProto_Dimension_DimValue{DimValue: 1}},
							},
						},
					},
				},
			},
		},
	}
	return m
}

// loopTestFoldingModel builds a Loop whose body routes a value through
// Split+Concat — a dtype-agnostic constant-folding pair — before accumulating
// it. Constant folding only produces the right answer when the value fed into
// it is the value the node will actually hold; a zero-filled placeholder
// silently folds to zeros instead. `throughCaptured` selects which side of the
// body boundary is put under test:
//
//	false: the CARRIED value goes through Split+Concat. On every iteration
//	       after the first, the carried input is a runtime node, so nothing
//	       may be folded. Threading the body output's declared-value_info
//	       placeholder through instead folds [0 0] and yields [2 2].
//
//	true:  the CAPTURED value goes through Split+Concat, and is declared in
//	       the parent graph's value_info (produced by an Identity from an
//	       initializer) so its Node.t is a zero-filled placeholder while its
//	       gorgonia node holds the real [2 2]. Reading Node.t folds [0 0] and
//	       leaves the accumulator at its initial [1 1].
//
// Either way the correct answer is [1 1] + 3*[2 2] = [7 7].
func loopTestFoldingModel(throughCaptured bool) *ir.ModelProto {
	split := func(in string, out ...string) *ir.NodeProto {
		return &ir.NodeProto{OpType: "Split", Input: []string{in}, Output: out}
	}
	concat := func(out string, in ...string) *ir.NodeProto {
		return &ir.NodeProto{
			OpType: "Concat", Input: in, Output: []string{out},
			Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
		}
	}

	bodyNodes := []*ir.NodeProto{
		{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
	}
	if throughCaptured {
		bodyNodes = append(bodyNodes,
			split("outer", "o_lo", "o_hi"),
			concat("outer_re", "o_lo", "o_hi"),
			&ir.NodeProto{OpType: "Add", Input: []string{"acc", "outer_re"}, Output: []string{"sum"}},
		)
	} else {
		bodyNodes = append(bodyNodes,
			split("acc", "a_lo", "a_hi"),
			concat("acc_re", "a_lo", "a_hi"),
			&ir.NodeProto{OpType: "Add", Input: []string{"acc_re", "outer"}, Output: []string{"sum"}},
		)
	}
	body := &ir.GraphProto{
		Name: "folding_body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 2),
		},
		Node: bodyNodes,
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 2),
		},
	}

	main := &ir.GraphProto{
		Name: "main",
		Initializer: []*ir.TensorProto{
			{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{3}},
			{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
			{Name: "acc0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{2}, FloatData: []float32{1, 1}},
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
	}

	if throughCaptured {
		// "outer" is an intermediate value declared in value_info, so its node
		// carries a zero-filled placeholder tensor alongside the real value
		// bound to its gorgonia node.
		main.Initializer = append(main.Initializer, &ir.TensorProto{
			Name: "outer_src", DataType: int32(ir.TensorProto_FLOAT),
			Dims: []int64{2}, FloatData: []float32{2, 2},
		})
		main.ValueInfo = []*ir.ValueInfoProto{loopTestF32ValueInfo("outer", 2)}
		main.Node = append([]*ir.NodeProto{
			{OpType: "Identity", Input: []string{"outer_src"}, Output: []string{"outer"}},
		}, main.Node...)
	} else {
		main.Initializer = append(main.Initializer, &ir.TensorProto{
			Name: "outer", DataType: int32(ir.TensorProto_FLOAT),
			Dims: []int64{2}, FloatData: []float32{2, 2},
		})
	}

	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph:       main,
	}
}

func runLoopFoldingModel(t *testing.T, throughCaptured bool) []float32 {
	t.Helper()
	raw, err := proto.Marshal(loopTestFoldingModel(throughCaptured))
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	data, ok := outs[0].Data().([]float32)
	require.True(t, ok, "output should be []float32, got %T", outs[0].Data())
	return data
}

// A carried value must never be seeded with the body output's declared
// value_info placeholder: the next iteration's constant folding would fold
// those zeros. Yields [2 2] instead of [7 7] if the placeholder leaks in.
func TestLoopCarriedValueIsNotConstantFolded(t *testing.T) {
	assert.Equal(t, []float32{7, 7}, runLoopFoldingModel(t, false),
		"carried value seeded from a placeholder tensor would fold to [2 2]")
}

// Same requirement on the captured boundary: a captured parent node declared in
// the enclosing graph's value_info carries a placeholder in Node.t, so the
// value must be read off its gorgonia node. Yields [1 1] if Node.t leaks in.
func TestLoopCapturedValueIsNotConstantFolded(t *testing.T) {
	assert.Equal(t, []float32{7, 7}, runLoopFoldingModel(t, true),
		"captured value seeded from a placeholder tensor would fold to [1 1]")
}

func TestLoopStaticUnroll(t *testing.T) {
	raw, err := proto.Marshal(loopTestModel(3))
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 3)
	assert.Equal(t, []float32{7}, outs[0].Data(), "final carried value")
	assert.Equal(t, []float32{42}, outs[1].Data(), "pass-through carried value must equal its initial value")
	assert.Equal(t, []int{3, 1}, []int(outs[2].Shape()))
	assert.Equal(t, []float32{3, 5, 7}, outs[2].Data(), "scan output stacked on new leading axis")
}

func TestLoopZeroTripCount(t *testing.T) {
	raw, err := proto.Marshal(loopTestModel(0))
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	err = model.UnmarshalBinary(raw)
	if err == nil {
		err = backend.Run()
	}
	require.Error(t, err, "zero trip count must fail loudly")
	assert.Contains(t, err.Error(), "loop:")
	assert.Contains(t, err.Error(), "trip count")
}

// loopTestConstantBody is a Loop body containing a zero-input Constant node:
//
//	(iter, cond_in, acc) -> (cond_out, sum)
//
// The Constant node has no ONNX inputs; decoder.go synthesizes a fake
// "<node.Name>/input" placeholder for it during decode. That synthesized
// name must not be misclassified as a captured value referring to the
// enclosing scope.
func loopTestConstantBody() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "constant_body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			loopTestF32ValueInfo("acc", 1),
		},
		Node: []*ir.NodeProto{
			{
				Name:   "const_node",
				OpType: "Constant",
				Output: []string{"c"},
				Attribute: []*ir.AttributeProto{
					{
						Name: "value",
						Type: ir.AttributeProto_TENSOR,
						T: &ir.TensorProto{
							DataType: int32(ir.TensorProto_FLOAT),
							Dims:     []int64{1},
							FloatData: []float32{
								2,
							},
						},
					},
				},
			},
			{OpType: "Add", Input: []string{"acc", "c"}, Output: []string{"sum"}},
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			loopTestF32ValueInfo("sum", 1),
		},
	}
}

// loopTestConstantModel wraps loopTestConstantBody in a top-level Loop with a
// fixed trip count.
func loopTestConstantModel(tripCount int64) *ir.ModelProto {
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name: "main",
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{tripCount}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "acc0", DataType: int32(ir.TensorProto_FLOAT), Dims: []int64{1}, FloatData: []float32{1}},
			},
			Node: []*ir.NodeProto{
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "acc0"},
					Output: []string{"final"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: loopTestConstantBody()},
					},
				},
			},
			Output: []*ir.ValueInfoProto{
				loopTestF32ValueInfo("final", 1),
			},
		},
	}
}

// A Loop body containing a Constant node (a zero-input op) must decode and
// run correctly: the decoder-synthesized fake input name for the Constant
// must not be misclassified as a captured value from the enclosing scope.
// acc0=1, and each of the 3 iterations adds the constant 2: 1 + 3*2 = 7.
func TestLoopBodyWithConstantNode(t *testing.T) {
	raw, err := proto.Marshal(loopTestConstantModel(3))
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []float32{7}, outs[0].Data(), "final carried value: 1 + 3*2")
}

func TestLoopNonConstantTripCount(t *testing.T) {
	raw, err := proto.Marshal(loopTestModelWithInputTripCount())
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	err = model.UnmarshalBinary(raw)
	if err == nil {
		err = backend.Run()
	}
	require.Error(t, err, "a non-constant trip count must fail loudly")
	assert.Contains(t, err.Error(), "loop:")
	assert.Contains(t, err.Error(), "constant")
}
