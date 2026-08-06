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

// provenanceValueInfo builds a value_info of the given element type with static
// dims.
func provenanceValueInfo(name string, elem ir.TensorProto_DataType, dims ...int64) *ir.ValueInfoProto {
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
					ElemType: int32(elem),
					Shape:    shape,
				},
			},
		},
	}
}

// buildProvenanceModel marshals and decodes m onto a fresh backend.
func buildProvenanceModel(t *testing.T, m *ir.ModelProto) (*Graph, *onnx.Model) {
	t.Helper()
	raw, err := proto.Marshal(m)
	require.NoError(t, err)
	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	return backend, model
}

// nodeNamed finds a node of the backend graph by its ONNX name.
func nodeNamed(t *testing.T, g *Graph, name string) *Node {
	t.Helper()
	it := g.g.Nodes()
	for it.Next() {
		n := it.Node().(*Node)
		if n.name == name {
			return n
		}
	}
	t.Fatalf("no node named %q in the graph", name)
	return nil
}

// A Shape feeds a chain of operators that each have to know their input's
// value while the graph is built. Shape derives its result from the exprgraph's
// static shapes, so it is a genuine compile-time constant and every step below
// it must keep folding. Gating the fold sites on provenance without marking
// this chain severs it: Split refuses to fold, Concat has nothing to
// concatenate, and Reshape fails outright with "shape input has no value".
func TestShapeSplitConcatReshapeChainStillFolds(t *testing.T) {
	m := &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name:  "shape_split_concat_reshape",
			Input: []*ir.ValueInfoProto{provenanceValueInfo("x", ir.TensorProto_FLOAT, 2, 3, 4, 5)},
			Node: []*ir.NodeProto{
				{OpType: "Shape", Input: []string{"x"}, Output: []string{"shape"}},
				// [2,3,4,5] -> [2,3] and [4,5]
				{
					OpType: "Split",
					Input:  []string{"shape"},
					Output: []string{"lo", "hi"},
					Attribute: []*ir.AttributeProto{
						{Name: "axis", Type: ir.AttributeProto_INT, I: 0},
					},
				},
				// Swapped, so the reshape is not a no-op: [4,5,2,3]
				{
					OpType: "Concat",
					Input:  []string{"hi", "lo"},
					Output: []string{"swapped"},
					Attribute: []*ir.AttributeProto{
						{Name: "axis", Type: ir.AttributeProto_INT, I: 0},
					},
				},
				{OpType: "Reshape", Input: []string{"x", "swapped"}, Output: []string{"y"}},
			},
			Output: []*ir.ValueInfoProto{provenanceValueInfo("y", ir.TensorProto_FLOAT, 4, 5, 2, 3)},
		},
	}

	backend, model := buildProvenanceModel(t, m)
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(2, 3, 4, 5),
		tensor.WithBacking(make([]float32, 2*3*4*5)))))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []int{4, 5, 2, 3}, []int(outs[0].Shape()))

	// Every step of the chain must carry the provenance that let it fold.
	for _, name := range []string{"shape", "lo", "hi", "swapped"} {
		assert.True(t, isConstNode(nodeNamed(t, backend, name)),
			"%q derives only from the graph's static shapes and must be a constant", name)
	}
}

// integerAddModel adds a graph input to an initializer. Both are integers, so
// this is the "fold integer arithmetic for shape computations" path.
func integerAddModel() *ir.ModelProto {
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name:  "integer_add",
			Input: []*ir.ValueInfoProto{provenanceValueInfo("k", ir.TensorProto_INT64, 1)},
			Initializer: []*ir.TensorProto{
				{Name: "two", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{2}},
			},
			Node: []*ir.NodeProto{
				{OpType: "Add", Input: []string{"k", "two"}, Output: []string{"sum"}},
			},
			Output: []*ir.ValueInfoProto{provenanceValueInfo("sum", ir.TensorProto_INT64, 1)},
		},
	}
}

// The integer fold used to take whatever tensor the operand carried. A graph
// input carries one too — the tensor SetInput bound before the graph was built
// — so the first run's value was computed into a constant and pinned into the
// graph. Every later run then returned the first run's answer, however the
// input changed.
func TestIntegerAddDoesNotBakeBoundGraphInput(t *testing.T) {
	backend, model := buildProvenanceModel(t, integerAddModel())

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]int64{1}))))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{3}, outs[0].Data(), "1 + 2")

	// The fold must not have happened, so the second run sees the new input.
	assert.False(t, isConstNode(nodeNamed(t, backend, "sum")),
		"a sum involving a graph input is not a compile-time constant")

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]int64{10}))))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{12}, outs[0].Data(),
		"10 + 2: the second run must not return the first run's answer")
}

// An Add of two initializers has no run-time component at all and must still
// fold, so that the shape computations that depend on it keep working.
func TestIntegerAddOfInitializersStillFolds(t *testing.T) {
	m := integerAddModel()
	m.Graph.Input = nil
	m.Graph.Initializer = append(m.Graph.Initializer, &ir.TensorProto{
		Name: "k", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{1},
	})

	backend, model := buildProvenanceModel(t, m)
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{3}, outs[0].Data())
	assert.True(t, isConstNode(nodeNamed(t, backend, "sum")),
		"a sum of two initializers is a compile-time constant")
}

// loopCapturedAddModel runs a Loop whose body adds a value captured from the
// enclosing scope to the carried accumulator. The captured value is a graph
// input, so the sum is 2*k after two iterations.
func loopCapturedAddModel() *ir.ModelProto {
	body := &ir.GraphProto{
		Name: "captured_add_body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			provenanceValueInfo("acc", ir.TensorProto_INT64, 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Add", Input: []string{"acc", "k"}, Output: []string{"sum"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			provenanceValueInfo("sum", ir.TensorProto_INT64, 1),
		},
	}
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name:  "loop_captured_add",
			Input: []*ir.ValueInfoProto{provenanceValueInfo("k", ir.TensorProto_INT64, 1)},
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{2}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
				{Name: "acc0", DataType: int32(ir.TensorProto_INT64), Dims: []int64{1}, Int64Data: []int64{0}},
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
			Output: []*ir.ValueInfoProto{provenanceValueInfo("final", ir.TensorProto_INT64, 1)},
		},
	}
}

// The same staleness inside a Loop body, where it also has a second way in: a
// value the body captures from the enclosing scope, and a carried value the
// loop threads from one iteration to the next. Neither may turn a graph input
// into a constant.
func TestLoopCapturedAddDoesNotBakeBoundGraphInput(t *testing.T) {
	backend, model := buildProvenanceModel(t, loopCapturedAddModel())

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]int64{1}))))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{2}, outs[0].Data(), "0 + 2*1")

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]int64{10}))))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{20}, outs[0].Data(),
		"0 + 2*10: the captured value must not be baked in by the first run")
}

// concatProvenanceModel concatenates two 1-D int64 values.
func concatProvenanceModel(secondIsInput bool) *ir.ModelProto {
	g := &ir.GraphProto{
		Name: "concat_provenance",
		Initializer: []*ir.TensorProto{
			{Name: "a", DataType: int32(ir.TensorProto_INT64), Dims: []int64{2}, Int64Data: []int64{1, 2}},
		},
		Node: []*ir.NodeProto{
			{
				OpType: "Concat",
				Input:  []string{"a", "b"},
				Output: []string{"joined"},
				Attribute: []*ir.AttributeProto{
					{Name: "axis", Type: ir.AttributeProto_INT, I: 0},
				},
			},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_INT64, 4)},
	}
	if secondIsInput {
		g.Input = []*ir.ValueInfoProto{provenanceValueInfo("b", ir.TensorProto_INT64, 2)}
	} else {
		g.Initializer = append(g.Initializer, &ir.TensorProto{
			Name: "b", DataType: int32(ir.TensorProto_INT64), Dims: []int64{2}, Int64Data: []int64{3, 4},
		})
	}
	return g2model(g)
}

func g2model(g *ir.GraphProto) *ir.ModelProto {
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph:       g,
	}
}

// A concat of two initializers has no run-time component, so it folds and its
// result is itself a constant available to build-time consumers.
func TestConcatOfInitializersFolds(t *testing.T) {
	backend, model := buildProvenanceModel(t, concatProvenanceModel(false))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3, 4}, outs[0].Data())
	assert.True(t, isConstNode(nodeNamed(t, backend, "joined")),
		"a concat of two initializers is a compile-time constant")
}

// A concat with a graph input must not fold. The graph still has to compute the
// right answer — it just computes it symbolically, per run, which is the whole
// point of refusing the fold.
func TestConcatWithGraphInputStaysSymbolic(t *testing.T) {
	backend, model := buildProvenanceModel(t, concatProvenanceModel(true))

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(2), tensor.WithBacking([]int64{3, 4}))))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 3, 4}, outs[0].Data())

	joined := nodeNamed(t, backend, "joined")
	assert.False(t, isConstNode(joined),
		"a concat involving a graph input is not a compile-time constant")

	// Symbolically is not "not at all": a second run must track the new input.
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(2), tensor.WithBacking([]int64{7, 8}))))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []int64{1, 2, 7, 8}, outs[0].Data(),
		"the second run must not return the first run's answer")
}

// ---------------------------------------------------------------------------
// Carried values threaded through an alias
// ---------------------------------------------------------------------------

// loopCarriedAliasModel carries a graph input through the loop unchanged, via
// an Identity, and negates it into a scan output on every iteration.
func loopCarriedAliasModel() *ir.ModelProto {
	body := &ir.GraphProto{
		Name: "carried_alias_body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			provenanceValueInfo("acc", ir.TensorProto_FLOAT, 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			// The carried value passes straight through, aliasing its node.
			{OpType: "Identity", Input: []string{"acc"}, Output: []string{"acc_out"}},
			{OpType: "Neg", Input: []string{"acc"}, Output: []string{"neg_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			provenanceValueInfo("acc_out", ir.TensorProto_FLOAT, 1),
			provenanceValueInfo("neg_out", ir.TensorProto_FLOAT, 1),
		},
	}
	return &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name:  "loop_carried_alias",
			Input: []*ir.ValueInfoProto{provenanceValueInfo("k", ir.TensorProto_FLOAT, 1)},
			Initializer: []*ir.TensorProto{
				{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{3}},
				{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
			},
			Node: []*ir.NodeProto{
				{
					OpType: "Loop",
					Input:  []string{"M", "cond", "k"},
					Output: []string{"final", "scan"},
					Attribute: []*ir.AttributeProto{
						{Name: "body", Type: ir.AttributeProto_GRAPH, G: loopCarriedAliasModelBody(body)},
					},
				},
			},
			Output: []*ir.ValueInfoProto{provenanceValueInfo("scan", ir.TensorProto_FLOAT, 3, 1)},
		},
	}
}

func loopCarriedAliasModelBody(g *ir.GraphProto) *ir.GraphProto { return g }

// A carried value that is a graph input is threaded symbolically, and must stay
// that way for every iteration. An Identity in the body aliases its input's
// gorgonia node, so asking gorgonia for that node's value hands back whatever
// SetInput bound — laundering the run's data into a constant that later
// iterations then fold. Iteration 0 would still track the input, so the failure
// is partial and quiet: the first slice of the scan output is right and the
// rest are stale.
func TestLoopCarriedGraphInputStaysSymbolicAcrossIterations(t *testing.T) {
	backend, model := buildProvenanceModel(t, loopCarriedAliasModel())

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]float32{1}))))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []float32{-1, -1, -1}, outs[0].Data())

	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(1), tensor.WithBacking([]float32{10}))))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, []float32{-10, -10, -10}, outs[0].Data(),
		"every iteration must track the new input, not just the first")
}

// ---------------------------------------------------------------------------
// Fold gates: a bound graph input must never be folded
// ---------------------------------------------------------------------------

// twoRunCase asserts that a single-output model returns fresh results on a
// second run with a different input — i.e. that nothing about run 1 was baked
// into the graph.
type twoRunCase struct {
	name     string
	model    *ir.ModelProto
	in1, in2 tensor.Tensor
	want1    interface{}
	want2    interface{}
}

func (c twoRunCase) run(t *testing.T) {
	t.Helper()
	backend, model := buildProvenanceModel(t, c.model)

	require.NoError(t, model.SetInput(0, c.in1))
	require.NoError(t, backend.Run())
	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, c.want1, outs[0].Data(), "first run")

	require.NoError(t, model.SetInput(0, c.in2))
	require.NoError(t, backend.Run())
	outs, err = model.GetOutputTensors()
	require.NoError(t, err)
	assert.Equal(t, c.want2, outs[0].Data(),
		"second run must not return the first run's answer")
}

func int64Init(name string, dims []int64, data []int64) *ir.TensorProto {
	return &ir.TensorProto{
		Name: name, DataType: int32(ir.TensorProto_INT64), Dims: dims, Int64Data: data,
	}
}

// gatherBoundIndicesModel gathers from an initializer with indices supplied at
// run time.
func gatherBoundIndicesModel() *ir.ModelProto {
	return g2model(&ir.GraphProto{
		Name:        "gather_bound_indices",
		Input:       []*ir.ValueInfoProto{provenanceValueInfo("idx", ir.TensorProto_INT64, 1)},
		Initializer: []*ir.TensorProto{int64Init("data", []int64{4}, []int64{10, 20, 30, 40})},
		Node: []*ir.NodeProto{
			{
				OpType: "Gather", Input: []string{"data", "idx"}, Output: []string{"picked"},
				Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
			},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("picked", ir.TensorProto_INT64, 1)},
	})
}

// castBoundInputModel casts a graph input to another integer type.
func castBoundInputModel() *ir.ModelProto {
	return g2model(&ir.GraphProto{
		Name:  "cast_bound_input",
		Input: []*ir.ValueInfoProto{provenanceValueInfo("x", ir.TensorProto_INT64, 2)},
		Node: []*ir.NodeProto{
			{
				OpType: "Cast", Input: []string{"x"}, Output: []string{"casted"},
				Attribute: []*ir.AttributeProto{
					{Name: "to", Type: ir.AttributeProto_INT, I: int64(ir.TensorProto_INT32)},
				},
			},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("casted", ir.TensorProto_INT32, 2)},
	})
}

// sliceBoundDataModel slices a graph input with constant bounds.
func sliceBoundDataModel() *ir.ModelProto {
	return g2model(&ir.GraphProto{
		Name:  "slice_bound_data",
		Input: []*ir.ValueInfoProto{provenanceValueInfo("data", ir.TensorProto_INT64, 4)},
		Initializer: []*ir.TensorProto{
			int64Init("starts", []int64{1}, []int64{1}),
			int64Init("ends", []int64{1}, []int64{3}),
		},
		Node: []*ir.NodeProto{
			{OpType: "Slice", Input: []string{"data", "starts", "ends"}, Output: []string{"sliced"}},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("sliced", ir.TensorProto_INT64, 2)},
	})
}

// Each of these folds a value that only exists at run time. Folding it computes
// the first run's answer and pins it into the graph, so every later run returns
// that same answer however the input changes.
func TestFoldGatesTrackEachRun(t *testing.T) {
	cases := []twoRunCase{
		{
			// pins gather.go's fold gate and its "bake the indices into
			// the op" path
			name:  "gather with run-time indices",
			model: gatherBoundIndicesModel(),
			in1:   tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{0})),
			in2:   tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{3})),
			want1: []int64{10},
			want2: []int64{40},
		},
		{
			// pins cast.go's fold gate
			name:  "cast of a run-time value",
			model: castBoundInputModel(),
			in1:   tensor.New(tensor.WithShape(2), tensor.WithBacking([]int64{1, 2})),
			in2:   tensor.New(tensor.WithShape(2), tensor.WithBacking([]int64{5, 6})),
			want1: []int32{1, 2},
			want2: []int32{5, 6},
		},
		{
			// pins slice.go's data fold gate
			name:  "slice of run-time data",
			model: sliceBoundDataModel(),
			in1:   tensor.New(tensor.WithShape(4), tensor.WithBacking([]int64{1, 2, 3, 4})),
			in2:   tensor.New(tensor.WithShape(4), tensor.WithBacking([]int64{5, 6, 7, 8})),
			want1: []int64{2, 3},
			want2: []int64{6, 7},
		},
	}
	for _, c := range cases {
		t.Run(c.name, c.run)
	}
}

// ---------------------------------------------------------------------------
// Conditional provenance marking: a static read never confers constness
// ---------------------------------------------------------------------------

// An operator may read a shape, a bound, or an axis list at build time whatever
// its provenance — there is no symbolic alternative for those. What it may not
// do is then claim the result is a compile-time constant, because a downstream
// fold gated on provenance would take that claim at face value and fold a value
// derived from run-time data.
//
// Each model below appends a plain initializer to the operator's output with a
// Concat. Concat folds only when every input is a proven constant, so it is the
// downstream consumer that would wrongly fold if the marking were unconditional.
func TestStaticReadDoesNotConferConstness(t *testing.T) {
	cases := []struct {
		name    string
		site    string
		model   *ir.ModelProto
		input   tensor.Tensor
		produce string
		want    interface{}
	}{
		{
			name: "expand with a run-time shape",
			site: "expand.go",
			model: g2model(&ir.GraphProto{
				Name:  "expand_bound_shape",
				Input: []*ir.ValueInfoProto{provenanceValueInfo("newshape", ir.TensorProto_INT64, 1)},
				Initializer: []*ir.TensorProto{
					int64Init("a", []int64{1}, []int64{7}),
					int64Init("tail", []int64{2}, []int64{8, 9}),
				},
				Node: []*ir.NodeProto{
					{OpType: "Expand", Input: []string{"a", "newshape"}, Output: []string{"expanded"}},
					{
						OpType: "Concat", Input: []string{"expanded", "tail"}, Output: []string{"joined"},
						Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
					},
				},
				Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_INT64, 5)},
			}),
			input:   tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{3})),
			produce: "expanded",
			want:    []int64{7, 7, 7, 8, 9},
		},
		{
			name: "slice with run-time bounds",
			site: "slice.go",
			model: g2model(&ir.GraphProto{
				Name:  "slice_bound_starts",
				Input: []*ir.ValueInfoProto{provenanceValueInfo("starts", ir.TensorProto_INT64, 1)},
				Initializer: []*ir.TensorProto{
					int64Init("data", []int64{4}, []int64{1, 2, 3, 4}),
					int64Init("ends", []int64{1}, []int64{3}),
					int64Init("tail", []int64{2}, []int64{8, 9}),
				},
				Node: []*ir.NodeProto{
					{OpType: "Slice", Input: []string{"data", "starts", "ends"}, Output: []string{"sliced"}},
					{
						OpType: "Concat", Input: []string{"sliced", "tail"}, Output: []string{"joined"},
						Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
					},
				},
				Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_INT64, 4)},
			}),
			input:   tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{1})),
			produce: "sliced",
			want:    []int64{2, 3, 8, 9},
		},
		{
			name: "constantofshape with a run-time shape",
			site: "constantofshape.go",
			model: g2model(&ir.GraphProto{
				Name:        "cos_bound_shape",
				Input:       []*ir.ValueInfoProto{provenanceValueInfo("shp", ir.TensorProto_INT64, 1)},
				Initializer: []*ir.TensorProto{int64Init("tail", []int64{2}, []int64{8, 9})},
				Node: []*ir.NodeProto{
					{
						OpType: "ConstantOfShape", Input: []string{"shp"}, Output: []string{"filled"},
						Attribute: []*ir.AttributeProto{
							{
								Name: "value", Type: ir.AttributeProto_TENSOR,
								T: int64Init("", []int64{1}, []int64{5}),
							},
						},
					},
					{
						OpType: "Concat", Input: []string{"filled", "tail"}, Output: []string{"joined"},
						Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
					},
				},
				Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_INT64, 5)},
			}),
			input:   tensor.New(tensor.WithShape(1), tensor.WithBacking([]int64{3})),
			produce: "filled",
			want:    []int64{5, 5, 5, 8, 9},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			backend, model := buildProvenanceModel(t, c.model)
			require.NoError(t, model.SetInput(0, c.input))
			require.NoError(t, backend.Run())

			outs, err := model.GetOutputTensors()
			require.NoError(t, err)
			require.Len(t, outs, 1)
			assert.Equal(t, c.want, outs[0].Data(), "the graph must still compute the right answer")

			assert.False(t, isConstNode(nodeNamed(t, backend, c.produce)),
				"%s: a value derived from a graph input is not a compile-time constant", c.site)
			assert.False(t, isConstNode(nodeNamed(t, backend, "joined")),
				"%s: the downstream fold must not have taken a false claim of constness", c.site)
		})
	}
}

// ---------------------------------------------------------------------------
// Aliases must carry provenance, or they sever the chain
// ---------------------------------------------------------------------------

// shapeChainThroughModel builds Shape(x) -> [passThrough] -> Split -> Concat ->
// Reshape. Every step is gated on provenance, so an alias in the middle that
// drops it breaks the whole chain and Reshape fails outright.
func shapeChainThroughModel(passThrough *ir.NodeProto, middle string) *ir.ModelProto {
	nodes := []*ir.NodeProto{{OpType: "Shape", Input: []string{"x"}, Output: []string{"shape"}}}
	nodes = append(nodes, passThrough)
	nodes = append(nodes,
		&ir.NodeProto{
			OpType: "Split", Input: []string{middle}, Output: []string{"lo", "hi"},
			Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
		},
		&ir.NodeProto{
			OpType: "Concat", Input: []string{"hi", "lo"}, Output: []string{"swapped"},
			Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
		},
		&ir.NodeProto{OpType: "Reshape", Input: []string{"x", "swapped"}, Output: []string{"y"}},
	)
	return g2model(&ir.GraphProto{
		Name:   "alias_chain",
		Input:  []*ir.ValueInfoProto{provenanceValueInfo("x", ir.TensorProto_FLOAT, 2, 3, 4, 5)},
		Node:   nodes,
		Output: []*ir.ValueInfoProto{provenanceValueInfo("y", ir.TensorProto_FLOAT, 4, 5, 2, 3)},
	})
}

// A Cast to the type the value already has, and a Concat of a single input, are
// both pure aliases: they hand back their input's gorgonia node untouched. An
// alias that forgets the input's provenance turns a constant into an apparent
// run-time value and severs every fold below it.
func TestAliasesCarryProvenance(t *testing.T) {
	cases := []struct {
		name   string
		site   string
		node   *ir.NodeProto
		middle string
	}{
		{
			name: "cast to the dtype the value already has",
			site: "cast.go",
			node: &ir.NodeProto{
				OpType: "Cast", Input: []string{"shape"}, Output: []string{"same"},
				Attribute: []*ir.AttributeProto{
					{Name: "to", Type: ir.AttributeProto_INT, I: int64(ir.TensorProto_INT64)},
				},
			},
			middle: "same",
		},
		{
			name: "concat of a single input",
			site: "concat.go",
			node: &ir.NodeProto{
				OpType: "Concat", Input: []string{"shape"}, Output: []string{"same"},
				Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
			},
			middle: "same",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			backend, model := buildProvenanceModel(t, shapeChainThroughModel(c.node, c.middle))
			require.NoError(t, model.SetInput(0, tensor.New(
				tensor.WithShape(2, 3, 4, 5),
				tensor.WithBacking(make([]float32, 2*3*4*5)))))
			require.NoError(t, backend.Run())

			outs, err := model.GetOutputTensors()
			require.NoError(t, err)
			assert.Equal(t, []int{4, 5, 2, 3}, []int(outs[0].Shape()))
			assert.True(t, isConstNode(nodeNamed(t, backend, "same")),
				"%s: an alias of a constant is a constant", c.site)
		})
	}
}

// ---------------------------------------------------------------------------
// A loop over constants produces a constant
// ---------------------------------------------------------------------------

func constantLoopModel() *ir.ModelProto {
	body := &ir.GraphProto{
		Name: "const_loop_body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			provenanceValueInfo("acc", ir.TensorProto_INT64, 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
			{OpType: "Add", Input: []string{"acc", "outer"}, Output: []string{"sum"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			provenanceValueInfo("sum", ir.TensorProto_INT64, 1),
		},
	}
	return g2model(&ir.GraphProto{
		Name: "constant_loop",
		Initializer: []*ir.TensorProto{
			{Name: "M", DataType: int32(ir.TensorProto_INT64), Int64Data: []int64{2}},
			{Name: "cond", DataType: int32(ir.TensorProto_BOOL), Int32Data: []int32{1}},
			int64Init("acc0", []int64{1}, []int64{1}),
			int64Init("outer", []int64{1}, []int64{2}),
			int64Init("tail", []int64{2}, []int64{8, 9}),
		},
		Node: []*ir.NodeProto{
			{
				OpType: "Loop", Input: []string{"M", "cond", "acc0"}, Output: []string{"final"},
				Attribute: []*ir.AttributeProto{
					{Name: "body", Type: ir.AttributeProto_GRAPH, G: body},
				},
			},
			{
				OpType: "Concat", Input: []string{"final", "tail"}, Output: []string{"joined"},
				Attribute: []*ir.AttributeProto{{Name: "axis", Type: ir.AttributeProto_INT, I: 0}},
			},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("joined", ir.TensorProto_INT64, 3)},
	})
}

// A loop whose trip count, carried value and captured values are all constants
// computes a constant. Publishing that provenance on the loop's own output is
// what lets the folds below it keep working; dropping it severs them for no
// reason.
func TestConstantLoopOutputStaysFoldable(t *testing.T) {
	backend, model := buildProvenanceModel(t, constantLoopModel())
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 1)
	assert.Equal(t, []int64{5, 8, 9}, outs[0].Data(), "1 + 2*2, then the tail")

	assert.True(t, isConstNode(nodeNamed(t, backend, "final")),
		"a loop over nothing but constants produces a constant")
	assert.True(t, isConstNode(nodeNamed(t, backend, "joined")),
		"and the fold below it must still fold")
}

// ---------------------------------------------------------------------------
// Broadcast axis limit
// ---------------------------------------------------------------------------

// gorgonia packs a broadcast pattern into a single byte, one nibble per
// operand, so it can only express axes 0..3. Building the pattern with a higher
// axis shifts the bit off the byte and the broadcast is silently dropped — the
// graph would run and return unbroadcast numbers. Newly reachable now that a
// run-time Expand takes the symbolic path at all.
func TestExpandBeyondGorgoniaBroadcastAxisLimitFailsLoudly(t *testing.T) {
	m := g2model(&ir.GraphProto{
		Name:  "expand_rank5",
		Input: []*ir.ValueInfoProto{provenanceValueInfo("x", ir.TensorProto_FLOAT, 2, 2, 2, 2, 1)},
		Initializer: []*ir.TensorProto{
			int64Init("newshape", []int64{5}, []int64{2, 2, 2, 2, 3}),
		},
		Node: []*ir.NodeProto{
			{OpType: "Expand", Input: []string{"x", "newshape"}, Output: []string{"expanded"}},
		},
		Output: []*ir.ValueInfoProto{provenanceValueInfo("expanded", ir.TensorProto_FLOAT, 2, 2, 2, 2, 3)},
	})

	backend, model := buildProvenanceModel(t, m)
	require.NoError(t, model.SetInput(0, tensor.New(
		tensor.WithShape(2, 2, 2, 2, 1),
		tensor.WithBacking(make([]float32, 16)))))

	err := backend.Run()
	require.Error(t, err, "a broadcast gorgonia cannot express must not be attempted")
	assert.Contains(t, err.Error(), "exceeds gorgonia's limit")
}
