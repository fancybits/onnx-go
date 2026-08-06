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

// ONNX lets a node reference the same producer tensor for several of its
// inputs: Mul(x, x) is how tf2onnx emits the variance self-square of a
// LayerNorm, And(c, c) is how it emits a frozen while-loop condition. Both
// inputs are wired as edges between the same pair of nodes, differing only by
// their weight (the input ordinal), so the graph storage has to keep parallel
// edges for them to survive.

// dupInputGraph wires a single producer node into `arity` consecutive input
// slots of a consumer node.
func dupInputGraph(t *testing.T, arity int) (*Graph, *Node, *Node) {
	t.Helper()
	g := NewGraph()
	in := g.NewNode()
	g.AddNode(in)
	out := g.NewNode()
	g.AddNode(out)
	for i := 0; i < arity; i++ {
		g.SetWeightedEdge(g.NewWeightedEdge(out, in, float64(i)))
	}
	return g, in.(*Node), out.(*Node)
}

// applySelfBinaryOp runs a binary operation whose two inputs are the very same
// producer node, and returns the computed tensor.
func applySelfBinaryOp(t *testing.T, opName string, a tensor.Tensor) tensor.Tensor {
	t.Helper()
	g, in, out := dupInputGraph(t, 2)
	require.NoError(t, in.SetTensor(a))
	require.NoError(t, g.ApplyOperation(onnx.Operation{Name: opName}, out))
	require.NoError(t, g.Run())
	return out.GetTensor()
}

// TestDuplicateInputEdgesAreBothStored is the minimal reproduction: a simple
// graph keyed on (from, to) alone would keep only the last of the two edges.
func TestDuplicateInputEdgesAreBothStored(t *testing.T) {
	g, in, out := dupInputGraph(t, 2)

	children := getOrderedChildren(g.g, out)
	require.Len(t, children, 2, "both input slots must be reported")
	assert.Same(t, in, children[0])
	assert.Same(t, in, children[1])
}

// TestDuplicateInputOrdinalsArePreserved checks the ordinal each duplicated
// edge was wired at, which is what the variadic/optional-input operators
// (Slice and friends) key on.
func TestDuplicateInputOrdinalsArePreserved(t *testing.T) {
	g := NewGraph()
	a := g.NewNode()
	g.AddNode(a)
	b := g.NewNode()
	g.AddNode(b)
	out := g.NewNode()
	g.AddNode(out)
	// Wire out-of-order and with a gap at ordinal 2, as an optional input
	// omitted by a model would produce.
	g.SetWeightedEdge(g.NewWeightedEdge(out, a, 3))
	g.SetWeightedEdge(g.NewWeightedEdge(out, b, 1))
	g.SetWeightedEdge(g.NewWeightedEdge(out, a, 0))

	children := getOrderedChildren(g.g, out.(*Node))
	require.Len(t, children, 3)
	assert.Same(t, a, children[0])
	assert.Same(t, b, children[1])
	assert.Same(t, a, children[2])

	byIdx := getChildrenByInputIndex(g.g, out.(*Node))
	require.Len(t, byIdx, 3)
	assert.Same(t, a, byIdx[0])
	assert.Same(t, b, byIdx[1])
	assert.Nil(t, byIdx[2], "an omitted optional input must stay absent")
	assert.Same(t, a, byIdx[3])
}

// TestDuplicateInputSetIsIdempotent: SetWeightedEdge is a set operation, so
// re-setting the same (from, to, ordinal) triple must replace, not append.
func TestDuplicateInputSetIsIdempotent(t *testing.T) {
	g, in, out := dupInputGraph(t, 2)
	g.SetWeightedEdge(g.NewWeightedEdge(out, in, 0))

	assert.Len(t, getOrderedChildren(g.g, out), 2)
}

// Removing a node or an edge must take the parallel edges with it, so the
// stored inputs never outlive the topology they describe.
func TestDuplicateInputRemoval(t *testing.T) {
	t.Run("edge", func(t *testing.T) {
		g, in, out := dupInputGraph(t, 2)
		g.g.RemoveEdge(out.ID(), in.ID())
		assert.Empty(t, getOrderedChildren(g.g, out))
		assert.False(t, g.HasEdgeFromTo(out.ID(), in.ID()))
	})
	t.Run("node", func(t *testing.T) {
		g, in, out := dupInputGraph(t, 2)
		g.g.RemoveNode(in.ID())
		assert.Empty(t, getOrderedChildren(g.g, out))
		assert.Nil(t, g.Node(in.ID()))
	})
}

func TestDuplicateInputMulSquares(t *testing.T) {
	x := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6}))
	got := applySelfBinaryOp(t, "Mul", x)
	assert.Equal(t, []float32{1, 4, 9, 16, 25, 36}, got.Data())
}

func TestDuplicateInputSubIsZero(t *testing.T) {
	x := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float32{1, -2, 3}))
	got := applySelfBinaryOp(t, "Sub", x)
	assert.Equal(t, []float32{0, 0, 0}, got.Data())
}

func TestDuplicateInputAdd(t *testing.T) {
	x := tensor.New(tensor.WithShape(3), tensor.WithBacking([]float32{1, -2, 3}))
	got := applySelfBinaryOp(t, "Add", x)
	assert.Equal(t, []float32{2, -4, 6}, got.Data())
}

// The bigru artifact's frozen while-loop condition is And(c, c) on a 0-d bool.
func TestDuplicateInputAnd(t *testing.T) {
	c := tensor.New(tensor.WithShape(4), tensor.WithBacking([]bool{true, true, false, false}))
	got := applySelfBinaryOp(t, "And", c)
	assert.Equal(t, []bool{true, true, false, false}, got.Data())

	scalar := tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{true}))
	assert.Equal(t, true, applySelfBinaryOp(t, "And", scalar).Data())
}

// dupInputValueInfo builds a value_info of the given element type and shape.
func dupInputValueInfo(name string, elem ir.TensorProto_DataType, dims ...int64) *ir.ValueInfoProto {
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

// TestDuplicateInputDecodePath is the end-to-end case the bigru artifact hits:
// a model whose nodes name the same tensor twice must decode and run.
// Mul(x, x) is the LayerNorm self-square (16 occurrences in that artifact);
// And(c, c) is the frozen loop condition that failed with
// "bad arity for operation (have 1, want 2)".
func TestDuplicateInputDecodePath(t *testing.T) {
	m := &ir.ModelProto{
		IrVersion:   8,
		OpsetImport: []*ir.OperatorSetIdProto{{Version: 15}},
		Graph: &ir.GraphProto{
			Name: "dupinput",
			Initializer: []*ir.TensorProto{
				{
					Name: "x", DataType: int32(ir.TensorProto_FLOAT),
					Dims: []int64{4}, FloatData: []float32{1, 2, 3, 4},
				},
				{
					Name: "c", DataType: int32(ir.TensorProto_BOOL),
					Dims: []int64{4}, Int32Data: []int32{1, 1, 0, 0},
				},
			},
			Node: []*ir.NodeProto{
				{OpType: "Mul", Input: []string{"x", "x"}, Output: []string{"sq"}},
				{OpType: "Mul", Input: []string{"sq", "sq"}, Output: []string{"quad"}},
				{OpType: "And", Input: []string{"c", "c"}, Output: []string{"cc"}},
			},
			Output: []*ir.ValueInfoProto{
				dupInputValueInfo("quad", ir.TensorProto_FLOAT, 4),
				dupInputValueInfo("cc", ir.TensorProto_BOOL, 4),
			},
		},
	}

	raw, err := proto.Marshal(m)
	require.NoError(t, err)

	backend := NewGraph()
	model := onnx.NewModel(backend)
	require.NoError(t, model.UnmarshalBinary(raw))
	require.NoError(t, backend.Run())

	outs, err := model.GetOutputTensors()
	require.NoError(t, err)
	require.Len(t, outs, 2)
	assert.Equal(t, []float32{1, 16, 81, 256}, outs[0].Data(), "((x*x)*(x*x))")
	assert.Equal(t, []bool{true, true, false, false}, outs[1].Data(), "And(c, c)")
}
