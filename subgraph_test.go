package onnx

import (
	"testing"

	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func float32ValueInfo(name string, dims ...int64) *ir.ValueInfoProto {
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

// A minimal body: inputs (iter, cond, acc); one node Add(acc, outer) -> sum
// where "outer" is captured from the enclosing scope; outputs (cond, sum).
func testBodyGraph() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "body",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			float32ValueInfo("acc", 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Add", Input: []string{"acc", "outer"}, Output: []string{"sum"}},
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			float32ValueInfo("sum", 1),
		},
	}
}

func TestSubgraphDecode(t *testing.T) {
	sg := &Subgraph{g: testBodyGraph()}
	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)

	require.Len(t, info.Inputs, 3)
	require.Len(t, info.Outputs, 2)
	assert.Len(t, info.Captured, 1)
	_, ok := info.Captured["outer"]
	assert.True(t, ok, "outer should be detected as captured")

	// Outputs must resolve to the nodes produced by the body's ops.
	require.NotNil(t, info.Outputs[1])
}

// A body where an output shares its name with an input: ONNX allows a
// Loop/If body to return one of its own inputs unchanged (a pass-through
// carried value). Decode must resolve the output to the exact same node
// as the input rather than creating a duplicate.
func testPassthroughBodyGraph() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "body_passthrough",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			float32ValueInfo("w", 1),
		},
		Node: []*ir.NodeProto{
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			float32ValueInfo("w", 1),
		},
	}
}

func TestSubgraphDecode_PassthroughOutputName(t *testing.T) {
	sg := &Subgraph{g: testPassthroughBodyGraph()}
	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)

	require.Len(t, info.Inputs, 3)
	require.Len(t, info.Outputs, 2)

	wInput := info.Inputs[2]
	wOutput := info.Outputs[1]
	require.NotNil(t, wInput)
	require.NotNil(t, wOutput)
	assert.Equal(t, wInput.ID(), wOutput.ID(),
		"pass-through output %q should reuse the input node, not duplicate it", "w")

	// Exactly one node per distinct name (iter, cond_in, w, cond_out) should
	// exist in the backend: no duplicate node was created for "w".
	assert.Equal(t, 4, backend.g.Nodes().Len(),
		"expected exactly one node per distinct body name, found a duplicate")
}

// A body containing a zero-input node (e.g. Constant). decoder.go
// synthesizes a fake "<node.Name>/input" name for it during
// applyGraphNodeOperations. That synthesized name must not be
// misclassified as captured: it does not refer to the enclosing scope.
func testInputlessNodeBodyGraph() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "body_inputless_node",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			float32ValueInfo("acc", 1),
		},
		Node: []*ir.NodeProto{
			{
				Name:   "const_node",
				OpType: "Constant",
				Output: []string{"c"},
			},
			{OpType: "Add", Input: []string{"acc", "c"}, Output: []string{"sum"}},
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			float32ValueInfo("sum", 1),
		},
	}
}

func TestSubgraphDecode_InputlessNodeNotCaptured(t *testing.T) {
	sg := &Subgraph{g: testInputlessNodeBodyGraph()}
	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)

	assert.Empty(t, info.Captured, "a zero-input node's synthesized fake input name must not be captured")
	_, ok := info.Captured["const_node/input"]
	assert.False(t, ok, "the decoder-synthesized fake input name must not leak into Captured")
}

// A body with a value_info entry that no node references. Such entries are
// declarations of intermediate shapes/types, not captured values, so they
// must not appear in Captured even though they aren't in `produced`.
func testUnreferencedValueInfoBodyGraph() *ir.GraphProto {
	g := testBodyGraph()
	g.ValueInfo = append(g.ValueInfo, float32ValueInfo("unused_value_info", 1))
	return g
}

func TestSubgraphDecode_UnreferencedValueInfoNotCaptured(t *testing.T) {
	sg := &Subgraph{g: testUnreferencedValueInfoBodyGraph()}
	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)

	// Only "outer" (referenced by the body's Add node) should be captured;
	// the unreferenced value_info entry must not leak in.
	require.Len(t, info.Captured, 1)
	_, ok := info.Captured["outer"]
	assert.True(t, ok, "outer should still be detected as captured")
	_, ok = info.Captured["unused_value_info"]
	assert.False(t, ok, "an unreferenced value_info entry must not appear in Captured")
}

// A body node with an omitted optional input: ONNX spells the omission as the
// empty name. It refers to nothing, so it must not be reported as a value
// captured from the enclosing scope.
func testOmittedOptionalInputBodyGraph() *ir.GraphProto {
	return &ir.GraphProto{
		Name: "body_omitted_optional",
		Input: []*ir.ValueInfoProto{
			{Name: "iter"},
			{Name: "cond_in"},
			float32ValueInfo("acc", 4),
		},
		Node: []*ir.NodeProto{
			// Slice(data, starts, ends, axes, steps) with axes omitted.
			{OpType: "Slice", Input: []string{"acc", "starts", "ends", "", "steps"}, Output: []string{"sum"}},
			{OpType: "Identity", Input: []string{"cond_in"}, Output: []string{"cond_out"}},
		},
		Output: []*ir.ValueInfoProto{
			{Name: "cond_out"},
			float32ValueInfo("sum", 2),
		},
	}
}

func TestSubgraphDecode_OmittedOptionalInputNotCaptured(t *testing.T) {
	sg := &Subgraph{g: testOmittedOptionalInputBodyGraph()}
	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)

	_, ok := info.Captured[""]
	assert.False(t, ok, "an omitted optional input must not appear in Captured")
	assert.NotContains(t, sg.CapturedNames(), "")
	// The real captures are still reported.
	assert.ElementsMatch(t, []string{"starts", "ends", "steps"}, sg.CapturedNames())
}

// A body output that names a value the body does not produce is a use of an
// enclosing-scope value, exactly as a node input would be.
func testOuterNamedOutputBodyGraph() *ir.GraphProto {
	g := testBodyGraph()
	g.Output = append(g.Output, float32ValueInfo("extra", 1))
	return g
}

func TestSubgraphDecode_OuterNamedOutputIsCaptured(t *testing.T) {
	sg := &Subgraph{g: testOuterNamedOutputBodyGraph()}
	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)

	require.Len(t, info.Outputs, 3)
	captured, ok := info.Captured["extra"]
	require.True(t, ok, "a body output naming an enclosing-scope value should be captured")
	assert.Equal(t, info.Outputs[2].ID(), captured.ID(),
		"the captured node and the reported output must be the same node")
	assert.ElementsMatch(t, []string{"outer", "extra"}, sg.CapturedNames())
}

// CapturedNames must agree with what Decode reports, and must be derivable
// before Decode ever runs.
func TestSubgraphCapturedNamesBeforeDecode(t *testing.T) {
	sg := &Subgraph{g: testBodyGraph()}
	assert.Equal(t, []string{"outer"}, sg.CapturedNames())

	backend := newTestBackend()
	info, err := sg.Decode(backend)
	require.NoError(t, err)
	require.Len(t, info.Captured, 1)
	assert.Equal(t, []string{"outer"}, sg.CapturedNames(),
		"decoding must not change the reported captures")
}

func TestGraphAttributePassthrough(t *testing.T) {
	attr := &ir.AttributeProto{
		Name: "body",
		Type: ir.AttributeProto_GRAPH,
		G:    testBodyGraph(),
	}
	v, err := toOperationAttribute(attr)
	require.NoError(t, err)
	sg, ok := v.(*Subgraph)
	require.True(t, ok, "GRAPH attribute should decode to *Subgraph, got %T", v)
	assert.NotNil(t, sg.g)
}
