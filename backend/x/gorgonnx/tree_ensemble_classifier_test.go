package gorgonnx

import (
	"math"
	"strings"
	"testing"

	"github.com/owulveryck/onnx-go"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// A two-tree forest over two features, small enough that the expected scores
// below are worked out by hand rather than copied from a runtime.
//
//	tree 0: x0 < 1.0 ? leaf +0.5 : leaf -0.25
//	tree 1: x1 < 0.0 ? leaf -1.0 : leaf +2.0
func tecBaseAttrs() map[string]interface{} {
	return map[string]interface{}{
		"nodes_treeids":      []int64{0, 0, 0, 1, 1, 1},
		"nodes_nodeids":      []int64{0, 1, 2, 0, 1, 2},
		"nodes_featureids":   []int64{0, 0, 0, 1, 0, 0},
		"nodes_values":       []float32{1.0, 0, 0, 0.0, 0, 0},
		"nodes_modes":        []string{"BRANCH_LT", "LEAF", "LEAF", "BRANCH_LT", "LEAF", "LEAF"},
		"nodes_truenodeids":  []int64{1, 0, 0, 1, 0, 0},
		"nodes_falsenodeids": []int64{2, 0, 0, 2, 0, 0},
		"class_treeids":      []int64{0, 0, 1, 1},
		"class_nodeids":      []int64{1, 2, 1, 2},
		"class_ids":          []int64{0, 0, 0, 0},
		"class_weights":      []float32{0.5, -0.25, -1.0, 2.0},
		"classlabels_int64s": []int64{0, 1},
		"post_transform":     "LOGISTIC",
	}
}

// tecHarness builds input -> TreeEnsembleClassifier -> (label, scores).
func tecHarness(t *testing.T, in tensor.Tensor, attrs map[string]interface{}) (tensor.Tensor, tensor.Tensor, error) {
	t.Helper()
	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	if err := input.(*Node).SetTensor(in); err != nil {
		t.Fatal(err)
	}
	outs := make([]graph.Node, 2)
	for i := range outs {
		o := g.NewNode()
		g.AddNode(o)
		g.SetWeightedEdge(g.NewWeightedEdge(o, input, 0))
		outs[i] = o
	}
	err := g.ApplyOperation(onnx.Operation{
		Name:       "TreeEnsembleClassifier",
		Domain:     "ai.onnx.ml",
		Attributes: attrs,
	}, outs...)
	if err != nil {
		return nil, nil, err
	}
	if err := g.Run(); err != nil {
		return nil, nil, err
	}
	return outs[0].(*Node).GetTensor(), outs[1].(*Node).GetTensor(), nil
}

func TestTreeEnsembleClassifierBinary(t *testing.T) {
	attrs := tecBaseAttrs()
	attrs["base_values"] = []float32{0.125}

	in := tensor.New(tensor.WithShape(4, 2), tensor.WithBacking([]float32{
		0.0, -1.0, // tree0 true (+0.5), tree1 true (-1.0)
		0.0, 1.0, // tree0 true (+0.5), tree1 false (+2.0)
		2.0, -1.0, // tree0 false (-0.25), tree1 true (-1.0)
		1.0, 0.0, // both on the threshold: BRANCH_LT sends both false
	}))
	wantScore := []float64{
		0.125 + 0.5 - 1.0,
		0.125 + 0.5 + 2.0,
		0.125 - 0.25 - 1.0,
		0.125 - 0.25 + 2.0,
	}

	label, scores, err := tecHarness(t, in, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := scores.Shape(), (tensor.Shape{4, 2}); !got.Eq(want) {
		t.Fatalf("scores shape %v, want %v", got, want)
	}
	got := scores.Data().([]float32)
	gotLabel := label.Data().([]int64)
	for i, s := range wantScore {
		p := 1 / (1 + math.Exp(-s))
		if d := math.Abs(float64(got[i*2+1]) - p); d > 1e-6 {
			t.Errorf("row %d p_class1 %v, want %v", i, got[i*2+1], p)
		}
		if d := math.Abs(float64(got[i*2]) - (1 - p)); d > 1e-6 {
			t.Errorf("row %d p_class0 %v, want %v", i, got[i*2], 1-p)
		}
		want := int64(0)
		if p > 0.5 {
			want = 1
		}
		if gotLabel[i] != want {
			t.Errorf("row %d label %d, want %d", i, gotLabel[i], want)
		}
	}
}

// A NaN feature takes the direction nodes_missing_value_tracks_true names,
// not the direction the comparison would produce.
func TestTreeEnsembleClassifierMissingValueRouting(t *testing.T) {
	nan := float32(math.NaN())
	in := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{nan, nan}))

	for _, tc := range []struct {
		name  string
		flag  int64
		score float64
	}{
		{"false child", 0, -0.25 + 2.0},
		{"true child", 1, 0.5 - 1.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			attrs := tecBaseAttrs()
			attrs["nodes_missing_value_tracks_true"] = []int64{tc.flag, 0, 0, tc.flag, 0, 0}
			_, scores, err := tecHarness(t, in, attrs)
			if err != nil {
				t.Fatal(err)
			}
			got := scores.Data().([]float32)
			want := 1 / (1 + math.Exp(-tc.score))
			if d := math.Abs(float64(got[1]) - want); d > 1e-6 {
				t.Errorf("p_class1 %v, want %v", got[1], want)
			}
		})
	}
}

func TestTreeEnsembleClassifierStringLabels(t *testing.T) {
	attrs := tecBaseAttrs()
	delete(attrs, "classlabels_int64s")
	attrs["classlabels_strings"] = []string{"negative", "positive"}

	in := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{
		0.0, 1.0, // score +2.5 -> class 1
		2.0, -1.0, // score -1.25 -> class 0
	}))
	label, _, err := tecHarness(t, in, attrs)
	if err != nil {
		t.Fatal(err)
	}
	got := label.Data().([]string)
	if want := []string{"positive", "negative"}; got[0] != want[0] || got[1] != want[1] {
		t.Errorf("labels %v, want %v", got, want)
	}
}

// The configurations onnx-go refuses, and the reason each is refused. A tree
// ensemble that routes or accumulates wrong returns a plausible number rather
// than an error, so each of these has to fail at graph-build time.
func TestTreeEnsembleClassifierRefusals(t *testing.T) {
	in := tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{0, 0}))

	for _, tc := range []struct {
		name string
		mut  func(map[string]interface{})
		want string
	}{
		{
			// onnxruntime and the onnx reference use different erf-inverse
			// approximations and both return NaN off [0, 1].
			name: "PROBIT",
			mut:  func(a map[string]interface{}) { a["post_transform"] = "PROBIT" },
			want: "post_transform",
		},
		{
			// With one weight column the two implementations take different
			// branches for every transform but LOGISTIC.
			name: "one weight column with SOFTMAX",
			mut:  func(a map[string]interface{}) { a["post_transform"] = "SOFTMAX" },
			want: "post_transform",
		},
		{
			// onnxruntime decides "binary" from the class count and drops the
			// second column; the reference keeps both.
			name: "two classes carrying two weight columns",
			mut: func(a map[string]interface{}) {
				a["class_treeids"] = []int64{0, 0, 0, 0, 1, 1, 1, 1}
				a["class_nodeids"] = []int64{1, 1, 2, 2, 1, 1, 2, 2}
				a["class_ids"] = []int64{0, 1, 0, 1, 0, 1, 0, 1}
				a["class_weights"] = []float32{0.5, 0.1, -0.25, 0.2, -1.0, 0.3, 2.0, 0.4}
			},
			want: "weight columns",
		},
		{
			name: "unknown split mode",
			mut: func(a map[string]interface{}) {
				a["nodes_modes"] = []string{"BRANCH_MEMBER", "LEAF", "LEAF", "BRANCH_LT", "LEAF", "LEAF"}
			},
			want: "split mode",
		},
		{
			name: "child pointer outside the tree",
			mut:  func(a map[string]interface{}) { a["nodes_truenodeids"] = []int64{7, 0, 0, 1, 0, 0} },
			want: "missing true child",
		},
		{
			name: "node reachable twice",
			mut:  func(a map[string]interface{}) { a["nodes_falsenodeids"] = []int64{1, 0, 0, 1, 0, 0} },
			want: "reachable more than once",
		},
		{
			name: "class weight on a split",
			mut:  func(a map[string]interface{}) { a["class_nodeids"] = []int64{0, 2, 1, 2} },
			want: "is a split, not a leaf",
		},
		{
			name: "attribute length mismatch",
			mut:  func(a map[string]interface{}) { a["nodes_values"] = []float32{1.0, 0, 0} },
			want: "nodes_values has 3 entries",
		},
		{
			name: "intercept per class mismatch",
			mut:  func(a map[string]interface{}) { a["base_values"] = []float32{0.1, 0.2, 0.3} },
			want: "base_values",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			attrs := tecBaseAttrs()
			tc.mut(attrs)
			_, _, err := tecHarness(t, in, attrs)
			if err == nil {
				t.Fatal("the configuration ran; it must be refused")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error is %q, want it to mention %q", err, tc.want)
			}
		})
	}
}

// A node from a domain the backend does not implement must not be dispatched
// on its bare name.
func TestOperatorDomainIsNotIgnored(t *testing.T) {
	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	if err := input.(*Node).SetTensor(tensor.New(tensor.WithShape(2), tensor.WithBacking([]float32{1, 2}))); err != nil {
		t.Fatal(err)
	}
	out := g.NewNode()
	g.AddNode(out)
	g.SetWeightedEdge(g.NewWeightedEdge(out, input, 0))

	if err := g.ApplyOperation(onnx.Operation{Name: "Relu", Domain: "com.example"}, out); err != nil {
		t.Fatal(err)
	}
	err := g.Run()
	if err == nil {
		t.Fatal("com.example::Relu ran as the default-domain Relu")
	}
	if !strings.Contains(err.Error(), "com.example::Relu") {
		t.Errorf("error is %q, want it to name com.example::Relu", err)
	}
}

func TestCheckOpsetRejectsUnknownDomain(t *testing.T) {
	g := NewGraph()
	if err := g.CheckOpset("", 15); err != nil {
		t.Errorf("default domain rejected: %v", err)
	}
	if err := g.CheckOpset(mlDomain, 1); err != nil {
		t.Errorf("ai.onnx.ml v1 rejected: %v", err)
	}
	if err := g.CheckOpset(mlDomain, 5); err == nil {
		t.Error("ai.onnx.ml v5 accepted; TreeEnsembleClassifier is deprecated there")
	}
	if err := g.CheckOpset("com.microsoft", 1); err == nil {
		t.Error("an unimplemented domain was accepted")
	}
}
