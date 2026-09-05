package gorgonnx

import (
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"
	"math"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html
//
// ai.onnx.ml::TreeEnsembleClassifier, over the configurations where the two
// available reference implementations — onnxruntime and onnx's own Python
// reference — compute the same thing. That is a real restriction, not a
// shortcut: ONNX ships no conformance node test for this operator (its two
// tree tests cover the newer ai.onnx.ml::TreeEnsemble, which deprecates this
// one and is a different operator), and across a 360-variant sweep the two
// implementations agreed on 110. Where they disagree there is nothing to be
// conformant to, so those configurations are refused rather than decided by
// coin flip. See the refusals in buildTreeEnsemble for which, and why.
//
// A tree ensemble that routes or accumulates slightly wrong returns a
// plausible number rather than an error, which is why the refusals are at
// graph-build time and the validation below is not lenient.

const mlDomain = "ai.onnx.ml"

// Operator set versions of ai.onnx.ml in which TreeEnsembleClassifier is
// defined. It is deprecated in v5 in favour of TreeEnsemble.
const (
	mlOpsetMin = 1
	mlOpsetMax = 4
)

func init() {
	registerDomain(mlDomain, "TreeEnsembleClassifier", newTreeEnsembleClassifier)
}

// Branch comparison modes. The ONNX-ML schema spells these as strings; they
// are interned to a byte so the hot loop compares an integer.
const (
	modeLeaf uint8 = iota
	modeLEQ
	modeLT
	modeGTE
	modeGT
	modeEQ
	modeNEQ
)

var branchModes = map[string]uint8{
	"LEAF":       modeLeaf,
	"BRANCH_LEQ": modeLEQ,
	"BRANCH_LT":  modeLT,
	"BRANCH_GTE": modeGTE,
	"BRANCH_GT":  modeGT,
	"BRANCH_EQ":  modeEQ,
	"BRANCH_NEQ": modeNEQ,
}

// Post-transforms. PROBIT is absent on purpose: onnxruntime and the onnx
// reference use different erf-inverse approximations and both return NaN off
// [0, 1], so the two disagree on every PROBIT variant.
const (
	postNone uint8 = iota
	postLogistic
	postSoftmax
	postSoftmaxZero
)

var postTransforms = map[string]uint8{
	"":             postNone,
	"NONE":         postNone,
	"LOGISTIC":     postLogistic,
	"SOFTMAX":      postSoftmax,
	"SOFTMAX_ZERO": postSoftmaxZero,
}

// treeNode is one node of the flattened forest. Children are indices into
// treeEnsemble.nodes, resolved once at build time, so evaluation never looks
// up a (tree, node) pair.
type treeNode struct {
	trueIdx  int32
	falseIdx int32
	feature  int32
	// leaf indexes this node's block in treeEnsemble.leafWeights, or is -1
	// for a split.
	leaf     int32
	value    float32
	mode     uint8
	missTrue bool
}

type treeEnsemble struct {
	nodes []treeNode
	roots []int32
	// leafWeights holds nClasses weights per leaf, laid out contiguously so
	// a leaf's whole contribution is one cache-friendly slice.
	leafWeights []float64
	// base is the per-class intercept, already widened to nClasses.
	base      []float64
	nClasses  int
	transform uint8
	// binary marks the shape a gradient boosted binary classifier exports:
	// every weight in one column, two class labels. Its scores are widened
	// to two columns before the post-transform, which is what makes it a
	// distinct code path rather than multiclass with n=2.
	binary bool

	labelsInt64  []int64
	labelsString []string

	// maxFeature is the highest feature index any split reads, used to
	// validate the input width once per call instead of once per node.
	maxFeature int32
	hash       uint32
}

func (e *treeEnsemble) numLabels() int {
	if e.labelsString != nil {
		return len(e.labelsString)
	}
	return len(e.labelsInt64)
}

// scores walks every tree for one feature row and writes the transformed
// class scores into out, which must have nClasses entries.
func (e *treeEnsemble) scores(row []float32, out []float64) {
	copy(out, e.base)
	for _, r := range e.roots {
		i := r
		for {
			n := &e.nodes[i]
			if n.mode == modeLeaf {
				w := e.leafWeights[int(n.leaf)*e.nClasses:]
				for c := 0; c < e.nClasses; c++ {
					out[c] += w[c]
				}
				break
			}
			v := row[n.feature]
			var takeTrue bool
			if v != v {
				// NaN: the node's own default direction decides. The spec
				// gives this rule for every mode; onnxruntime departs from
				// it under BRANCH_NEQ, where NaN != value is true in IEEE
				// and it descends the true branch regardless of the flag.
				takeTrue = n.missTrue
			} else {
				switch n.mode {
				case modeLEQ:
					takeTrue = v <= n.value
				case modeLT:
					takeTrue = v < n.value
				case modeGTE:
					takeTrue = v >= n.value
				case modeGT:
					takeTrue = v > n.value
				case modeEQ:
					takeTrue = v == n.value
				case modeNEQ:
					takeTrue = v != n.value
				}
			}
			if takeTrue {
				i = n.trueIdx
			} else {
				i = n.falseIdx
			}
		}
	}
	if e.binary {
		// One weight column widened to two: the accumulated score is the
		// class-1 score and its negation is class 0, so that an increasing
		// post-transform sends them to complementary probabilities.
		out[1] = out[0]
		out[0] = -out[1]
	}
	applyPostTransform(e.transform, out)
}

func applyPostTransform(kind uint8, v []float64) {
	switch kind {
	case postNone:
	case postLogistic:
		for i := range v {
			v[i] = logistic(v[i])
		}
	case postSoftmax:
		max := v[0]
		for _, x := range v[1:] {
			if x > max {
				max = x
			}
		}
		var s float64
		for i := range v {
			v[i] = math.Exp(v[i] - max)
			s += v[i]
		}
		for i := range v {
			v[i] /= s
		}
	case postSoftmaxZero:
		// Scores indistinguishable from zero are scaled rather than
		// exponentiated, so a class with no weight stays at zero instead of
		// picking up exp(-max). The 1e-7 cutoff and the all-0.5 fallback are
		// both part of the operator's definition.
		max := v[0]
		for _, x := range v[1:] {
			if x > max {
				max = x
			}
		}
		expNegMax := math.Exp(-max)
		var s float64
		for i := range v {
			if v[i] > 1e-7 || v[i] < -1e-7 {
				v[i] = math.Exp(v[i] - max)
			} else {
				v[i] *= expNegMax
			}
			s += v[i]
		}
		if s == 0 {
			for i := range v {
				v[i] = 0.5
			}
			return
		}
		for i := range v {
			v[i] /= s
		}
	}
}

// logistic in the form the operator's reference uses, which keeps exp from
// overflowing for large negative scores.
func logistic(x float64) float64 {
	v := 1 / (1 + math.Exp(-math.Abs(x)))
	if x < 0 {
		return 1 - v
	}
	return v
}

type treeEnsembleClassifier struct {
	ens *treeEnsemble
}

func newTreeEnsembleClassifier() operator {
	return &treeEnsembleClassifier{}
}

func (t *treeEnsembleClassifier) init(o onnx.Operation) error {
	ens, err := buildTreeEnsemble(o)
	if err != nil {
		return err
	}
	t.ens = ens
	return nil
}

// apply wires the two outputs of the operator. Output 0 is the predicted
// label, output 1 the class scores; the label is derived from the scores
// rather than from a second walk of the forest.
func (t *treeEnsembleClassifier) apply(g *Graph, ns ...*Node) error {
	if len(ns) == 0 || len(ns) > 2 {
		return fmt.Errorf("TreeEnsembleClassifier: want 1 or 2 outputs, have %d", len(ns))
	}
	children := getOrderedChildren(g.g, ns[0])
	if err := checkCondition(children, 1); err != nil {
		return err
	}
	input := children[0].gorgoniaNode
	if input.Dims() != 2 {
		return fmt.Errorf("TreeEnsembleClassifier: input must be a rank-2 tensor, have rank %d", input.Dims())
	}
	rows := input.Shape()[0]

	scores := gorgonia.NewUniqueNode(
		gorgonia.WithType(gorgonia.TensorType{Dims: 2, Of: tensor.Float32}),
		gorgonia.WithOp(&treeEnsembleScoreOp{ens: t.ens}),
		gorgonia.WithChildren(gorgonia.Nodes{input}),
		gorgonia.In(g.exprgraph),
		gorgonia.WithShape(rows, t.ens.nClasses),
	)

	labelDtype := tensor.Int64
	if t.ens.labelsString != nil {
		labelDtype = tensor.String
	}
	ns[0].gorgoniaNode = gorgonia.NewUniqueNode(
		gorgonia.WithType(gorgonia.TensorType{Dims: 1, Of: labelDtype}),
		gorgonia.WithOp(&treeEnsembleLabelOp{ens: t.ens}),
		gorgonia.WithChildren(gorgonia.Nodes{scores}),
		gorgonia.In(g.exprgraph),
		gorgonia.WithShape(rows),
	)
	if len(ns) == 2 {
		ns[1].gorgoniaNode = scores
	}
	return nil
}

// buildTreeEnsemble validates the operator's attributes and compiles them
// into a flattened forest. Everything that can be checked without a feature
// row is checked here so that evaluation is a pure walk.
func buildTreeEnsemble(o onnx.Operation) (*treeEnsemble, error) {
	notImpl := func(attr, msg string) error {
		return &onnx.ErrNotImplemented{
			Operator:      mlDomain + "::TreeEnsembleClassifier",
			AttributeName: attr,
			Message:       msg,
		}
	}

	ens := &treeEnsemble{maxFeature: -1}

	labelsInt, err := attrInts(o, "classlabels_int64s", false)
	if err != nil {
		return nil, err
	}
	labelsStr, err := attrStrings(o, "classlabels_strings", false)
	if err != nil {
		return nil, err
	}
	switch {
	case len(labelsInt) != 0 && len(labelsStr) != 0:
		return nil, notImpl("classlabels_int64s", "the model declares both int64 and string class labels")
	case len(labelsInt) != 0:
		ens.labelsInt64 = labelsInt
	case len(labelsStr) != 0:
		ens.labelsString = labelsStr
	default:
		return nil, fmt.Errorf("TreeEnsembleClassifier: the model declares no class labels")
	}
	ens.nClasses = ens.numLabels()
	if ens.nClasses < 2 {
		return nil, notImpl("classlabels_int64s", fmt.Sprintf("a single-class ensemble is not supported (have %d classes)", ens.nClasses))
	}

	post, err := attrString(o, "post_transform")
	if err != nil {
		return nil, err
	}
	transform, ok := postTransforms[post]
	if !ok {
		return nil, notImpl("post_transform", fmt.Sprintf("%q is not supported; onnxruntime and the onnx reference implementation disagree on it", post))
	}
	ens.transform = transform

	treeIDs, err := attrInts(o, "nodes_treeids", true)
	if err != nil {
		return nil, err
	}
	nodeIDs, err := attrInts(o, "nodes_nodeids", true)
	if err != nil {
		return nil, err
	}
	featureIDs, err := attrInts(o, "nodes_featureids", true)
	if err != nil {
		return nil, err
	}
	values, err := attrFloats(o, "nodes_values", true)
	if err != nil {
		return nil, err
	}
	trueIDs, err := attrInts(o, "nodes_truenodeids", true)
	if err != nil {
		return nil, err
	}
	falseIDs, err := attrInts(o, "nodes_falsenodeids", true)
	if err != nil {
		return nil, err
	}
	modeNames, err := attrStrings(o, "nodes_modes", true)
	if err != nil {
		return nil, err
	}
	missing, err := attrInts(o, "nodes_missing_value_tracks_true", false)
	if err != nil {
		return nil, err
	}

	n := len(nodeIDs)
	if n == 0 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: the ensemble has no nodes")
	}
	for name, l := range map[string]int{
		"nodes_treeids":      len(treeIDs),
		"nodes_featureids":   len(featureIDs),
		"nodes_values":       len(values),
		"nodes_truenodeids":  len(trueIDs),
		"nodes_falsenodeids": len(falseIDs),
		"nodes_modes":        len(modeNames),
	} {
		if l != n {
			return nil, fmt.Errorf("TreeEnsembleClassifier: %s has %d entries, nodes_nodeids has %d", name, l, n)
		}
	}
	if len(missing) != 0 && len(missing) != n {
		return nil, fmt.Errorf("TreeEnsembleClassifier: nodes_missing_value_tracks_true has %d entries, nodes_nodeids has %d", len(missing), n)
	}

	// Resolve (tree, node) pairs to flat indices once.
	type ref struct{ tree, node int64 }
	index := make(map[ref]int32, n)
	for i := 0; i < n; i++ {
		r := ref{treeIDs[i], nodeIDs[i]}
		if _, dup := index[r]; dup {
			return nil, fmt.Errorf("TreeEnsembleClassifier: tree %d declares node %d twice", r.tree, r.node)
		}
		index[r] = int32(i)
	}

	ens.nodes = make([]treeNode, n)
	seenTree := make(map[int64]bool, len(index))
	var treeOrder []int64
	var nLeaves int32
	for i := 0; i < n; i++ {
		mode, ok := branchModes[modeNames[i]]
		if !ok {
			return nil, notImpl("nodes_modes", fmt.Sprintf("split mode %q is not supported", modeNames[i]))
		}
		nd := treeNode{mode: mode, value: values[i], feature: -1, trueIdx: -1, falseIdx: -1, leaf: -1}
		if mode == modeLeaf {
			nd.leaf = nLeaves
			nLeaves++
		} else {
			if featureIDs[i] < 0 || featureIDs[i] > math.MaxInt32 {
				return nil, fmt.Errorf("TreeEnsembleClassifier: node %d has feature index %d", i, featureIDs[i])
			}
			nd.feature = int32(featureIDs[i])
			if nd.feature > ens.maxFeature {
				ens.maxFeature = nd.feature
			}
			tIdx, ok := index[ref{treeIDs[i], trueIDs[i]}]
			if !ok {
				return nil, fmt.Errorf("TreeEnsembleClassifier: tree %d node %d points at missing true child %d", treeIDs[i], nodeIDs[i], trueIDs[i])
			}
			fIdx, ok := index[ref{treeIDs[i], falseIDs[i]}]
			if !ok {
				return nil, fmt.Errorf("TreeEnsembleClassifier: tree %d node %d points at missing false child %d", treeIDs[i], nodeIDs[i], falseIDs[i])
			}
			nd.trueIdx, nd.falseIdx = tIdx, fIdx
			if len(missing) != 0 {
				nd.missTrue = missing[i] != 0
			}
		}
		ens.nodes[i] = nd
		if !seenTree[treeIDs[i]] {
			seenTree[treeIDs[i]] = true
			treeOrder = append(treeOrder, treeIDs[i])
		}
	}

	for _, t := range treeOrder {
		root, ok := index[ref{t, 0}]
		if !ok {
			return nil, fmt.Errorf("TreeEnsembleClassifier: tree %d has no node 0 to start from", t)
		}
		ens.roots = append(ens.roots, root)
	}

	classTrees, err := attrInts(o, "class_treeids", true)
	if err != nil {
		return nil, err
	}
	classNodes, err := attrInts(o, "class_nodeids", true)
	if err != nil {
		return nil, err
	}
	classIDs, err := attrInts(o, "class_ids", true)
	if err != nil {
		return nil, err
	}
	weights, err := attrFloats(o, "class_weights", true)
	if err != nil {
		return nil, err
	}
	if len(classTrees) != len(classNodes) || len(classTrees) != len(classIDs) || len(classTrees) != len(weights) {
		return nil, fmt.Errorf("TreeEnsembleClassifier: class_treeids/class_nodeids/class_ids/class_weights have lengths %d/%d/%d/%d",
			len(classTrees), len(classNodes), len(classIDs), len(weights))
	}

	distinct := make(map[int64]bool, ens.nClasses)
	for _, c := range classIDs {
		if c < 0 || int(c) >= ens.nClasses {
			return nil, fmt.Errorf("TreeEnsembleClassifier: class_ids names weight column %d, but the model declares %d classes", c, ens.nClasses)
		}
		distinct[c] = true
	}
	switch {
	case len(distinct) == 1 && ens.nClasses == 2:
		// The gradient boosted binary shape: one weight column, two labels.
		if !distinct[0] {
			return nil, notImpl("class_ids", "a single weight column must be column 0")
		}
		ens.binary = true
		if ens.transform != postLogistic {
			// With one weight column onnxruntime and the onnx reference take
			// different branches for every transform but LOGISTIC.
			return nil, notImpl("post_transform",
				fmt.Sprintf("a single weight column is only supported with LOGISTIC, have %q", post))
		}
	case len(distinct) == ens.nClasses && ens.nClasses >= 3:
		// Genuine multiclass: one weight column per class.
	default:
		// The remaining case is two classes carrying two weight columns.
		// onnxruntime decides "binary" from the class count and drops the
		// second column; the onnx reference decides it from the number of
		// distinct class_ids and keeps both, returning scores that do not
		// sum to one. There is no third opinion to break the tie.
		return nil, notImpl("class_ids",
			fmt.Sprintf("%d classes carrying %d weight columns is not supported; onnxruntime and the onnx reference implementation disagree on it",
				ens.nClasses, len(distinct)))
	}

	base, err := attrFloats(o, "base_values", false)
	if err != nil {
		return nil, err
	}
	ens.base = make([]float64, ens.nClasses)
	switch {
	case len(base) == 0:
	case ens.binary && len(base) == 1:
		ens.base[0] = float64(base[0])
	case !ens.binary && len(base) == ens.nClasses:
		for i, b := range base {
			ens.base[i] = float64(b)
		}
	default:
		return nil, notImpl("base_values", fmt.Sprintf("want one intercept per class (%d) or none, have %d", ens.nClasses, len(base)))
	}

	ens.leafWeights = make([]float64, int(nLeaves)*ens.nClasses)
	for i := range classIDs {
		idx, ok := index[ref{classTrees[i], classNodes[i]}]
		if !ok {
			return nil, fmt.Errorf("TreeEnsembleClassifier: class weight %d names tree %d node %d, which the ensemble does not declare", i, classTrees[i], classNodes[i])
		}
		nd := &ens.nodes[idx]
		if nd.mode != modeLeaf {
			return nil, fmt.Errorf("TreeEnsembleClassifier: class weight %d names tree %d node %d, which is a split, not a leaf", i, classTrees[i], classNodes[i])
		}
		ens.leafWeights[int(nd.leaf)*ens.nClasses+int(classIDs[i])] += float64(weights[i])
	}

	if err := checkTreeStructure(ens); err != nil {
		return nil, err
	}
	ens.hash = hashTreeEnsemble(ens)
	return ens, nil
}

// checkTreeStructure walks every tree once and rejects a forest that reaches
// any node twice, which covers both a cycle and a node with two parents.
// Without it a malformed model turns scores into an infinite loop at
// inference time.
func checkTreeStructure(e *treeEnsemble) error {
	seen := make([]bool, len(e.nodes))
	stack := make([]int32, 0, 64)
	for _, root := range e.roots {
		stack = append(stack[:0], root)
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if seen[i] {
				return fmt.Errorf("TreeEnsembleClassifier: node %d is reachable more than once, so the ensemble is not a forest", i)
			}
			seen[i] = true
			n := &e.nodes[i]
			if n.mode == modeLeaf {
				continue
			}
			stack = append(stack, n.trueIdx, n.falseIdx)
		}
	}
	return nil
}

func hashTreeEnsemble(e *treeEnsemble) uint32 {
	h := fnv.New32a()
	binary.Write(h, binary.LittleEndian, []byte("ai.onnx.ml::TreeEnsembleClassifier"))
	binary.Write(h, binary.LittleEndian, e.base)
	binary.Write(h, binary.LittleEndian, e.leafWeights)
	binary.Write(h, binary.LittleEndian, e.roots)
	binary.Write(h, binary.LittleEndian, int64(e.nClasses))
	binary.Write(h, binary.LittleEndian, e.transform)
	binary.Write(h, binary.LittleEndian, e.binary)
	binary.Write(h, binary.LittleEndian, e.labelsInt64)
	for _, s := range e.labelsString {
		binary.Write(h, binary.LittleEndian, []byte(s))
	}
	for i := range e.nodes {
		n := &e.nodes[i]
		binary.Write(h, binary.LittleEndian, n.trueIdx)
		binary.Write(h, binary.LittleEndian, n.falseIdx)
		binary.Write(h, binary.LittleEndian, n.feature)
		binary.Write(h, binary.LittleEndian, n.leaf)
		binary.Write(h, binary.LittleEndian, n.value)
		binary.Write(h, binary.LittleEndian, n.mode)
		binary.Write(h, binary.LittleEndian, n.missTrue)
	}
	return h.Sum32()
}

// Attribute readers. The decoder hands INTS through as []int64, FLOATS as
// []float32, STRINGS as []string and STRING as string.

func attrInts(o onnx.Operation, name string, required bool) ([]int64, error) {
	v, ok := o.Attributes[name]
	if !ok {
		if required {
			return nil, fmt.Errorf("TreeEnsembleClassifier: required attribute %q is missing", name)
		}
		return nil, nil
	}
	s, ok := v.([]int64)
	if !ok {
		return nil, fmt.Errorf("TreeEnsembleClassifier: attribute %q is %T, want []int64", name, v)
	}
	return s, nil
}

func attrFloats(o onnx.Operation, name string, required bool) ([]float32, error) {
	v, ok := o.Attributes[name]
	if !ok {
		if required {
			return nil, fmt.Errorf("TreeEnsembleClassifier: required attribute %q is missing", name)
		}
		return nil, nil
	}
	s, ok := v.([]float32)
	if !ok {
		return nil, fmt.Errorf("TreeEnsembleClassifier: attribute %q is %T, want []float32", name, v)
	}
	return s, nil
}

func attrStrings(o onnx.Operation, name string, required bool) ([]string, error) {
	v, ok := o.Attributes[name]
	if !ok {
		if required {
			return nil, fmt.Errorf("TreeEnsembleClassifier: required attribute %q is missing", name)
		}
		return nil, nil
	}
	s, ok := v.([]string)
	if !ok {
		return nil, fmt.Errorf("TreeEnsembleClassifier: attribute %q is %T, want []string", name, v)
	}
	return s, nil
}

func attrString(o onnx.Operation, name string) (string, error) {
	v, ok := o.Attributes[name]
	if !ok {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("TreeEnsembleClassifier: attribute %q is %T, want string", name, v)
	}
	return s, nil
}

// treeEnsembleScoreOp produces the (N, nClasses) score tensor.
type treeEnsembleScoreOp struct {
	ens *treeEnsemble
}

func (op *treeEnsembleScoreOp) Arity() int { return 1 }

func (op *treeEnsembleScoreOp) Type() hm.Type {
	in := gorgonia.TensorType{Dims: 2, Of: tensor.Float32}
	out := gorgonia.TensorType{Dims: 2, Of: tensor.Float32}
	return hm.NewFnType(in, out)
}

func (op *treeEnsembleScoreOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if len(inputs) != 1 || inputs[0] == nil {
		return nil, fmt.Errorf("TreeEnsembleClassifier: infershape needs one input shape")
	}
	s, ok := inputs[0].(tensor.Shape)
	if !ok || len(s) != 2 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: input must be rank 2, have %v", inputs[0])
	}
	return tensor.Shape{s[0], op.ens.nClasses}, nil
}

func (op *treeEnsembleScoreOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: expected 1 input, got %d", len(inputs))
	}
	in, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("TreeEnsembleClassifier: only dense tensors are supported")
	}
	if in.Dtype() != tensor.Float32 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: input dtype is %v, want float32", in.Dtype())
	}
	shape := in.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: input must be rank 2, have %v", shape)
	}
	rows, cols := shape[0], shape[1]
	if int(op.ens.maxFeature) >= cols {
		return nil, fmt.Errorf("TreeEnsembleClassifier: the ensemble splits on feature %d but the input has %d columns", op.ens.maxFeature, cols)
	}
	nc := op.ens.nClasses
	data := in.Float32s()
	out := make([]float32, rows*nc)
	acc := make([]float64, nc)
	for r := 0; r < rows; r++ {
		op.ens.scores(data[r*cols:(r+1)*cols], acc)
		for c := 0; c < nc; c++ {
			out[r*nc+c] = float32(acc[c])
		}
	}
	return tensor.New(tensor.WithShape(rows, nc), tensor.WithBacking(out)), nil
}

func (op *treeEnsembleScoreOp) ReturnsPtr() bool     { return false }
func (op *treeEnsembleScoreOp) CallsExtern() bool    { return false }
func (op *treeEnsembleScoreOp) OverwritesInput() int { return -1 }

func (op *treeEnsembleScoreOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("TreeEnsembleClassifier/scores"))
	binary.Write(h, binary.LittleEndian, op.ens.hash)
}

func (op *treeEnsembleScoreOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *treeEnsembleScoreOp) String() string { return "TreeEnsembleClassifierScores" }

// treeEnsembleLabelOp turns the score tensor into the predicted label.
type treeEnsembleLabelOp struct {
	ens *treeEnsemble
}

func (op *treeEnsembleLabelOp) Arity() int { return 1 }

func (op *treeEnsembleLabelOp) Type() hm.Type {
	in := gorgonia.TensorType{Dims: 2, Of: tensor.Float32}
	of := tensor.Dtype(tensor.Int64)
	if op.ens.labelsString != nil {
		of = tensor.String
	}
	out := gorgonia.TensorType{Dims: 1, Of: of}
	return hm.NewFnType(in, out)
}

func (op *treeEnsembleLabelOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if len(inputs) != 1 || inputs[0] == nil {
		return nil, fmt.Errorf("TreeEnsembleClassifier: label infershape needs one input shape")
	}
	s, ok := inputs[0].(tensor.Shape)
	if !ok || len(s) != 2 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: label input must be rank 2, have %v", inputs[0])
	}
	return tensor.Shape{s[0]}, nil
}

func (op *treeEnsembleLabelOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: label expected 1 input, got %d", len(inputs))
	}
	in, ok := inputs[0].(*tensor.Dense)
	if !ok || in.Dtype() != tensor.Float32 {
		return nil, fmt.Errorf("TreeEnsembleClassifier: label input must be a dense float32 tensor")
	}
	shape := in.Shape()
	nc := op.ens.nClasses
	if len(shape) != 2 || shape[1] != nc {
		return nil, fmt.Errorf("TreeEnsembleClassifier: label input has shape %v, want (N, %d)", shape, nc)
	}
	rows := shape[0]
	data := in.Float32s()
	best := make([]int, rows)
	for r := 0; r < rows; r++ {
		row := data[r*nc : (r+1)*nc]
		b := 0
		for c := 1; c < nc; c++ {
			if row[c] > row[b] {
				b = c
			}
		}
		best[r] = b
	}
	if op.ens.labelsString != nil {
		out := make([]string, rows)
		for r, b := range best {
			out[r] = op.ens.labelsString[b]
		}
		return tensor.New(tensor.WithShape(rows), tensor.WithBacking(out)), nil
	}
	out := make([]int64, rows)
	for r, b := range best {
		out[r] = op.ens.labelsInt64[b]
	}
	return tensor.New(tensor.WithShape(rows), tensor.WithBacking(out)), nil
}

func (op *treeEnsembleLabelOp) ReturnsPtr() bool     { return false }
func (op *treeEnsembleLabelOp) CallsExtern() bool    { return false }
func (op *treeEnsembleLabelOp) OverwritesInput() int { return -1 }

func (op *treeEnsembleLabelOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("TreeEnsembleClassifier/label"))
	binary.Write(h, binary.LittleEndian, op.ens.hash)
}

func (op *treeEnsembleLabelOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *treeEnsembleLabelOp) String() string { return "TreeEnsembleClassifierLabel" }
