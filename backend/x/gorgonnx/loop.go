package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop
//
// Static-unroll subset: the trip count must resolve to a compile-time
// constant, and the loop condition is ignored — the body is instantiated
// exactly M times. This matches tf2onnx while_loop output for Keras RNNs,
// where cond is the `i < M` bookkeeping and is true for exactly M
// iterations by construction. A non-constant trip count is a build-time
// error. Dynamic termination is deliberately unsupported: gorgonia ops
// return a single Value (Loop's N outputs would need stateful projection
// ops), and a dynamic trip count makes scan-output shapes unknowable at
// compile time, which TapeMachine's register preallocation cannot express.

func init() {
	register("Loop", newLoop)
}

type loop struct {
	body *onnx.Subgraph
}

func newLoop() operator {
	return &loop{}
}

func (l *loop) init(o onnx.Operation) error {
	b, ok := o.Attributes["body"].(*onnx.Subgraph)
	if !ok {
		return fmt.Errorf("loop: missing or invalid body subgraph attribute")
	}
	l.body = b
	return nil
}

func (l *loop) apply(g *Graph, ns ...*Node) error {
	children := getOrderedChildren(g.g, ns[0])
	if len(children) < 2 {
		return fmt.Errorf("loop: expected at least 2 inputs (M, cond), got %d", len(children))
	}

	// Name lookup over the parent graph, for resolving captured values.
	byName := make(map[string]*Node)
	it := g.g.Nodes()
	for it.Next() {
		pn := it.Node().(*Node)
		if pn.name != "" {
			byName[pn.name] = pn
		}
	}

	// A captured value carries no edge from the Loop node to its producer, so
	// the walk's readiness check — which only sees the explicit inputs — can
	// reach this operator before the producer has been built. Establish that
	// every capture resolves before doing any work, and ask to be retried
	// later if one does not. Doing this first also keeps the retry free of
	// side effects: nothing below has run yet.
	for _, name := range l.body.CapturedNames() {
		pn, ok := byName[name]
		if !ok {
			return fmt.Errorf("loop: captured value %q is not defined in the enclosing scope: %w", name, errNotReady)
		}
		if pn.gorgoniaNode == nil {
			return fmt.Errorf("loop: captured value %q is not built yet: %w", name, errNotReady)
		}
	}

	if !isConstNode(children[0]) {
		// A trip count supplied as a graph input carries a tensor too — a
		// placeholder, or whatever SetInput bound into it before the build —
		// so only its provenance tells the two apart.
		return fmt.Errorf("loop: trip count must be a constant initializer, not a graph input or a computed value (static unroll)")
	}
	mVals := tensorToInt64Slice(children[0].t)
	if len(mVals) != 1 || mVals[0] <= 0 {
		return fmt.Errorf("loop: trip count must be a positive compile-time constant (static unroll), got %v", mVals)
	}
	m := int(mVals[0])

	carried := children[2:]
	nCarried := len(carried)
	scanCount := len(ns) - nCarried
	if scanCount < 0 {
		return fmt.Errorf("loop: %d outputs but %d carried values", len(ns), nCarried)
	}

	carriedVals := make([]*gorgonia.Node, nCarried)
	carriedTs := make([]tensor.Tensor, nCarried)
	for i, c := range carried {
		if c.gorgoniaNode == nil {
			return fmt.Errorf("loop: carried input %d has no gorgonia node", i)
		}
		carriedVals[i] = c.gorgoniaNode
		// Only a genuine constant may be threaded into the body as a value:
		// the body's ops fold what they are given, and folding a graph
		// input's bound tensor would bake this run's data into the graph.
		// Everything else is threaded symbolically, through the gorgonia
		// node alone.
		if isConstNode(c) {
			carriedTs[i] = c.t
		}
	}

	scans := make([][]*gorgonia.Node, scanCount)
	for t := 0; t < m; t++ {
		child := NewGraph()
		child.exprgraph = g.exprgraph

		info, err := l.body.Decode(child)
		if err != nil {
			// %v for the same reason as the populateExprgraph call below: this
			// runs after the first iteration has already been emitted.
			return fmt.Errorf("loop: decoding body (iteration %d): %v", t, err)
		}
		if len(info.Inputs) != nCarried+2 {
			return fmt.Errorf("loop: body has %d inputs, want %d (iter, cond, %d carried)",
				len(info.Inputs), nCarried+2, nCarried)
		}
		if len(info.Outputs) != 1+nCarried+scanCount {
			return fmt.Errorf("loop: body has %d outputs, want %d (cond, %d carried, %d scan)",
				len(info.Outputs), 1+nCarried+scanCount, nCarried, scanCount)
		}

		// Seed the boundary before populating: the walk only materialises a
		// gorgonia node for leaves whose gorgoniaNode is still nil, so
		// pre-seeded boundary nodes are stitched straight into the enclosing
		// exprgraph.
		iterN, ok := info.Inputs[0].(*Node)
		if !ok {
			return fmt.Errorf("loop: body iteration-number input is not a gorgonnx node")
		}
		// iter and cond are 0-d scalars per the ONNX Loop body signature.
		iterT := tensor.New(tensor.WithShape(), tensor.WithBacking([]int64{int64(t)}))
		iterN.t = iterT
		iterN.MarkConst()
		iterN.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, iterT, gorgonia.WithName(getUniqNodeName("loop_iter")))

		condN, ok := info.Inputs[1].(*Node)
		if !ok {
			return fmt.Errorf("loop: body condition input is not a gorgonnx node")
		}
		condT := tensor.New(tensor.WithShape(), tensor.WithBacking([]bool{true}))
		condN.t = condT
		condN.MarkConst()
		condN.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, condT, gorgonia.WithName(getUniqNodeName("loop_cond")))

		for i := 0; i < nCarried; i++ {
			cn, ok := info.Inputs[2+i].(*Node)
			if !ok {
				return fmt.Errorf("loop: body carried input %d is not a gorgonnx node", i)
			}
			cn.gorgoniaNode = carriedVals[i]
			// Overwrite unconditionally: a body input declared with a type
			// arrives here holding a zero-filled placeholder, and leaving that
			// in place would let the body's ops fold zeros.
			cn.t, cn.constant = carriedTs[i], carriedTs[i] != nil
		}
		for name, capNode := range info.Captured {
			pn, ok := byName[name]
			if !ok || pn.gorgoniaNode == nil {
				return fmt.Errorf("loop: cannot resolve captured value %q in enclosing scope", name)
			}
			cn, ok := capNode.(*Node)
			if !ok {
				return fmt.Errorf("loop: captured value %q is not a gorgonnx node", name)
			}
			cn.gorgoniaNode = pn.gorgoniaNode
			// Only a constant crosses the boundary as a value. A parent node
			// that is not one carries either a zero-filled placeholder (the
			// enclosing graph's value_info gives intermediate nodes one) or a
			// tensor bound by SetInput for this run — folding either into the
			// body would silently compute the wrong thing. Overwrite in both
			// cases: the body's own declaration of the name leaves a
			// zero-filled placeholder here otherwise.
			if isConstNode(pn) {
				cn.t, cn.constant = pn.t, true
			} else {
				cn.t, cn.constant = nil, false
			}
		}

		if err := child.populateExprgraph(); err != nil {
			// Deliberately %v, not %w: only the capture pre-check above may
			// report errNotReady, because only it runs before any side effect.
			// By this point iterations 0..t have already been written into the
			// enclosing exprgraph, so letting a nested Loop's errNotReady
			// escape would have the walk retry this operator and unroll the
			// body a second time on top of the first.
			return fmt.Errorf("loop: building body (iteration %d): %v", t, err)
		}

		for i := 0; i < nCarried; i++ {
			outN, ok := info.Outputs[1+i].(*Node)
			if !ok || outN.gorgoniaNode == nil {
				return fmt.Errorf("loop: body carried output %d not built (iteration %d)", i, t)
			}
			carriedVals[i] = outN.gorgoniaNode
			// Thread the output's own provenance into the next iteration.
			//
			// Asking gorgonia for the node's value instead would launder a
			// run-time value into a constant. Every way a value *enters* the
			// body is gated above, but that is not enough: a body output can
			// carry a gorgonia value it is not entitled to. An Identity aliases
			// its input's node, so a carried value threaded symbolically —
			// a graph input, whose leaf holds whatever SetInput bound — is
			// reachable through the alias. And an operator that reads a static
			// requirement, such as ConstantOfShape, materialises a value node
			// from a shape that may itself have come from a graph input.
			// Node.constant is the only thing that distinguishes those.
			carriedTs[i] = nil
			if isConstNode(outN) {
				carriedTs[i] = outN.t
			}
		}
		for s := 0; s < scanCount; s++ {
			outN, ok := info.Outputs[1+nCarried+s].(*Node)
			if !ok || outN.gorgoniaNode == nil {
				return fmt.Errorf("loop: body scan output %d not built (iteration %d)", s, t)
			}
			scans[s] = append(scans[s], outN.gorgoniaNode)
		}
	}

	for i := 0; i < nCarried; i++ {
		ns[i].gorgoniaNode = carriedVals[i]
		// Publish the provenance the last iteration arrived at. A loop over
		// nothing but constants produces a constant, and dropping that would
		// sever every fold below the loop for no reason. carriedTs is already
		// gated on the body output's own provenance, so a non-nil entry here is
		// a value the loop is entitled to call constant.
		if carriedTs[i] != nil {
			ns[i].t = carriedTs[i]
			ns[i].constant = true
		}
	}
	for s := 0; s < scanCount; s++ {
		expanded := make([]*gorgonia.Node, m)
		for t, v := range scans[s] {
			newShape := append(tensor.Shape{1}, v.Shape().Clone()...)
			var err error
			expanded[t], err = gorgonia.Reshape(v, newShape)
			if err != nil {
				return fmt.Errorf("loop: reshaping scan output %d (iteration %d): %w", s, t, err)
			}
		}
		var err error
		ns[nCarried+s].gorgoniaNode, err = gorgonia.Concat(0, expanded...)
		if err != nil {
			return fmt.Errorf("loop: stacking scan output %d: %w", s, err)
		}
	}
	return nil
}
