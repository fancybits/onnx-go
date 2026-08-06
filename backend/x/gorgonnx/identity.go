package gorgonnx

import "github.com/owulveryck/onnx-go"

type identity struct{}

func init() {
	register("Identity", newIdentity)
}

func newIdentity() operator {
	return &identity{}
}

func (a *identity) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}

	// Identity aliases its input exactly, so it carries the input's value and
	// its provenance along with it. Passing the gorgonia node alone would leave
	// a constant looking like a run-time value to build-time consumers, and —
	// worse — leave a run-time value reachable through the alias's Value() with
	// nothing to mark it as one.
	n.gorgoniaNode = children[0].gorgoniaNode
	n.t = children[0].t
	n.constant = children[0].constant
	return err
}

func (a *identity) init(o onnx.Operation) error {
	return nil
}
