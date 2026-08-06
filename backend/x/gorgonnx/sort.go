package gorgonnx

import (
	"sort"

	"gonum.org/v1/gonum/graph"
)

// getOrderedChildren returns the children of the node n — its ONNX inputs —
// sorted by input ordinal. A child appears once per input slot it is wired to,
// so a node whose inputs name the same tensor twice (Mul(x, x)) yields that
// child twice.
func getOrderedChildren(g *weightedDirectedGraph, n *Node) []*Node {
	edges := g.outWeightedEdges(n.ID())
	nodes := make([]*Node, len(edges))
	for i, e := range edges {
		nodes[i] = e.To().(*Node)
	}

	return nodes
}

// getChildrenByInputIndex returns the children of the node n keyed by the ONNX
// input ordinal they are wired to. Use it instead of getOrderedChildren for
// operators with optional inputs, where an omitted input leaves a hole in the
// ordinals and positions in the ordered slice no longer match input numbers.
func getChildrenByInputIndex(g *weightedDirectedGraph, n *Node) map[int]*Node {
	edges := g.outWeightedEdges(n.ID())
	children := make(map[int]*Node, len(edges))
	for _, e := range edges {
		children[int(e.Weight())] = e.To().(*Node)
	}

	return children
}

func sortByWeight(edges []graph.WeightedEdge) {
	sort.Stable(byWeight(edges))
}

type byWeight []graph.WeightedEdge

func (w byWeight) Len() int           { return len(w) }
func (w byWeight) Swap(i, j int)      { w[i], w[j] = w[j], w[i] }
func (w byWeight) Less(i, j int) bool { return w[i].Weight() < w[j].Weight() }
