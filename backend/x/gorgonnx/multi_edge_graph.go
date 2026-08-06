package gorgonnx

import (
	"math"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
)

// weightedDirectedGraph holds the structure of a Graph.
//
// It is a simple.WeightedDirectedGraph — which stores at most one edge per
// (from, to) node pair — augmented with the list of every edge that has been
// set, bucketed by the ID of the node the edge starts from.
//
// The side table is what makes duplicated inputs work. An ONNX node may
// reference the same producer tensor for several of its inputs (Mul(x, x) for
// a square, And(c, c) for a frozen loop condition, ...); the decoder wires each
// input as one edge from the consumer to the producer, with the input ordinal
// carried as the edge weight. Those edges share their (from, to) pair, so the
// simple graph keeps only the last of them and the operator sees fewer inputs
// than it has. The bucket keeps them all, keyed by (to, weight), and
// getOrderedChildren reads the bucket instead of the simple graph adjacency.
//
// The embedded simple graph is still the source of truth for the topology
// (node set, reachability, root detection): it only ever needed to know that
// two nodes are connected, not how many times. Its From, To, Edge and
// WeightedEdge methods therefore still report a single edge per node pair —
// read the inputs of a node with outWeightedEdges (or the getOrderedChildren
// and getChildrenByInputIndex helpers) rather than with those.
type weightedDirectedGraph struct {
	*simple.WeightedDirectedGraph
	// outEdges holds the edges leaving each node, in the order they were set.
	// Within a bucket an edge is unique by (to node ID, weight).
	outEdges map[int64][]graph.WeightedEdge
}

func newWeightedDirectedGraph() *weightedDirectedGraph {
	return &weightedDirectedGraph{
		WeightedDirectedGraph: simple.NewWeightedDirectedGraph(math.MaxFloat64, -1),
		outEdges:              make(map[int64][]graph.WeightedEdge),
	}
}

// SetWeightedEdge adds a weighted edge from one node to another, adding the
// nodes if they are not already present. Unlike the simple graph it shadows,
// it keeps edges that share their (from, to) pair but differ by weight, so a
// node can reference the same producer for more than one of its inputs.
// Setting an edge that matches an existing one on both its (from, to) pair and
// its weight replaces that edge, keeping the call idempotent.
func (g *weightedDirectedGraph) SetWeightedEdge(e graph.WeightedEdge) {
	g.WeightedDirectedGraph.SetWeightedEdge(e)

	from := e.From().ID()
	for i, existing := range g.outEdges[from] {
		if existing.To().ID() == e.To().ID() && existing.Weight() == e.Weight() {
			g.outEdges[from][i] = e
			return
		}
	}
	g.outEdges[from] = append(g.outEdges[from], e)
}

// RemoveNode removes a node and all its edges from the graph.
func (g *weightedDirectedGraph) RemoveNode(id int64) {
	g.WeightedDirectedGraph.RemoveNode(id)
	delete(g.outEdges, id)
	for from, edges := range g.outEdges {
		kept := edges[:0]
		for _, e := range edges {
			if e.To().ID() != id {
				kept = append(kept, e)
			}
		}
		g.outEdges[from] = kept
	}
}

// RemoveEdge removes every edge running from uid to vid.
func (g *weightedDirectedGraph) RemoveEdge(uid, vid int64) {
	g.WeightedDirectedGraph.RemoveEdge(uid, vid)
	edges := g.outEdges[uid]
	kept := edges[:0]
	for _, e := range edges {
		if e.To().ID() != vid {
			kept = append(kept, e)
		}
	}
	g.outEdges[uid] = kept
}

// outWeightedEdges returns the edges leaving the node id, sorted by weight —
// that is, the inputs of that node in ONNX input order. The returned slice is
// a copy and is safe to keep.
func (g *weightedDirectedGraph) outWeightedEdges(id int64) []graph.WeightedEdge {
	edges := make([]graph.WeightedEdge, len(g.outEdges[id]))
	copy(edges, g.outEdges[id])
	sortByWeight(edges)
	return edges
}
