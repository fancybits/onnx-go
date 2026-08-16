package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Node is compatible with graph.Node and onnx.DataCarrier
type Node struct {
	id        int64
	t         tensor.Tensor
	operation *onnx.Operation
	name      string
	// constant reports that t is a compile-time constant (an initializer, or
	// a value the backend itself created from one) rather than a placeholder
	// that only holds a meaningful value at run time. A graph input carries a
	// tensor too — possibly one already bound by SetInput before the graph is
	// built — so the presence of t alone says nothing about provenance.
	constant bool
	// gorgoniaNode stores a pointer to the node of the exprgraph
	gorgoniaNode *gorgonia.Node
}

// ID to fulfill the graph.Node interface
func (n *Node) ID() int64 {
	return n.id
}

// SetTensor assign the tensor N to the underlying node
func (n *Node) SetTensor(t tensor.Tensor) error {
	n.t = t
	if n.gorgoniaNode == nil {
		return nil
	}

	// gorgonia.Let rejects a differently shaped value, and the caller should
	// not have to Reset by hand to feed a new batch size. Record it and leave
	// the exprgraph stale; Run rebuilds before executing.
	//
	// Only the batch dimension may move. A tensor that differs anywhere else
	// is a caller mistake, not a rebatch, and must still be reported here --
	// otherwise a transposed or wrongly-laid-out input is accepted silently
	// and fails later inside some operator, or not at all.
	if batchOnlyReshape(n.gorgoniaNode.Shape(), t.Shape()) {
		return nil
	}

	return gorgonia.Let(n.gorgoniaNode, t)
}

// shapeChanged reports that this node carries a tensor the built graph cannot
// accept.
func (n *Node) shapeChanged() bool {
	return n.gorgoniaNode != nil && n.t != nil && batchOnlyReshape(n.gorgoniaNode.Shape(), n.t.Shape())
}

// batchOnlyReshape reports whether want and got differ only in their leading
// dimension. Equal shapes are not a reshape.
func batchOnlyReshape(want, got tensor.Shape) bool {
	if len(want) != len(got) || len(want) == 0 || want.Eq(got) {
		return false
	}

	for i := 1; i < len(want); i++ {
		if want[i] != got[i] {
			return false
		}
	}

	return true
}

// GetTensor value from the node
func (n *Node) GetTensor() tensor.Tensor {
	return n.t
}

// MarkConst records that the tensor carried by this node is a compile-time
// constant. It fulfils the onnx.ConstMarker interface.
func (n *Node) MarkConst() {
	n.constant = true
}

// isConstNode reports whether n carries a compile-time constant tensor, i.e.
// one that is safe to fold into the expression graph.
func isConstNode(n *Node) bool {
	return n != nil && n.constant && n.t != nil
}

// GetName get the name of the node
func (n *Node) GetName() string {
	return n.name
}

// SetName set the name of the node
func (n *Node) SetName(name string) {
	n.name = name
}
