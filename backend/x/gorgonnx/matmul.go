package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

type matMul struct{}

func init() {
	register("MatMul", newMatMul)
}

func newMatMul() operator {
	return &matMul{}
}

func (a *matMul) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	aNode := children[0].gorgoniaNode
	bNode := children[1].gorgoniaNode
	aDims := len(aNode.Shape())
	bDims := len(bNode.Shape())

	// Handle different dimension cases
	switch {
	case aDims <= 2 && bDims <= 2:
		// Standard 2D matrix multiplication
		n.gorgoniaNode, err = gorgonia.Mul(aNode, bNode)
	case aDims == 3 && bDims == 3:
		// Batched matrix multiplication with broadcasting support
		aShape := aNode.Shape()
		bShape := bNode.Shape()
		aBatch := aShape[0]
		bBatch := bShape[0]

		if aBatch != bBatch && (aBatch == 1 || bBatch == 1) {
			// Need to broadcast - ONNX allows batch dim of 1 to be broadcast
			var leftAxes, rightAxes []byte
			if aBatch == 1 && bBatch > 1 {
				leftAxes = []byte{0} // broadcast A's batch dimension
			} else if bBatch == 1 && aBatch > 1 {
				rightAxes = []byte{0} // broadcast B's batch dimension
			}
			pattern := gorgonia.NewBroadcastPattern(leftAxes, rightAxes)
			aNode, bNode, err = gorgonia.Broadcast(aNode, bNode, pattern)
			if err != nil {
				return err
			}
		}
		n.gorgoniaNode, err = gorgonia.BatchedMatMul(aNode, bNode)
	case aDims == 3 && bDims == 2:
		// Broadcast b to match batch dimension of a
		// Reshape b from (K, N) to (1, K, N) then broadcast to (batch, K, N)
		bShape := bNode.Shape()
		bNode, err = gorgonia.Reshape(bNode, []int{1, bShape[0], bShape[1]})
		if err != nil {
			return err
		}
		// Now broadcast B's batch dimension to match A's batch dimension
		aShape := aNode.Shape()
		if aShape[0] > 1 {
			pattern := gorgonia.NewBroadcastPattern(nil, []byte{0}) // broadcast B's axis 0
			aNode, bNode, err = gorgonia.Broadcast(aNode, bNode, pattern)
			if err != nil {
				return err
			}
		}
		n.gorgoniaNode, err = gorgonia.BatchedMatMul(aNode, bNode)
	case aDims == 2 && bDims == 3:
		// Broadcast a to match batch dimension of b
		// Reshape a from (M, K) to (1, M, K) then broadcast to (batch, M, K)
		aShape := aNode.Shape()
		aNode, err = gorgonia.Reshape(aNode, []int{1, aShape[0], aShape[1]})
		if err != nil {
			return err
		}
		// Now broadcast A's batch dimension to match B's batch dimension
		bShape := bNode.Shape()
		if bShape[0] > 1 {
			pattern := gorgonia.NewBroadcastPattern([]byte{0}, nil) // broadcast A's axis 0
			aNode, bNode, err = gorgonia.Broadcast(aNode, bNode, pattern)
			if err != nil {
				return err
			}
		}
		n.gorgoniaNode, err = gorgonia.BatchedMatMul(aNode, bNode)
	default:
		return &onnx.ErrNotImplemented{
			Operator: "Matmul",
			Message:  "dimension too high",
		}
	}

	return err
}

func (a *matMul) init(o onnx.Operation) error {
	return nil
}
