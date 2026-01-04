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
		// Batched matrix multiplication
		n.gorgoniaNode, err = gorgonia.BatchedMatMul(aNode, bNode)
	case aDims == 3 && bDims == 2:
		// Broadcast b to match batch dimension of a
		// Reshape b from (K, N) to (1, K, N) then broadcast
		bShape := bNode.Shape()
		bNode, err = gorgonia.Reshape(bNode, []int{1, bShape[0], bShape[1]})
		if err != nil {
			return err
		}
		n.gorgoniaNode, err = gorgonia.BatchedMatMul(aNode, bNode)
	case aDims == 2 && bDims == 3:
		// Broadcast a to match batch dimension of b
		aShape := aNode.Shape()
		aNode, err = gorgonia.Reshape(aNode, []int{1, aShape[0], aShape[1]})
		if err != nil {
			return err
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
