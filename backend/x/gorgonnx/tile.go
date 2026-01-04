package gorgonnx

import (
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile

func init() {
	register("Tile", newTile)
}

type tileOp struct {
	repeats []int64
}

func (t *tileOp) Arity() int { return 1 }

func (t *tileOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (t *tileOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("tile: infershape failed, nil shape")
	}
	inputShape := inputs[0].(tensor.Shape)
	outputShape := make(tensor.Shape, len(inputShape))
	for i := range inputShape {
		if i < len(t.repeats) {
			outputShape[i] = inputShape[i] * int(t.repeats[i])
		} else {
			outputShape[i] = inputShape[i]
		}
	}
	return outputShape, nil
}

func (t *tileOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("tile: expected 1 input, got %d", len(inputs))
	}

	input, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("tile: only dense tensors are supported")
	}

	inputShape := input.Shape()
	numDims := len(inputShape)

	// Ensure repeats has same length as inputShape
	repeats := make([]int64, numDims)
	for i := range repeats {
		if i < len(t.repeats) {
			repeats[i] = t.repeats[i]
		} else {
			repeats[i] = 1
		}
	}

	outputShape := make(tensor.Shape, numDims)
	for i := range inputShape {
		outputShape[i] = inputShape[i] * int(repeats[i])
	}

	// Create output tensor
	result := tensor.New(tensor.WithShape(outputShape...), tensor.Of(input.Dtype()))

	// Tile by copying data using generic N-dimensional approach
	err := tileTensorGeneric(input, result, repeats)
	if err != nil {
		return nil, err
	}

	return result, nil
}

func tileTensorGeneric(src, dst *tensor.Dense, repeats []int64) error {
	srcShape := src.Shape()
	numDims := len(srcShape)

	// Iterate over all destination elements
	dstShape := dst.Shape()
	totalSize := dstShape.TotalSize()
	for dstIdx := 0; dstIdx < totalSize; dstIdx++ {
		dstCoords := flatToCoords(dstIdx, dstShape)

		// Map dst coords to src coords using modulo
		srcCoords := make([]int, numDims)
		for dim := 0; dim < numDims; dim++ {
			srcCoords[dim] = dstCoords[dim] % srcShape[dim]
		}

		val, err := src.At(srcCoords...)
		if err != nil {
			return err
		}
		if err := dst.SetAt(val, dstCoords...); err != nil {
			return err
		}
	}

	return nil
}

func (t *tileOp) ReturnsPtr() bool     { return false }
func (t *tileOp) CallsExtern() bool    { return false }
func (t *tileOp) OverwritesInput() int { return -1 }

func (t *tileOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("tile"))
	for _, r := range t.repeats {
		binary.Write(h, binary.LittleEndian, r)
	}
}

func (t *tileOp) Hashcode() uint32 {
	h := fnv.New32a()
	t.WriteHash(h)
	return h.Sum32()
}

func (t *tileOp) String() string { return "Tile" }

type tile struct{}

func newTile() operator {
	return &tile{}
}

func (t *tile) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	input := children[0].gorgoniaNode
	repeatsNode := children[1]

	// Get repeats from the tensor (check both t and gorgoniaNode.Value())
	repeatsTensor := getTensorFromNode(repeatsNode)
	if repeatsTensor == nil {
		return fmt.Errorf("tile: repeats must be a constant tensor")
	}

	repeats := tensorToInt64Slice(repeatsTensor)

	op := &tileOp{repeats: repeats}
	n.gorgoniaNode, err = gorgonia.ApplyOp(op, input)
	return err
}

func (t *tile) init(o onnx.Operation) error {
	return nil
}
