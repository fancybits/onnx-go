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

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence

func init() {
	register("ReverseSequence", newReverseSequence)
}

type reverseSequenceOp struct {
	batchAxis int
	timeAxis  int
	seqLens   []int64
}

func (r *reverseSequenceOp) Arity() int { return 1 }

func (r *reverseSequenceOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (r *reverseSequenceOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("reverseSequence: infershape failed, nil shape")
	}
	return inputs[0].(tensor.Shape).Clone(), nil
}

func (r *reverseSequenceOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("reverseSequence: expected 1 input, got %d", len(inputs))
	}

	input, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("reverseSequence: only dense tensors are supported")
	}

	inputShape := input.Shape()
	result := input.Clone().(*tensor.Dense)

	// For each batch, reverse the first seqLen elements along the time axis
	batchSize := inputShape[r.batchAxis]
	timeSize := inputShape[r.timeAxis]

	for b := 0; b < batchSize; b++ {
		seqLen := int(r.seqLens[b])
		if seqLen > timeSize {
			seqLen = timeSize
		}

		// Reverse the elements from 0 to seqLen-1 along the time axis for this batch
		for t := 0; t < seqLen/2; t++ {
			reverseT := seqLen - 1 - t

			// Get indices for the two positions to swap
			idx1 := make([]int, len(inputShape))
			idx2 := make([]int, len(inputShape))

			idx1[r.batchAxis] = b
			idx1[r.timeAxis] = t
			idx2[r.batchAxis] = b
			idx2[r.timeAxis] = reverseT

			// For other dimensions, iterate over all possible values
			err := reverseSequenceSwap(result, inputShape, idx1, idx2, 0, r.batchAxis, r.timeAxis)
			if err != nil {
				return nil, err
			}
		}
	}

	return result, nil
}

// reverseSequenceSwap recursively swaps elements along non-batch/time dimensions
func reverseSequenceSwap(t *tensor.Dense, shape tensor.Shape, idx1, idx2 []int, dim, batchAxis, timeAxis int) error {
	if dim >= len(shape) {
		// Base case: perform the swap
		val1, err := t.At(idx1...)
		if err != nil {
			return err
		}
		val2, err := t.At(idx2...)
		if err != nil {
			return err
		}
		if err := t.SetAt(val2, idx1...); err != nil {
			return err
		}
		if err := t.SetAt(val1, idx2...); err != nil {
			return err
		}
		return nil
	}

	// Skip batch and time axes (already handled in outer loop)
	if dim == batchAxis || dim == timeAxis {
		return reverseSequenceSwap(t, shape, idx1, idx2, dim+1, batchAxis, timeAxis)
	}

	// Iterate over this dimension
	for i := 0; i < shape[dim]; i++ {
		idx1[dim] = i
		idx2[dim] = i
		if err := reverseSequenceSwap(t, shape, idx1, idx2, dim+1, batchAxis, timeAxis); err != nil {
			return err
		}
	}
	return nil
}

func (r *reverseSequenceOp) ReturnsPtr() bool     { return false }
func (r *reverseSequenceOp) CallsExtern() bool    { return false }
func (r *reverseSequenceOp) OverwritesInput() int { return -1 }

func (r *reverseSequenceOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("reverseSequence"))
	binary.Write(h, binary.LittleEndian, int64(r.batchAxis))
	binary.Write(h, binary.LittleEndian, int64(r.timeAxis))
}

func (r *reverseSequenceOp) Hashcode() uint32 {
	h := fnv.New32a()
	r.WriteHash(h)
	return h.Sum32()
}

func (r *reverseSequenceOp) String() string { return "ReverseSequence" }

type reverseSequence struct {
	batchAxis int
	timeAxis  int
}

func newReverseSequence() operator {
	return &reverseSequence{
		batchAxis: 1, // default
		timeAxis:  0, // default
	}
}

func (r *reverseSequence) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	input := children[0].gorgoniaNode
	seqLensNode := children[1]

	seqLensTensor := getTensorFromNode(seqLensNode)
	if seqLensTensor == nil {
		return fmt.Errorf("reverseSequence: sequence_lens must be a constant tensor")
	}

	seqLens := tensorToInt64Slice(seqLensTensor)

	op := &reverseSequenceOp{
		batchAxis: r.batchAxis,
		timeAxis:  r.timeAxis,
		seqLens:   seqLens,
	}
	n.gorgoniaNode, err = gorgonia.ApplyOp(op, input)
	return err
}

func (r *reverseSequence) init(o onnx.Operation) error {
	if batchAxis, ok := o.Attributes["batch_axis"]; ok {
		if ba, ok := batchAxis.(int64); ok {
			r.batchAxis = int(ba)
		}
	}

	if timeAxis, ok := o.Attributes["time_axis"]; ok {
		if ta, ok := timeAxis.(int64); ok {
			r.timeAxis = int(ta)
		}
	}

	return nil
}
