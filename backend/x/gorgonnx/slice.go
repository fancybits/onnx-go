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

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice
// ONNX Opset 10+: starts, ends, axes, steps are inputs (not attributes)

func init() {
	register("Slice", newSlice)
}

// sliceParam holds normalized parameters for one axis
type sliceParam struct {
	start, end, step int64
}

// normalizeSliceParam normalizes start/end/step for a given dimension size
func normalizeSliceParam(start, end, step, dimSize int64) sliceParam {
	if step == 0 {
		step = 1
	}

	// Handle negative indices
	if start < 0 {
		start = dimSize + start
	}
	if end < 0 {
		end = dimSize + end
	}

	// Clamp indices based on step direction
	if step > 0 {
		start = clamp(start, 0, dimSize)
		end = clamp(end, 0, dimSize)
	} else {
		start = clamp(start, -1, dimSize-1)
		end = clamp(end, -1, dimSize-1)
	}

	return sliceParam{start, end, step}
}

// sliceSize calculates the output size for a slice
func (p sliceParam) size() int {
	var size int64
	if p.step > 0 {
		size = (p.end - p.start + p.step - 1) / p.step
	} else {
		size = (p.start - p.end - p.step - 1) / (-p.step)
	}
	if size < 0 {
		return 0
	}
	return int(size)
}

func clamp(v, min, max int64) int64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// doSlice performs the slice operation on a tensor
func doSlice(input *tensor.Dense, starts, ends, axes, steps []int64) (*tensor.Dense, error) {
	inputShape := input.Shape()
	numDims := len(inputShape)

	// Initialize params for all dimensions (default: full slice)
	params := make([]sliceParam, numDims)
	for i := range params {
		params[i] = sliceParam{0, int64(inputShape[i]), 1}
	}

	// Apply specified slices
	outputShape := make(tensor.Shape, numDims)
	copy(outputShape, inputShape)

	for i, axis := range axes {
		if axis < 0 {
			axis = int64(numDims) + axis
		}
		params[axis] = normalizeSliceParam(starts[i], ends[i], steps[i], int64(inputShape[axis]))
		outputShape[axis] = params[axis].size()
	}

	result := tensor.New(tensor.WithShape(outputShape...), tensor.Of(input.Dtype()))

	if outputShape.TotalSize() == 0 {
		return result, nil
	}

	// Copy data with slicing
	srcShape := input.Shape()
	totalSize := outputShape.TotalSize()
	for dstIdx := 0; dstIdx < totalSize; dstIdx++ {
		dstCoords := flatToCoords(dstIdx, outputShape)

		srcCoords := make([]int, numDims)
		for dim := 0; dim < numDims; dim++ {
			srcCoords[dim] = int(params[dim].start) + dstCoords[dim]*int(params[dim].step)
			if srcCoords[dim] < 0 || srcCoords[dim] >= srcShape[dim] {
				return nil, fmt.Errorf("slice: source index out of bounds: dim %d, index %d, size %d", dim, srcCoords[dim], srcShape[dim])
			}
		}

		val, err := input.At(srcCoords...)
		if err != nil {
			return nil, err
		}
		if err := result.SetAt(val, dstCoords...); err != nil {
			return nil, err
		}
	}

	return result, nil
}

type sliceOp struct {
	starts []int64
	ends   []int64
	axes   []int64
	steps  []int64
}

func (s *sliceOp) Arity() int { return 1 }

func (s *sliceOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (s *sliceOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("slice: infershape failed, nil shape")
	}
	inputShape := inputs[0].(tensor.Shape)
	outputShape := make(tensor.Shape, len(inputShape))
	copy(outputShape, inputShape)

	for i, axis := range s.axes {
		if axis < 0 {
			axis = int64(len(inputShape)) + axis
		}
		p := normalizeSliceParam(s.starts[i], s.ends[i], s.steps[i], int64(inputShape[axis]))
		outputShape[axis] = p.size()
	}

	return outputShape, nil
}

func (s *sliceOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("slice: expected 1 input, got %d", len(inputs))
	}

	input, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("slice: only dense tensors are supported")
	}

	return doSlice(input, s.starts, s.ends, s.axes, s.steps)
}

func (s *sliceOp) ReturnsPtr() bool     { return false }
func (s *sliceOp) CallsExtern() bool    { return false }
func (s *sliceOp) OverwritesInput() int { return -1 }

func (s *sliceOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("slice"))
	for _, v := range s.starts {
		binary.Write(h, binary.LittleEndian, v)
	}
	for _, v := range s.ends {
		binary.Write(h, binary.LittleEndian, v)
	}
	for _, v := range s.axes {
		binary.Write(h, binary.LittleEndian, v)
	}
	for _, v := range s.steps {
		binary.Write(h, binary.LittleEndian, v)
	}
}

func (s *sliceOp) Hashcode() uint32 {
	h := fnv.New32a()
	s.WriteHash(h)
	return h.Sum32()
}

func (s *sliceOp) String() string { return "Slice" }

type slice struct{}

func newSlice() operator {
	return &slice{}
}

func (s *slice) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	// Slice has 3-5 inputs: data, starts, ends, [axes], [steps]. The optional
	// ones may be omitted, so key the children on their input ordinal rather
	// than on their position.
	childByWeight := getChildrenByInputIndex(g.g, n)

	// Get data (input 0)
	dataNode := childByWeight[0]
	if dataNode == nil || dataNode.gorgoniaNode == nil {
		return fmt.Errorf("slice: data input is missing")
	}
	data := dataNode.gorgoniaNode
	dataShape := data.Shape()
	numDims := len(dataShape)

	// Get slice parameters from inputs
	startsNode := childByWeight[1]
	endsNode := childByWeight[2]
	axesNode := childByWeight[3]
	stepsNode := childByWeight[4]

	// starts/ends/axes/steps are baked into the slice the graph performs — they
	// determine the output shape — so they have to be read at build time
	// whatever their provenance. See staticTensorFromNode.
	startsTensor := staticTensorFromNode(startsNode)
	endsTensor := staticTensorFromNode(endsNode)

	if startsTensor == nil {
		return fmt.Errorf("slice: starts input is missing or not a constant tensor")
	}
	if endsTensor == nil {
		return fmt.Errorf("slice: ends input is missing or not a constant tensor")
	}

	starts := tensorToInt64Slice(startsTensor)
	ends := tensorToInt64Slice(endsTensor)

	// A parameter read above may have come from a graph input, so anything
	// derived from it is not a compile-time constant. Track that, so the fold
	// below cannot claim provenance its inputs do not have. Omitted optional
	// inputs contribute nothing: their defaults are constants.
	paramsConst := isConstNode(startsNode) && isConstNode(endsNode)

	// Optional axes (defaults to 0, 1, 2, ...)
	var axes []int64
	if axesTensor := staticTensorFromNode(axesNode); axesTensor != nil {
		axes = tensorToInt64Slice(axesTensor)
		paramsConst = paramsConst && isConstNode(axesNode)
	} else {
		axesLen := len(starts)
		if axesLen > numDims {
			axesLen = numDims
			starts = starts[:axesLen]
			ends = ends[:axesLen]
		}
		axes = make([]int64, axesLen)
		for i := range axes {
			axes[i] = int64(i)
		}
	}

	// Optional steps (defaults to 1)
	var steps []int64
	if stepsTensor := staticTensorFromNode(stepsNode); stepsTensor != nil {
		steps = tensorToInt64Slice(stepsTensor)
		paramsConst = paramsConst && isConstNode(stepsNode)
	} else {
		steps = make([]int64, len(starts))
		for i := range steps {
			steps[i] = 1
		}
	}

	// Validate axes
	for _, axis := range axes {
		if axis < 0 {
			axis = int64(numDims) + axis
		}
		if axis < 0 || int(axis) >= numDims {
			return fmt.Errorf("slice: axis %d out of range for %d dimensions", axis, numDims)
		}
	}

	// Check if we can use Gorgonia's built-in Slice (step=1, no size-1 results)
	canUseBuiltin := true
	for i, axis := range axes {
		if steps[i] != 1 {
			canUseBuiltin = false
			break
		}
		if axis < 0 {
			axis = int64(numDims) + axis
		}
		p := normalizeSliceParam(starts[i], ends[i], steps[i], int64(dataShape[axis]))
		if p.size() == 1 {
			canUseBuiltin = false
			break
		}
	}

	// If data is a genuine compile-time constant, perform the slice immediately.
	// Anything else goes down the symbolic path, which computes the same result
	// at run time from whatever is bound then.
	dataTensor := constTensorFromNode(dataNode)
	if dataTensor != nil {
		dataD, ok := dataTensor.(*tensor.Dense)
		if !ok {
			return fmt.Errorf("slice: data must be a dense tensor")
		}
		result, err := doSlice(dataD, starts, ends, axes, steps)
		if err != nil {
			return fmt.Errorf("slice constant: %w", err)
		}
		// The data was a proven constant, but a slice parameter may not have
		// been: the result is only a constant when every input to it was.
		n.t = result
		n.constant = paramsConst
		n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("slice_const")))
		return nil
	}

	// Use custom Op for complex cases (non-1 steps or size-1 results)
	if !canUseBuiltin {
		op := &sliceOp{starts: starts, ends: ends, axes: axes, steps: steps}
		var err error
		n.gorgoniaNode, err = gorgonia.ApplyOp(op, data)
		return err
	}

	// Use Gorgonia's built-in Slice for simple cases
	slices := make([]tensor.Slice, numDims)
	for i, axis := range axes {
		if axis < 0 {
			axis = int64(numDims) + axis
		}
		p := normalizeSliceParam(starts[i], ends[i], 1, int64(dataShape[axis]))
		if p.start > p.end {
			p.start = p.end
		}
		slices[axis] = gorgonia.S(int(p.start), int(p.end))
	}

	var err error
	n.gorgoniaNode, err = gorgonia.Slice(data, slices...)
	return err
}

func (s *slice) init(o onnx.Operation) error {
	return nil
}
