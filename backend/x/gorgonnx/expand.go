package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand

func init() {
	register("Expand", newExpand)
}

type expand struct{}

func newExpand() operator {
	return &expand{}
}

func (e *expand) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	if err := checkCondition(children, 2); err != nil {
		return err
	}

	inputNode := children[0]
	shapeNode := children[1]

	// Get target shape from the shape tensor. It determines the output shape,
	// so it has to be read at build time whatever its provenance.
	shapeTensor := staticTensorFromNode(shapeNode)
	if shapeTensor == nil {
		return fmt.Errorf("expand: shape must be a constant tensor")
	}
	targetShape := tensorToIntSlice(shapeTensor)

	// If input is a constant tensor, compute the expand directly
	if inputTensor := constTensorFromNode(inputNode); inputTensor != nil {
		inputD, ok := inputTensor.(*tensor.Dense)
		if !ok {
			return fmt.Errorf("expand: input must be a dense tensor")
		}
		result, err := doExpand(inputD, targetShape)
		if err != nil {
			return err
		}
		// The input was a proven constant, but the shape may have come from a
		// graph input: the result is one only when both were.
		n.t = result
		n.constant = isConstNode(shapeNode)
		n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("expand_result")))
		return nil
	}

	// For non-constant inputs, use broadcasting at runtime
	input := inputNode.gorgoniaNode

	// The symbolic path below broadcasts via a Hadamard product against a
	// ones tensor, which only typechecks for numeric dtypes. A bool input
	// that reaches here (e.g. a direct graph input bound at run time) cannot
	// take the constant fast path above, and gorgonia's multiply would fail
	// deep in the tape with an opaque typeclass error. Refuse loudly instead
	// of silently mishandling — or, worse, letting a future numeric-only
	// change quietly produce wrong bool results.
	if input.Dtype() == tensor.Bool {
		return fmt.Errorf("expand: runtime bool expansion is not supported (symbolic path is numeric-only)")
	}

	inputShape := input.Shape()
	outputShape, _ := computeBroadcastShape(inputShape, targetShape)

	// Pad input shape to match output dimensions if needed
	paddedInputShape := padShapeFront(inputShape, len(outputShape))

	var toExpand *gorgonia.Node
	var err error
	if len(outputShape) != len(inputShape) {
		toExpand, err = gorgonia.Reshape(input, paddedInputShape)
		if err != nil {
			return fmt.Errorf("expand: reshape failed: %v", err)
		}
	} else {
		toExpand = input
	}

	// Use broadcasting with ones tensor
	ones := tensor.Ones(input.Dtype(), outputShape...)
	onesNode := gorgonia.NodeFromAny(g.exprgraph, ones, gorgonia.WithName(getUniqNodeName("expand_ones")))

	pattern := buildBroadcastPattern(paddedInputShape, outputShape)
	// gorgonia packs a broadcast pattern into one byte, a nibble per operand,
	// and builds it with `byte(1) << axis` — for the left operand, shifted a
	// further 4 bits. An axis of 4 or more therefore shifts its bit clean off
	// the byte and the broadcast is dropped with no error: the op would run and
	// return unbroadcast numbers. Refuse loudly instead.
	for _, axis := range pattern {
		if axis >= gorgoniaBroadcastAxisLimit {
			return fmt.Errorf("expand: broadcast along axis %d exceeds gorgonia's limit of %d axes",
				axis, gorgoniaBroadcastAxisLimit)
		}
	}
	n.gorgoniaNode, err = gorgonia.BroadcastHadamardProd(toExpand, onesNode, pattern, nil)
	return err
}

// gorgoniaBroadcastAxisLimit mirrors gorgonia's unexported bcAllowableAxes: a
// BroadcastPattern is a single byte split into one nibble per operand, so only
// axes 0..3 can be expressed.
const gorgoniaBroadcastAxisLimit = 4

func (e *expand) init(o onnx.Operation) error {
	return nil
}

// padShapeFront pads a shape with 1s at the front to reach targetLen
func padShapeFront(shape tensor.Shape, targetLen int) tensor.Shape {
	if len(shape) >= targetLen {
		return shape
	}
	padded := make(tensor.Shape, targetLen)
	offset := targetLen - len(shape)
	for i := 0; i < offset; i++ {
		padded[i] = 1
	}
	copy(padded[offset:], shape)
	return padded
}

// buildBroadcastPattern lists the axes to broadcast the input along: those
// where inputShape[i]==1 and outputShape[i]>1.
//
// gorgonia.NewBroadcastPattern takes axis *indices*, not a per-axis flag array
// — it sets one bit per element it is given. Returning flags made every
// zero-flagged axis read as "broadcast axis 0", so a (3,1) input expanded to
// (3,4) came out as (9,4).
func buildBroadcastPattern(inputShape, outputShape tensor.Shape) []byte {
	var axes []byte
	for i := range outputShape {
		if inputShape[i] == 1 && outputShape[i] > 1 {
			axes = append(axes, byte(i))
		}
	}
	return axes
}

// computeBroadcastShape computes the output shape and broadcast flags for input
func computeBroadcastShape(inputShape, targetShape tensor.Shape) (tensor.Shape, []bool) {
	maxLen := len(inputShape)
	if len(targetShape) > maxLen {
		maxLen = len(targetShape)
	}

	paddedInput := padShapeFront(inputShape, maxLen)
	paddedTarget := padShapeFront(targetShape, maxLen)

	outputShape := make(tensor.Shape, maxLen)
	inputBroadcast := make([]bool, maxLen)

	for i := 0; i < maxLen; i++ {
		if paddedInput[i] == paddedTarget[i] {
			outputShape[i] = paddedInput[i]
		} else if paddedInput[i] == 1 {
			outputShape[i] = paddedTarget[i]
			inputBroadcast[i] = true
		} else if paddedTarget[i] == 1 {
			outputShape[i] = paddedInput[i]
		} else {
			// Shapes are incompatible - use target
			outputShape[i] = paddedTarget[i]
			inputBroadcast[i] = (paddedInput[i] == 1)
		}
	}

	return outputShape, inputBroadcast
}

// doExpand performs expand on a constant tensor
func doExpand(input *tensor.Dense, targetShape tensor.Shape) (*tensor.Dense, error) {
	inputShape := input.Shape()
	outputShape, broadcast := computeBroadcastShape(inputShape, targetShape)

	result := tensor.New(tensor.WithShape(outputShape...), tensor.Of(input.Dtype()))

	// Copy with broadcasting
	paddedSrcShape := padShapeFront(inputShape, len(outputShape))
	offset := len(outputShape) - len(inputShape)

	totalSize := outputShape.TotalSize()
	for dstIdx := 0; dstIdx < totalSize; dstIdx++ {
		dstCoords := flatToCoords(dstIdx, outputShape)

		// Map dst coords to src coords (with broadcasting)
		srcCoords := make([]int, len(inputShape))
		for i := 0; i < len(inputShape); i++ {
			dstDim := offset + i
			if broadcast[dstDim] || paddedSrcShape[dstDim] == 1 {
				srcCoords[i] = 0
			} else {
				srcCoords[i] = dstCoords[dstDim]
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
