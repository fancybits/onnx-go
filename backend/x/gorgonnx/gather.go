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

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather

func init() {
	register("Gather", newGather)
}

// gatherOutputShape computes: data.shape[:axis] + indices.shape + data.shape[axis+1:]
func gatherOutputShape(dataShape, indicesShape tensor.Shape, axis int) tensor.Shape {
	if axis < 0 {
		axis = len(dataShape) + axis
	}
	outputShape := make(tensor.Shape, 0, len(dataShape)-1+len(indicesShape))
	outputShape = append(outputShape, dataShape[:axis]...)
	outputShape = append(outputShape, indicesShape...)
	outputShape = append(outputShape, dataShape[axis+1:]...)
	return outputShape
}

// doGather performs the gather operation on tensors
func doGather(data *tensor.Dense, indices []int64, indicesShape tensor.Shape, axis int) (*tensor.Dense, error) {
	if axis < 0 {
		axis = len(data.Shape()) + axis
	}

	outputShape := gatherOutputShape(data.Shape(), indicesShape, axis)
	result := tensor.New(tensor.WithShape(outputShape...), tensor.Of(data.Dtype()))

	// Use typed gather to avoid interface{} boxing allocations
	switch src := data.Data().(type) {
	case []float32:
		doGatherTyped(src, result.Float32s(), data, result, indices, indicesShape, axis)
	case []float64:
		doGatherTyped(src, result.Float64s(), data, result, indices, indicesShape, axis)
	case []int32:
		doGatherTyped(src, result.Int32s(), data, result, indices, indicesShape, axis)
	case []int64:
		doGatherTyped(src, result.Int64s(), data, result, indices, indicesShape, axis)
	default:
		return doGatherGeneric(data, result, indices, indicesShape, axis)
	}
	return result, nil
}

// doGatherTyped performs gather with direct slice access to avoid boxing
func doGatherTyped[T any](srcData, dstData []T, data, result *tensor.Dense, indices []int64, indicesShape tensor.Shape, axis int) {
	srcShape := data.Shape()
	dstShape := result.Shape()
	srcStrides := data.Strides()
	dstStrides := result.Strides()
	numIndicesDims := len(indicesShape)

	dstCoords := make([]int, len(dstShape))
	srcCoords := make([]int, len(srcShape))

	for dstIdx := 0; dstIdx < dstShape.TotalSize(); dstIdx++ {
		flatToCoordsInPlace(dstIdx, dstShape, dstCoords)
		copy(srcCoords[:axis], dstCoords[:axis])

		idxFlat := coordsToFlat(dstCoords[axis:axis+numIndicesDims], indicesShape)
		srcIdx := int(indices[idxFlat])
		if srcIdx < 0 {
			srcIdx = srcShape[axis] + srcIdx
		}
		srcCoords[axis] = srcIdx
		copy(srcCoords[axis+1:], dstCoords[axis+numIndicesDims:])

		dstData[coordsToFlatWithStrides(dstCoords, dstStrides)] = srcData[coordsToFlatWithStrides(srcCoords, srcStrides)]
	}
}

// doGatherGeneric is the fallback using interface{} (slower due to boxing)
func doGatherGeneric(data, result *tensor.Dense, indices []int64, indicesShape tensor.Shape, axis int) (*tensor.Dense, error) {
	srcShape := data.Shape()
	dstShape := result.Shape()
	numIndicesDims := len(indicesShape)

	dstCoords := make([]int, len(dstShape))
	srcCoords := make([]int, len(srcShape))

	for dstIdx := 0; dstIdx < dstShape.TotalSize(); dstIdx++ {
		flatToCoordsInPlace(dstIdx, dstShape, dstCoords)
		copy(srcCoords[:axis], dstCoords[:axis])

		idxFlat := coordsToFlat(dstCoords[axis:axis+numIndicesDims], indicesShape)
		srcIdx := int(indices[idxFlat])
		if srcIdx < 0 {
			srcIdx = srcShape[axis] + srcIdx
		}
		srcCoords[axis] = srcIdx
		copy(srcCoords[axis+1:], dstCoords[axis+numIndicesDims:])

		val, err := data.At(srcCoords...)
		if err != nil {
			return nil, err
		}
		if err := result.SetAt(val, dstCoords...); err != nil {
			return nil, err
		}
	}
	return result, nil
}

// gatherOp implements gather for gorgonia's expression graph
type gatherOp struct {
	axis         int
	indices      []int64      // nil if indices come from second input at runtime
	indicesShape tensor.Shape
}

func (g *gatherOp) Arity() int {
	if g.indices == nil {
		return 2 // runtime indices
	}
	return 1 // constant indices baked in
}

func (g *gatherOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	if g.indices == nil {
		b := hm.TypeVariable('b')
		return hm.NewFnType(a, b, a)
	}
	return hm.NewFnType(a, a)
}

func (g *gatherOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("gather: infershape failed, nil shape")
	}
	return gatherOutputShape(inputs[0].(tensor.Shape), g.indicesShape, g.axis), nil
}

func (g *gatherOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	data, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("gather: data must be a dense tensor")
	}

	indices := g.indices
	indicesShape := g.indicesShape

	// If indices not baked in, get from second input
	if indices == nil {
		if len(inputs) != 2 {
			return nil, fmt.Errorf("gather: expected 2 inputs for runtime indices, got %d", len(inputs))
		}
		indicesTensor, ok := inputs[1].(*tensor.Dense)
		if !ok {
			return nil, fmt.Errorf("gather: indices must be a dense tensor")
		}
		indices = tensorToInt64Slice(indicesTensor)
		indicesShape = indicesTensor.Shape()
	}

	return doGather(data, indices, indicesShape, g.axis)
}

func (g *gatherOp) ReturnsPtr() bool     { return false }
func (g *gatherOp) CallsExtern() bool    { return false }
func (g *gatherOp) OverwritesInput() int { return -1 }

func (g *gatherOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("gather"))
	binary.Write(h, binary.LittleEndian, int64(g.axis))
	for _, idx := range g.indices {
		binary.Write(h, binary.LittleEndian, idx)
	}
}

func (g *gatherOp) Hashcode() uint32 {
	h := fnv.New32a()
	g.WriteHash(h)
	return h.Sum32()
}

func (g *gatherOp) String() string { return "Gather" }

type gather struct {
	axis int
}

func newGather() operator {
	return &gather{}
}

func (g *gather) apply(gg *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(gg.g, n)
	if err := checkCondition(children, 2); err != nil {
		return err
	}

	dataNode := children[0]
	indicesNode := children[1]
	axis := g.axis

	// Only compile-time constant provenance admits a build-time decision here:
	// both the fold below and the "bake the indices into the op" path turn a
	// tensor into part of the graph, which is wrong for anything a later run
	// can change.
	dataTensor := constTensorFromNode(dataNode)
	indicesTensor := constTensorFromNode(indicesNode)

	// If both data and indices are constants, perform gather immediately
	if dataTensor != nil && indicesTensor != nil {
		dataD, ok := dataTensor.(*tensor.Dense)
		if !ok {
			return fmt.Errorf("gather: data must be a dense tensor")
		}

		result, err := doGather(dataD, tensorToInt64Slice(indicesTensor), indicesTensor.Shape(), axis)
		if err != nil {
			return fmt.Errorf("gather constant: %w", err)
		}

		// Record the fold as a value with provenance, so that a downstream
		// build-time consumer can tell it apart from a run-time tensor.
		n.t = result
		n.constant = true
		n.gorgoniaNode = gorgonia.NodeFromAny(gg.exprgraph, result, gorgonia.WithName(getUniqNodeName("gather_const")))
		return nil
	}

	// Build the gorgonia op
	var op *gatherOp
	var children_nodes gorgonia.Nodes
	var dataShape tensor.Shape

	if indicesTensor != nil {
		// Constant indices - bake them into the op
		data := dataNode.gorgoniaNode
		dataShape = data.Shape()
		op = &gatherOp{
			axis:         axis,
			indices:      tensorToInt64Slice(indicesTensor),
			indicesShape: indicesTensor.Shape(),
		}
		children_nodes = gorgonia.Nodes{data}
	} else {
		// Runtime indices
		if dataNode.gorgoniaNode == nil {
			return fmt.Errorf("gather: data input has no gorgonia node")
		}
		if indicesNode.gorgoniaNode == nil {
			return fmt.Errorf("gather: indices input has no gorgonia node")
		}

		data := dataNode.gorgoniaNode
		indicesGNode := indicesNode.gorgoniaNode
		dataShape = data.Shape()
		op = &gatherOp{
			axis:         axis,
			indices:      nil, // runtime
			indicesShape: indicesGNode.Shape(),
		}
		children_nodes = gorgonia.Nodes{data, indicesGNode}
	}

	outputShape := gatherOutputShape(dataShape, op.indicesShape, axis)
	tt := gorgonia.TensorType{Dims: len(outputShape), Of: dataNode.gorgoniaNode.Dtype()}
	n.gorgoniaNode = gorgonia.NewUniqueNode(
		gorgonia.WithType(tt),
		gorgonia.WithOp(op),
		gorgonia.WithChildren(children_nodes),
		gorgonia.In(gg.exprgraph),
		gorgonia.WithShape(outputShape...),
	)
	return nil
}

func (g *gather) init(o onnx.Operation) error {
	// Default axis is 0
	g.axis = 0

	if axis, ok := o.Attributes["axis"]; ok {
		if axisInt, ok := axis.(int64); ok {
			g.axis = int(axisInt)
		} else {
			return fmt.Errorf("gather: axis is not an int64")
		}
	}

	return nil
}
