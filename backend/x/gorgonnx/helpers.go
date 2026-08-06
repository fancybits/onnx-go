package gorgonnx

import "gorgonia.org/tensor"

// isIntegerTensor returns true if the tensor has an integer dtype
func isIntegerTensor(t tensor.Tensor) bool {
	switch t.Dtype() {
	case tensor.Int, tensor.Int8, tensor.Int16, tensor.Int32, tensor.Int64,
		tensor.Uint, tensor.Uint8, tensor.Uint16, tensor.Uint32, tensor.Uint64:
		return true
	default:
		return false
	}
}

// flatToCoords converts a flat index to multi-dimensional coordinates
func flatToCoords(idx int, shape tensor.Shape) []int {
	coords := make([]int, len(shape))
	flatToCoordsInPlace(idx, shape, coords)
	return coords
}

// flatToCoordsInPlace converts a flat index to multi-dimensional coordinates
// writing to the provided coords slice to avoid allocations
func flatToCoordsInPlace(idx int, shape tensor.Shape, coords []int) {
	for i := len(shape) - 1; i >= 0; i-- {
		coords[i] = idx % shape[i]
		idx /= shape[i]
	}
}

// coordsToFlat converts multi-dimensional coordinates to a flat index
func coordsToFlat(coords []int, shape tensor.Shape) int {
	idx := 0
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		idx += coords[i] * stride
		stride *= shape[i]
	}
	return idx
}

// coordsToFlatWithStrides converts multi-dimensional coordinates to a flat index using precomputed strides
func coordsToFlatWithStrides(coords []int, strides []int) int {
	idx := 0
	for i, c := range coords {
		idx += c * strides[i]
	}
	return idx
}

// staticTensorFromNode retrieves a tensor an operator cannot build its part of
// the graph without: a target shape, an axis list, a repeat count. Such a value
// determines the output shape or the structure of the op itself, so it has to
// be read while the exprgraph is built — there is no symbolic alternative to
// fall back to, and refusing a value here means refusing the model.
//
// It therefore accepts a tensor bound to a graph input, which is why it is NOT
// interchangeable with constTensorFromNode: reading one specialises the graph
// to that value for as long as the graph lives. That is the same contract the
// graph already has with its inputs' shapes — callers that change such a value
// between runs must Reset() and rebuild, exactly as they must to change a batch
// size. Use this only for build-time structural requirements. Anything that
// folds a value into the computation, where a symbolic path exists that would
// compute it at run time instead, must use constTensorFromNode.
func staticTensorFromNode(node *Node) tensor.Tensor {
	if node == nil {
		return nil
	}
	if node.t != nil {
		return node.t
	}
	if node.gorgoniaNode != nil && node.gorgoniaNode.Value() != nil {
		if t, ok := node.gorgoniaNode.Value().(tensor.Tensor); ok {
			return t
		}
	}
	return nil
}

// constTensorFromNode returns the tensor a node carries, but only when that
// tensor is a compile-time constant: an initializer, a Constant, a value
// derived from the graph's static shapes, or a fold this backend computed from
// those. It returns nil for everything else.
//
// This is the only admissible source for a constant fold: computing a value at
// build time and pinning it into the graph, where the symbolic path would
// otherwise have computed it per run. Reading Node.t or gorgonia's Value()
// instead is unsound — NodeFromAny binds a value to every leaf, so a graph
// input is indistinguishable from a genuine constant, and SetInput may already
// have bound this run's data into it before the graph is built. Folding that
// bakes one run's inputs into the graph forever, and later runs silently keep
// returning the first run's answer. The provenance flag is the only evidence
// that tells the two apart; there is deliberately no fallback.
func constTensorFromNode(node *Node) tensor.Tensor {
	if !isConstNode(node) {
		return nil
	}
	return node.t
}

// allConstInputs reports whether every node in ns is a compile-time constant.
// A fold's output inherits constant provenance only when this holds: an
// operator applied to a run-time value yields a run-time value.
func allConstInputs(ns []*Node) bool {
	for _, n := range ns {
		if !isConstNode(n) {
			return false
		}
	}
	return true
}

// tensorToInt64Slice extracts int64 values from a tensor in flat order.
// Handles scalar (0D), 1D, and multi-dimensional tensors.
func tensorToInt64Slice(t tensor.Tensor) []int64 {
	data := t.Data()

	switch d := data.(type) {
	case int64:
		return []int64{d}
	case []int64:
		result := make([]int64, len(d))
		copy(result, d)
		return result
	case int32:
		return []int64{int64(d)}
	case []int32:
		result := make([]int64, len(d))
		for i, v := range d {
			result[i] = int64(v)
		}
		return result
	case int:
		return []int64{int64(d)}
	case []int:
		result := make([]int64, len(d))
		for i, v := range d {
			result[i] = int64(v)
		}
		return result
	case float64:
		return []int64{int64(d)}
	case []float64:
		result := make([]int64, len(d))
		for i, v := range d {
			result[i] = int64(v)
		}
		return result
	case float32:
		return []int64{int64(d)}
	case []float32:
		result := make([]int64, len(d))
		for i, v := range d {
			result[i] = int64(v)
		}
		return result
	default:
		// Fallback: use flat index access for other types
		size := t.Shape().TotalSize()
		if size == 0 {
			return []int64{}
		}
		dense, ok := t.(*tensor.Dense)
		if !ok {
			return []int64{}
		}
		result := make([]int64, size)
		for i := 0; i < size; i++ {
			val := dense.Get(i)
			switch v := val.(type) {
			case int64:
				result[i] = v
			case int32:
				result[i] = int64(v)
			case int:
				result[i] = int64(v)
			case float64:
				result[i] = int64(v)
			case float32:
				result[i] = int64(v)
			}
		}
		return result
	}
}

// tensorToIntSlice extracts int values from a tensor as a shape.
func tensorToIntSlice(t tensor.Tensor) tensor.Shape {
	dense, ok := t.(*tensor.Dense)
	if !ok {
		return nil
	}
	size := t.Shape().TotalSize()
	result := make(tensor.Shape, size)

	for i := 0; i < size; i++ {
		val := dense.Get(i)
		switch v := val.(type) {
		case int64:
			result[i] = int(v)
		case int32:
			result[i] = int(v)
		case int:
			result[i] = v
		case float64:
			result[i] = int(v)
		case float32:
			result[i] = int(v)
		}
	}

	return result
}

