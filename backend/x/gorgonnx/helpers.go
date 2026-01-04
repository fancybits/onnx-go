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
	for i := len(shape) - 1; i >= 0; i-- {
		coords[i] = idx % shape[i]
		idx /= shape[i]
	}
	return coords
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

// getTensorFromNode retrieves a tensor from a node, checking both t and gorgoniaNode.Value()
func getTensorFromNode(node *Node) tensor.Tensor {
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

