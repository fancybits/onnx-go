package ir

import (
	"encoding/binary"
	"math"

	"github.com/pkg/errors"

	"gorgonia.org/tensor"
)

// Tensor returns a Gorgonia compatible tensor
func (tx *TensorProto) Tensor() (tensor.Tensor, error) {
	if tx.Segment != nil {
		return nil, errors.Wrap(ErrNotYetImplemented, "This tensor is segmented")
	}
	// Get the data type
	if tx.DataType == int32(TensorProto_UNDEFINED) {
		return nil, errors.New("This tensor datatype is undefined")
	}
	dt, err := TensorProto_DataType(tx.DataType).Dtype()
	if err != nil {
		return nil, err
	}
	var size = make([]int, len(tx.Dims))
	for i := range tx.Dims {
		size[i] = int(tx.Dims[i])
	}
	opts := []tensor.ConsOpt{tensor.WithShape(size...), tensor.Of(dt)}
	var consopts []tensor.ConsOpt
	switch dt {
	case tensor.Bool:
		consopts, err = generateConsOptsFromBoolTensor(tx)
	case tensor.Float32:
		consopts, err = generateConsOptsFromFloat32Tensor(tx)
	case tensor.Float64:
		consopts, err = generateConsOptsFromFloat64Tensor(tx)
	case tensor.Int64:
		consopts, err = generateConsOptsFromInt64Tensor(tx)
	case tensor.Int32:
		consopts, err = generateConsOptsFromInt32Tensor(tx)
	case tensor.Int16:
		consopts, err = generateConsOptsFromInt16Tensor(tx)
	case tensor.Int8:
		consopts, err = generateConsOptsFromInt8Tensor(tx)
	case tensor.Uint64:
		consopts, err = generateConsOptsFromUint64Tensor(tx)
	case tensor.Uint32:
		consopts, err = generateConsOptsFromUint32Tensor(tx)
	case tensor.Uint16:
		consopts, err = generateConsOptsFromUint16Tensor(tx)
	case tensor.Uint8:
		consopts, err = generateConsOptsFromUint8Tensor(tx)
	default:
		err = errors.Wrapf(ErrNotYetImplemented, "Unknown type %v", dt)
	}
	if err != nil {
		return nil, err
	}
	opts = append(opts, consopts...)
	return tensor.New(opts...), nil
}

func generateConsOptsFromBoolTensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.Int32Data != nil:
		backing := make([]bool, len(tx.Int32Data))
		for i := 0; i < len(tx.Int32Data); i++ {
			backing[i] = tx.Int32Data[i] != 0
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.RawData != nil:
		// ONNX uses 1 byte per boolean in RawData
		backing := make([]bool, len(tx.RawData))
		for i, b := range tx.RawData {
			backing[i] = b != 0
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromFloat32Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%4 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numFloats := len(tx.RawData) / 4
		backing := make([]float32, numFloats)
		for i := 0; i < numFloats; i++ {
			offset := i * 4
			uintElement := binary.LittleEndian.Uint32(tx.RawData[offset : offset+4])
			backing[i] = math.Float32frombits(uintElement)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.FloatData != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.FloatData)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromFloat64Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.DoubleData != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.DoubleData)}, nil
	case tx.RawData != nil:
		if len(tx.RawData)%8 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numFloats := len(tx.RawData) / 8
		backing := make([]float64, numFloats)
		for i := 0; i < numFloats; i++ {
			offset := i * 8
			uintElement := binary.LittleEndian.Uint64(tx.RawData[offset : offset+8])
			backing[i] = math.Float64frombits(uintElement)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromInt64Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%8 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numInts := len(tx.RawData) / 8
		backing := make([]int64, numInts)
		for i := 0; i < numInts; i++ {
			offset := i * 8
			uintElement := binary.LittleEndian.Uint64(tx.RawData[offset : offset+8])
			backing[i] = int64(uintElement)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int64Data != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.Int64Data)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromInt32Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%4 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numInts := len(tx.RawData) / 4
		backing := make([]int32, numInts)
		for i := 0; i < numInts; i++ {
			offset := i * 4
			uintElement := binary.LittleEndian.Uint32(tx.RawData[offset : offset+4])
			backing[i] = int32(uintElement)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int32Data != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.Int32Data)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromInt16Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%2 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numInts := len(tx.RawData) / 2
		backing := make([]int16, numInts)
		for i := 0; i < numInts; i++ {
			offset := i * 2
			uintElement := binary.LittleEndian.Uint16(tx.RawData[offset : offset+2])
			backing[i] = int16(uintElement)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int32Data != nil:
		// ONNX stores int16 in int32 field
		backing := make([]int16, len(tx.Int32Data))
		for i, v := range tx.Int32Data {
			backing[i] = int16(v)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromInt8Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		backing := make([]int8, len(tx.RawData))
		for i, b := range tx.RawData {
			backing[i] = int8(b)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int32Data != nil:
		// ONNX stores int8 in int32 field
		backing := make([]int8, len(tx.Int32Data))
		for i, v := range tx.Int32Data {
			backing[i] = int8(v)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromUint64Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%8 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numInts := len(tx.RawData) / 8
		backing := make([]uint64, numInts)
		for i := 0; i < numInts; i++ {
			offset := i * 8
			backing[i] = binary.LittleEndian.Uint64(tx.RawData[offset : offset+8])
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Uint64Data != nil:
		return []tensor.ConsOpt{tensor.WithBacking(tx.Uint64Data)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromUint32Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%4 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numInts := len(tx.RawData) / 4
		backing := make([]uint32, numInts)
		for i := 0; i < numInts; i++ {
			offset := i * 4
			backing[i] = binary.LittleEndian.Uint32(tx.RawData[offset : offset+4])
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Uint64Data != nil:
		// ONNX stores uint32 in uint64 field
		backing := make([]uint32, len(tx.Uint64Data))
		for i, v := range tx.Uint64Data {
			backing[i] = uint32(v)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromUint16Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		if len(tx.RawData)%2 != 0 {
			return nil, errors.Wrapf(ErrCorruptedData, "%v", nil)
		}
		numInts := len(tx.RawData) / 2
		backing := make([]uint16, numInts)
		for i := 0; i < numInts; i++ {
			offset := i * 2
			backing[i] = binary.LittleEndian.Uint16(tx.RawData[offset : offset+2])
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int32Data != nil:
		// ONNX stores uint16 in int32 field
		backing := make([]uint16, len(tx.Int32Data))
		for i, v := range tx.Int32Data {
			backing[i] = uint16(v)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}

func generateConsOptsFromUint8Tensor(tx *TensorProto) ([]tensor.ConsOpt, error) {
	switch {
	case tx.RawData != nil:
		backing := make([]uint8, len(tx.RawData))
		copy(backing, tx.RawData)
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	case tx.Int32Data != nil:
		// ONNX stores uint8 in int32 field
		backing := make([]uint8, len(tx.Int32Data))
		for i, v := range tx.Int32Data {
			backing[i] = uint8(v)
		}
		return []tensor.ConsOpt{tensor.WithBacking(backing)}, nil
	default:
		return nil, errors.New("No data found")
	}
}
