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

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast

func init() {
	register("Cast", newCast)
}

// castOp is a custom Gorgonia Op that converts tensor types
type castOp struct {
	to tensor.Dtype
}

func (c *castOp) Arity() int { return 1 }

func (c *castOp) Type() hm.Type {
	// Use same type variable for input/output to satisfy Gorgonia's type inference
	// The actual type conversion happens at runtime in Do()
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (c *castOp) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, fmt.Errorf("cast: nil input shape")
	}
	s, ok := inputs[0].(tensor.Shape)
	if !ok {
		return nil, fmt.Errorf("cast: expected tensor.Shape, got %T", inputs[0])
	}
	return s.Clone(), nil
}

func (c *castOp) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("cast: expected 1 input, got %d", len(inputs))
	}

	input, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, fmt.Errorf("cast: only dense tensors are supported")
	}

	// If already the target type, return as-is
	if input.Dtype() == c.to {
		return input.Clone().(tensor.Tensor), nil
	}

	// Create output tensor with target type
	result := tensor.New(tensor.WithShape(input.Shape()...), tensor.Of(c.to))

	// Convert values using flat index access
	size := input.Shape().TotalSize()
	for i := 0; i < size; i++ {
		val := input.Get(i)
		converted := convertValue(val, c.to)
		result.Set(i, converted)
	}

	return result, nil
}

func convertValue(val interface{}, to tensor.Dtype) interface{} {
	// Convert to float64 as intermediate
	var f float64
	switch v := val.(type) {
	case float32:
		f = float64(v)
	case float64:
		f = v
	case int:
		f = float64(v)
	case int8:
		f = float64(v)
	case int16:
		f = float64(v)
	case int32:
		f = float64(v)
	case int64:
		f = float64(v)
	case uint8:
		f = float64(v)
	case uint16:
		f = float64(v)
	case uint32:
		f = float64(v)
	case uint64:
		f = float64(v)
	case bool:
		if v {
			f = 1
		} else {
			f = 0
		}
	default:
		return val // Can't convert, return as-is
	}

	// Convert to target type
	switch to {
	case tensor.Float32:
		return float32(f)
	case tensor.Float64:
		return f
	case tensor.Int:
		return int(f)
	case tensor.Int8:
		return int8(f)
	case tensor.Int16:
		return int16(f)
	case tensor.Int32:
		return int32(f)
	case tensor.Int64:
		return int64(f)
	case tensor.Uint8:
		return uint8(f)
	case tensor.Uint16:
		return uint16(f)
	case tensor.Uint32:
		return uint32(f)
	case tensor.Uint64:
		return uint64(f)
	case tensor.Bool:
		return f != 0
	default:
		return val
	}
}

func (c *castOp) ReturnsPtr() bool     { return false }
func (c *castOp) CallsExtern() bool    { return false }
func (c *castOp) OverwritesInput() int { return -1 }

func (c *castOp) WriteHash(h hash.Hash) {
	binary.Write(h, binary.LittleEndian, []byte("cast"))
	binary.Write(h, binary.LittleEndian, []byte(c.to.String()))
}

func (c *castOp) Hashcode() uint32 {
	h := fnv.New32a()
	c.WriteHash(h)
	return h.Sum32()
}

func (c *castOp) String() string { return fmt.Sprintf("Cast[%v]", c.to) }

type cast struct {
	to tensor.Dtype
}

func newCast() operator {
	return &cast{}
}

func (c *cast) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}

	input := children[0].gorgoniaNode

	// If the input type already matches, pass through. Like Identity, this
	// aliases the input exactly, so it carries the input's value and provenance
	// with it — dropping them would sever a constant chain that merely happens
	// to cast a value to the type it already has.
	if input.Dtype() == c.to {
		n.gorgoniaNode = input
		n.t = children[0].t
		n.constant = children[0].constant
		return nil
	}

	// If the input is a genuine compile-time constant, perform the cast
	// immediately and create a new constant. This is important for shape
	// computations that need values at graph construction time. A graph input
	// also has a value bound to it, but casting that would bake one run's data
	// into the graph, so it goes down the symbolic path below instead.
	if inputT := constTensorFromNode(children[0]); inputT != nil {
		op := &castOp{to: c.to}
		result, err := op.Do(inputT)
		if err != nil {
			return fmt.Errorf("cast constant: %w", err)
		}
		if t, ok := result.(tensor.Tensor); ok {
			n.t = t
			n.constant = true
		}
		n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, result, gorgonia.WithName(getUniqNodeName("cast_const")))
		return nil
	}

	// Create a tensor type with the target dtype
	tt := gorgonia.TensorType{Dims: input.Dims(), Of: c.to}

	// Create the cast operation node with explicit type and shape
	op := &castOp{to: c.to}
	n.gorgoniaNode = gorgonia.NewUniqueNode(
		gorgonia.WithType(tt),
		gorgonia.WithOp(op),
		gorgonia.WithChildren(gorgonia.Nodes{input}),
		gorgonia.In(g.exprgraph),
		gorgonia.WithShape(input.Shape()...),
	)
	return nil
}

func (c *cast) init(o onnx.Operation) error {
	to, ok := o.Attributes["to"]
	if !ok {
		return fmt.Errorf("cast: required attribute 'to' not found")
	}

	toInt, ok := to.(int64)
	if !ok {
		return fmt.Errorf("cast: 'to' attribute is not an int64")
	}

	// Map ONNX TensorProto_DataType to tensor.Dtype
	switch toInt {
	case 1: // FLOAT
		c.to = tensor.Float32
	case 2: // UINT8
		c.to = tensor.Uint8
	case 3: // INT8
		c.to = tensor.Int8
	case 4: // UINT16
		c.to = tensor.Uint16
	case 5: // INT16
		c.to = tensor.Int16
	case 6: // INT32
		c.to = tensor.Int32
	case 7: // INT64
		c.to = tensor.Int64
	case 9: // BOOL
		c.to = tensor.Bool
	case 11: // DOUBLE
		c.to = tensor.Float64
	case 12: // UINT32
		c.to = tensor.Uint32
	case 13: // UINT64
		c.to = tensor.Uint64
	default:
		return fmt.Errorf("cast: unsupported target type %d", toInt)
	}

	return nil
}
