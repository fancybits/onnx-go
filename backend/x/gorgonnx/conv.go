package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

func init() {
	register("Conv", newConv)
}

func newConv() operator {
	return &conv{}
}

// conv to be compatible with:
//    https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// and
//    https://godoc.org/gorgonia.org/gorgonia#Conv2d
// test with go test -run=TestONNX/Conv
type conv struct {
	autopad     string
	pad         []int
	stride      []int
	dilation    []int
	group       int
	kernelShape tensor.Shape
}

func (c *conv) apply(g *Graph, ns ...*Node) error {
	n := ns[0]
	children := getOrderedChildren(g.g, n)
	var err error
	if len(children) < 2 || len(children) > 3 {
		return errors.New("Conv: bad arity")
	}
	err = c.autopadding(children)
	if err != nil {
		return err
	}
	// If kernelShape wasn't set from attribute, derive it from the filter tensor
	if c.kernelShape == nil {
		filterShape := children[1].gorgoniaNode.Shape()
		if len(filterShape) >= 2 {
			c.kernelShape = filterShape[len(filterShape)-2:] // Last 2 dimensions (H, W)
		}
	}

	var convN *gorgonia.Node

	if c.group > 1 {
		// Grouped convolution: split input and filter, apply conv per group, concatenate
		convN, err = c.applyGroupedConv(children[0].gorgoniaNode, children[1].gorgoniaNode)
		if err != nil {
			return &errOp{"conv", err}
		}
	} else {
		// Standard convolution
		convN, err = nnops.Conv2d(
			children[0].gorgoniaNode,
			children[1].gorgoniaNode,
			c.kernelShape,
			c.pad,
			c.stride,
			c.dilation)
		if err != nil {
			return &errOp{"conv", err}
		}
	}

	if len(children) == 3 {
		b, err := gorgonia.Reshape(children[2].gorgoniaNode, []int{1, children[2].gorgoniaNode.Shape()[0], 1, 1})
		if err != nil {
			return &errOp{
				"conv",
				err,
			}
		}
		convA, ba, err := gorgonia.Broadcast(convN, b, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
		if err != nil {
			return &errOp{
				"conv",
				err,
			}
		}
		n.gorgoniaNode, err = gorgonia.Add(convA, ba)
		if err != nil {
			return &errOp{
				"conv",
				err,
			}
		}
	} else {
		n.gorgoniaNode = convN
	}
	return nil
}

// applyGroupedConv implements grouped convolution by splitting input and filter,
// applying separate convolutions per group, and concatenating the results.
// This handles depthwise convolutions (group == in_channels) and general grouped convolutions.
func (c *conv) applyGroupedConv(input, filter *gorgonia.Node) (*gorgonia.Node, error) {
	inputShape := input.Shape()
	filterShape := filter.Shape()

	// Debug: print input shapes
	if inputShape.Dims() != 4 {
		return nil, fmt.Errorf("grouped conv: input should be 4D, got %dD (shape=%v)", inputShape.Dims(), inputShape)
	}
	if filterShape.Dims() != 4 {
		return nil, fmt.Errorf("grouped conv: filter should be 4D, got %dD (shape=%v)", filterShape.Dims(), filterShape)
	}

	// Input shape: (N, C_in, H, W)
	// Filter shape: (C_out, C_in/group, Kh, Kw)
	inChannels := inputShape[1]
	outChannels := filterShape[0]
	channelsPerGroupIn := inChannels / c.group
	channelsPerGroupOut := outChannels / c.group

	// Validate dimensions
	if inChannels%c.group != 0 {
		return nil, errors.New("Conv: input channels must be divisible by group")
	}
	if outChannels%c.group != 0 {
		return nil, errors.New("Conv: output channels must be divisible by group")
	}

	groupOutputs := make([]*gorgonia.Node, c.group)

	for grp := 0; grp < c.group; grp++ {
		// Slice input channels for this group
		inStart := grp * channelsPerGroupIn
		inEnd := inStart + channelsPerGroupIn

		inputSlice, err := gorgonia.Slice(input,
			nil, // batch: all
			tensor.S(inStart, inEnd), // channels: this group
			nil, // height: all
			nil, // width: all
		)
		if err != nil {
			return nil, fmt.Errorf("slicing input: %w", err)
		}

		// Check if slice squeezed dimensions - gorgonia may squeeze singleton dims
		if inputSlice.Shape().Dims() != 4 {
			// Reshape to restore 4D: (batch, channels_per_group, height, width)
			newShape := tensor.Shape{inputShape[0], channelsPerGroupIn, inputShape[2], inputShape[3]}
			inputSlice, err = gorgonia.Reshape(inputSlice, newShape)
			if err != nil {
				return nil, fmt.Errorf("reshaping input slice from %v to %v: %w", inputSlice.Shape(), newShape, err)
			}
		}

		// Slice filter for this group
		outStart := grp * channelsPerGroupOut
		outEnd := outStart + channelsPerGroupOut

		filterSlice, err := gorgonia.Slice(filter,
			tensor.S(outStart, outEnd), // output channels: this group
			nil, // input channels per group: all
			nil, // kernel height: all
			nil, // kernel width: all
		)
		if err != nil {
			return nil, fmt.Errorf("slicing filter: %w", err)
		}

		// Check if slice squeezed dimensions
		if filterSlice.Shape().Dims() != 4 {
			// Reshape to restore 4D: (out_channels_per_group, in_channels_per_group, kH, kW)
			newShape := tensor.Shape{channelsPerGroupOut, filterShape[1], filterShape[2], filterShape[3]}
			filterSlice, err = gorgonia.Reshape(filterSlice, newShape)
			if err != nil {
				return nil, fmt.Errorf("reshaping filter slice from %v to %v: %w", filterSlice.Shape(), newShape, err)
			}
		}

		// Apply convolution for this group
		groupConv, err := nnops.Conv2d(
			inputSlice,
			filterSlice,
			c.kernelShape,
			c.pad,
			c.stride,
			c.dilation)
		if err != nil {
			return nil, fmt.Errorf("group %d: inputSlice=%v, filterSlice=%v, kernelShape=%v: %w",
				grp, inputSlice.Shape(), filterSlice.Shape(), c.kernelShape, err)
		}

		groupOutputs[grp] = groupConv
	}

	// Concatenate all group outputs along channel axis
	result, err := gorgonia.Concat(1, groupOutputs...)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// autopadding needs to be applied now because it needs to be aware of the shape of the nodes
func (c *conv) autopadding(children []*Node) error {
	switch c.autopad {
	case "NOTSET":
	case "":
	case "VALID":
		c.pad = []int{0, 0}
	case "SAME_UPPER":
		for i, v := range children[0].gorgoniaNode.Shape()[2:] {
			outputD := ceilDivInt(v, c.stride[i])
			c.pad[i] = (outputD-1)*c.stride[i] + (c.kernelShape[i]-1)*c.dilation[i] + 1 - v
			if c.pad[i] < 0 {
				c.pad[i] = 0
			}
			if c.pad[i]%2 != 0 {
				return &onnx.ErrNotImplemented{
					Operator:       "conv",
					AttributeName:  "pads",
					AttributeValue: c.pad[i],
					Message:        "Asymetric padding",
				}
			}
			c.pad[i] /= 2
		}
	default:
		return &onnx.ErrNotImplemented{
			Operator: "Conv",
			Message:  "auto_pad " + c.autopad + " not implemented",
		}
	}
	return nil
}

func (c *conv) init(o onnx.Operation) error {
	autoPad, ok := o.Attributes["auto_pad"]
	if ok {
		c.autopad = autoPad.(string)
	}
	c.initKernelShape(o)
	err := c.initPads(o)
	if err != nil {
		return err
	}
	c.initStrides(o)
	c.initDilations(o)
	c.initGroup(o)
	return nil
}

func (c *conv) initGroup(o onnx.Operation) {
	c.group = 1
	group, ok := o.Attributes["group"]
	if ok {
		if g, ok := group.(int64); ok {
			c.group = int(g)
		}
	}
}

func (c *conv) initKernelShape(o onnx.Operation) {
	kernelShape, ok := o.Attributes["kernel_shape"]
	if ok {
		if kernelShape, ok := kernelShape.([]int64); ok {
			c.kernelShape = make([]int, len(kernelShape))
			for i := 0; i < len(kernelShape); i++ {
				c.kernelShape[i] = int(kernelShape[i])
			}
		}
	}
}

func (c *conv) initPads(o onnx.Operation) error {
	c.pad = []int{0, 0}
	pad, ok := o.Attributes["pads"]
	if ok {
		if pad, ok := pad.([]int64); ok {

			// ONNX pads format: [x1_begin, x2_begin, x1_end, x2_end]
			// For symmetric padding, begin must equal end for each dimension:
			// pad[0] == pad[2] (dimension 1) and pad[1] == pad[3] (dimension 2)
			if len(pad) == 4 && (pad[0] != pad[2] || pad[1] != pad[3]) {
				return &onnx.ErrNotImplemented{
					Operator:       "Conv",
					AttributeName:  "pads",
					AttributeValue: pad,
					Message:        "Asymetric padding",
				}
			}

			if len(pad) == 4 {
				// Since padding is symmetric (validated above),
				// use begin padding for each dimension
				c.pad[0] = int(pad[0]) // height (dimension 1)
				c.pad[1] = int(pad[1]) // width (dimension 2)
			} else if len(pad) == 2 {
				for i := 0; i < 2; i++ {
					c.pad[i] = int(pad[i])
				}
			}
		}
	}
	return nil
}

func (c *conv) initStrides(o onnx.Operation) {
	c.stride = []int{1, 1}
	stride, ok := o.Attributes["strides"]
	if ok {
		if stride, ok := stride.([]int64); ok {
			if len(stride) == 4 {
				for i := 0; i < 2; i++ {
					c.stride[i] = int(stride[2*i])
				}
			} else if len(stride) == 2 {
				for i := 0; i < 2; i++ {
					c.stride[i] = int(stride[i])
				}
			}
		}
	}
}

func (c *conv) initDilations(o onnx.Operation) {
	c.dilation = []int{1, 1}
	dilation, ok := o.Attributes["dilations"]
	if ok {
		if dilation, ok := dilation.([]int64); ok {
			c.dilation = make([]int, len(dilation))
			for i := 0; i < len(dilation); i++ {
				c.dilation[i] = int(dilation[i])
			}
		}
	}
}
