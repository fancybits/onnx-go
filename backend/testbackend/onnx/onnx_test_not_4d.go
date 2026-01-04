package onnxtest

// this file is auto-generated... DO NOT EDIT

import (
	"github.com/owulveryck/onnx-go/backend/testbackend"
	"gorgonia.org/tensor"
	"math"
)

var _ = math.NaN // prevent unused import error

func init() {
	testbackend.Register("Not", "TestNot4d", NewTestNot4d)
}

// NewTestNot4d version: 3.
func NewTestNot4d() *testbackend.TestCase {
	return &testbackend.TestCase{
		OpType: "Not",
		Title:  "TestNot4d",
		ModelB: []byte{0x8, 0x3, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x58, 0xa, 0xd, 0xa, 0x1, 0x78, 0x12, 0x3, 0x6e, 0x6f, 0x74, 0x22, 0x3, 0x4e, 0x6f, 0x74, 0x12, 0xb, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x6e, 0x6f, 0x74, 0x5f, 0x34, 0x64, 0x5a, 0x1b, 0xa, 0x1, 0x78, 0x12, 0x16, 0xa, 0x14, 0x8, 0x9, 0x12, 0x10, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x4, 0xa, 0x2, 0x8, 0x5, 0xa, 0x2, 0x8, 0x6, 0x62, 0x1d, 0xa, 0x3, 0x6e, 0x6f, 0x74, 0x12, 0x16, 0xa, 0x14, 0x8, 0x9, 0x12, 0x10, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x4, 0xa, 0x2, 0x8, 0x5, 0xa, 0x2, 0x8, 0x6, 0x42, 0x4, 0xa, 0x0, 0x10, 0x1},

		/*

		   &ir.NodeProto{
		     Input:     []string{"x"},
		     Output:    []string{"not"},
		     Name:      "",
		     OpType:    "Not",
		     Attributes: ([]*ir.AttributeProto) <nil>
		   ,
		   },


		*/

		Input: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(3, 4, 5, 6),
				tensor.WithBacking([]bool{true, false, true, false, false, false, false, true, false, true, true, false, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, false, false, true, true, true, true, false, true, false, true, true, false, true, true, true, false, true, true, false, false, false, true, true, true, false, true, false, true, false, true, true, false, true, false, false, true, true, true, true, true, false, true, false, false, false, true, false, false, false, false, true, false, false, false, false, true, true, true, false, true, false, false, true, true, true, true, true, false, false, true, false, false, false, true, false, false, false, false, true, false, false, true, false, true, false, true, false, false, true, false, true, true, true, true, false, false, true, true, true, false, false, false, true, false, true, true, true, false, false, false, false, true, true, false, false, false, false, true, false, true, true, true, false, false, false, false, false, true, false, true, true, false, true, true, false, false, false, false, false, false, true, true, false, false, true, false, true, false, true, true, false, false, false, true, false, false, false, false, false, true, true, true, false, true, true, false, true, true, false, false, true, false, false, true, false, true, false, true, true, true, false, true, true, true, false, false, true, false, false, true, true, true, false, true, false, true, false, false, true, false, false, false, true, false, false, true, false, false, false, true, false, false, false, false, false, false, true, false, false, false, false, true, false, true, true, false, true, false, false, true, false, true, false, true, true, false, false, false, false, false, false, true, true, true, false, false, true, false, false, false, false, false, false, true, true, true, true, true, false, true, false, false, false, false, true, false, true, false, true, true, true, false, false, true, false, true, false, false, true, false, false, false, true, true, true, false, false, false, false, false, true, true, true, true, false, false, true, true, false, true, true, true, false, false, false, true, false, false, true, true, true, true, true, true, true, true, false, true, false, true, true, false, false, true, false, false}),
			),
		},
		ExpectedOutput: []tensor.Tensor{

			tensor.New(
				tensor.WithShape(3, 4, 5, 6),
				tensor.WithBacking([]bool{false, true, false, true, true, true, true, false, true, false, false, true, false, false, false, true, true, false, true, false, false, false, false, false, false, false, false, false, false, true, true, false, true, false, true, true, false, false, false, false, true, false, true, false, false, true, false, false, false, true, false, false, true, true, true, false, false, false, true, false, true, false, true, false, false, true, false, true, true, false, false, false, false, false, true, false, true, true, true, false, true, true, true, true, false, true, true, true, true, false, false, false, true, false, true, true, false, false, false, false, false, true, true, false, true, true, true, false, true, true, true, true, false, true, true, false, true, false, true, false, true, true, false, true, false, false, false, false, true, true, false, false, false, true, true, true, false, true, false, false, false, true, true, true, true, false, false, true, true, true, true, false, true, false, false, false, true, true, true, true, true, false, true, false, false, true, false, false, true, true, true, true, true, true, false, false, true, true, false, true, false, true, false, false, true, true, true, false, true, true, true, true, true, false, false, false, true, false, false, true, false, false, true, true, false, true, true, false, true, false, true, false, false, false, true, false, false, false, true, true, false, true, true, false, false, false, true, false, true, false, true, true, false, true, true, true, false, true, true, false, true, true, true, false, true, true, true, true, true, true, false, true, true, true, true, false, true, false, false, true, false, true, true, false, true, false, true, false, false, true, true, true, true, true, true, false, false, false, true, true, false, true, true, true, true, true, true, false, false, false, false, false, true, false, true, true, true, true, false, true, false, true, false, false, false, true, true, false, true, false, true, true, false, true, true, true, false, false, false, true, true, true, true, true, false, false, false, false, true, true, false, false, true, false, false, false, true, true, true, false, true, true, false, false, false, false, false, false, false, false, true, false, true, false, false, true, true, false, true, true}),
			),
		},
	}
}
