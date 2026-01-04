package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"text/template"

	"github.com/davecgh/go-spew/spew"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"gorgonia.org/tensor"
)

var (
	testdir   *string
	outputdir *string
)

func main() {
	testdir = flag.String("testpath", ".", "path to the onnx test directory")
	outputdir = flag.String("outputdir", "", "path to the outputdir")
	op := flag.String("op", "", "the operator who needs tests")
	flag.Parse()
	if *op == "" {
		flag.Usage()
		os.Exit(0)
	}
	// locate all the directories with the pattern test_op_...
	files, err := os.ReadDir(*testdir)
	if err != nil {
		log.Fatal(err)
	}
	re := regexp.MustCompile("^test_" + *op + "(_*)(.*)")
	testcases := make([]testCases, 0)
	for _, file := range files {
		if !file.IsDir() {
			return
		}
		elements := re.FindAllStringSubmatch(file.Name(), -1)
		if len(elements) == 0 {
			continue
		}
		log.Println("-->", file.Name())
		info, err := file.Info()
		if err != nil {
			log.Fatal(err)
		}
		optype, testtitle, err := processFile(info)
		if err != nil {
			log.Println(err)
			continue
		}
		// Add the file to the testcases
		testcases = append(testcases, testCases{
			optype, testtitle, ""})
	}
	/*
		output := os.Stdout
		if *outputdir != "" {
			output, err = os.Create(filepath.Join(*outputdir, "onnx_register_testcases.go"))
			if err != nil {
				log.Fatal(err)
			}
			defer output.Close()
		}
		processTemplate(testCasesTemplate, testcases, output)
	*/
}

// returns the optype and the title
func processFile(file os.FileInfo) (string, string, error) {
	defer func() {
		if err := recover(); err != nil {
			log.Println(err)
		}
	}()
	var tv testValue
	var mv modelValue
	tv.TestName = toCamelCase(file.Name())
	b, err := os.ReadFile(*testdir + file.Name() + "/model.onnx")
	if err != nil {
		return "", "", err
	}
	tv.ModelB = fmt.Sprintf("%#v", b)
	model := new(ir.ModelProto)
	err = model.XXX_Unmarshal(b)
	if err != nil {
		return "", "", err
	}
	tv.Description = fmt.Sprintf("version: %v. %v", model.GetIrVersion(), model.GetDocString())
	mv.TestName = fmt.Sprintf("%#v", file.Name())
	mv.IrVersion = fmt.Sprintf("%v", model.IrVersion)
	mv.OpsetVersion = fmt.Sprintf("%v", model.OpsetImport[0].Version)
	mv.NodeProto = make([]nodeProtoValue, len(model.Graph.Node))
	for i := range model.Graph.Node {
		mv.NodeProto[i] = nodeProtoValue{
			Input:         fmt.Sprintf("%#v", model.Graph.Node[i].Input),
			Output:        fmt.Sprintf("%#v", model.Graph.Node[i].Output),
			Name:          fmt.Sprintf("%#v", model.Graph.Node[i].Name),
			OpType:        fmt.Sprintf("%#v", model.Graph.Node[i].OpType),
			AttributeDump: spew.Sdump(model.Graph.Node[i].Attribute),
		}
	}
	tv.OpType = model.Graph.Node[0].OpType

	processModelGraphInput(model, &mv)
	processModelGraphOutput(model, &mv)

	// Use graph inputs/outputs when:
	// 1. Multi-node graphs (more than 1 node)
	// 2. Single-node graphs where graph has more inputs than the node
	//    (e.g., CastLike expanded has 2 graph inputs but Cast node has 1 input)
	node := model.GetGraph().GetNode()[0]
	useGraphInputs := len(model.GetGraph().GetNode()) > 1 ||
		len(model.Graph.Input) > len(node.GetInput()) ||
		len(model.Graph.Output) > len(node.GetOutput())

	if useGraphInputs {
		log.Printf("Using graph inputs/outputs (nodes=%d, graph inputs=%d, node inputs=%d)",
			len(model.GetGraph().GetNode()), len(model.Graph.Input), len(node.GetInput()))
		err = processGraphInputs(file.Name(), model, &tv)
		if err != nil {
			return "", "", err
		}
		err = processGraphOutputs(file.Name(), model, &tv)
		if err != nil {
			return "", "", err
		}
	} else {
		err = processModelGraphNodeInput(file.Name(), node, &tv)
		if err != nil {
			return "", "", err
		}
		err = processModelGraphNodeOutput(file.Name(), node, &tv)
		if err != nil {
			return "", "", err
		}
	}

	processModelGraphValueInfo(model, &mv)

	// TestTemplate
	output := os.Stdout
	outputTest := os.Stdout
	if *outputdir != "" {
		output, err = os.Create(filepath.Join(*outputdir, "onnx_"+file.Name()+".go"))
		if err != nil {
			return "", "", err
		}
		defer output.Close()
		outputTest, err = os.Create(filepath.Join(*outputdir, "onnx_"+file.Name()+"_test.go"))
		if err != nil {
			return "", "", err
		}
		defer outputTest.Close()
	}
	tv.ModelValue = mv
	err = processTemplate(testTemplate, tv, output)
	if err != nil {
		return "", "", err
	}
	err = processTemplate(testTestTemplate, tv, outputTest)
	if err != nil {
		return "", "", err
	}
	/*
		err = processTemplate(modelTemplate, mv, output)
		if err != nil {
			return err
		}
	*/
	return mv.NodeProto[0].OpType, tv.TestName, nil
}

func processModelGraphInput(model *ir.ModelProto, mv *modelValue) {
	mv.Input = make([]valueInfoProto, len(model.Graph.Input))
	for i := range model.Graph.Input {
		mv.Input[i] = valueInfoProto{
			Name:     model.Graph.Input[i].Name,
			ElemType: fmt.Sprintf("%v", model.Graph.Input[i].Type.GetTensorType().ElemType),
			Dims:     make([]string, len(model.Graph.Input[i].Type.GetTensorType().Shape.Dim)),
		}
		for j, v := range model.Graph.Input[i].Type.GetTensorType().Shape.Dim {
			mv.Input[i].Dims[j] = fmt.Sprintf("%v", v.GetValue().(*ir.TensorShapeProto_Dimension_DimValue).DimValue)
		}
	}
}

func processModelGraphOutput(model *ir.ModelProto, mv *modelValue) {
	mv.Output = make([]valueInfoProto, len(model.Graph.Output))
	for i := range model.Graph.Output {
		mv.Output[i] = valueInfoProto{
			Name:     model.Graph.Output[i].Name,
			ElemType: fmt.Sprintf("%v", model.Graph.Output[i].Type.GetTensorType().ElemType),
			Dims:     make([]string, len(model.Graph.Output[i].Type.GetTensorType().Shape.Dim)),
		}
		for j, v := range model.Graph.Output[i].Type.GetTensorType().Shape.Dim {
			mv.Output[i].Dims[j] = fmt.Sprintf("%v", v.GetValue().(*ir.TensorShapeProto_Dimension_DimValue).DimValue)
		}
	}
}

func processModelGraphValueInfo(model *ir.ModelProto, mv *modelValue) {
	mv.ValueInfo = make([]valueInfoProto, len(model.Graph.ValueInfo))
	for i := range model.Graph.ValueInfo {
		mv.ValueInfo[i] = valueInfoProto{
			Name:     model.Graph.ValueInfo[i].Name,
			ElemType: fmt.Sprintf("%v", model.Graph.ValueInfo[i].Type.GetTensorType().ElemType),
			Dims:     make([]string, len(model.Graph.ValueInfo[i].Type.GetTensorType().Shape.Dim)),
		}
		for j, v := range model.Graph.ValueInfo[i].Type.GetTensorType().Shape.Dim {
			mv.ValueInfo[i].Dims[j] = fmt.Sprintf("%v", v.GetValue().(*ir.TensorShapeProto_Dimension_DimValue).DimValue)

		}
	}
}

func processModelGraphNodeInput(filename string, node *ir.NodeProto, tv *testValue) error {
	tv.Input = make([]iO, len(node.GetInput()))
	for i := range node.GetInput() {
		// Open the tensorproto sample file
		filepath := fmt.Sprintf("%v%v/test_data_set_0/input_%v.pb", *testdir, filename, i)
		b, err := os.ReadFile(filepath)
		if err != nil {
			return err
		}
		sampleTestData := new(ir.TensorProto)
		err = sampleTestData.XXX_Unmarshal(b)
		if err != nil {
			return err
		}
		t, err := sampleTestData.Tensor()
		if err != nil {
			return err
		}
		data := formatTensorData(t)
		shape := fmt.Sprintf("%#v", t.Shape())
		if len(t.Shape()) == 0 {
			shape = "(1)"
		}
		tv.Input[i] = iO{
			Shape: shape,
			Data:  data,
		}
	}
	return nil
}

func processModelGraphNodeOutput(filename string, node *ir.NodeProto, tv *testValue) error {
	tv.ExpectedOutput = make([]iO, len(node.GetOutput()))
	for i := range node.Output {
		// Open the tensorproto sample file
		filepath := fmt.Sprintf("%v%v/test_data_set_0/output_%v.pb", *testdir, filename, i)
		b, err := os.ReadFile(filepath)
		if err != nil {
			return err
		}
		sampleTestData := new(ir.TensorProto)
		err = sampleTestData.XXX_Unmarshal(b)
		if err != nil {
			return err

		}
		t, err := sampleTestData.Tensor()
		if err != nil {
			return err
		}
		shape := fmt.Sprintf("%#v", t.Shape())
		data := formatTensorData(t)
		if len(t.Shape()) == 0 {
			shape = "(1)"
		}
		tv.ExpectedOutput[i] = iO{
			Shape: shape,
			Data:  data,
		}
	}
	return nil
}

// processGraphInputs reads inputs for multi-node graphs using graph inputs
func processGraphInputs(filename string, model *ir.ModelProto, tv *testValue) error {
	numInputs := len(model.Graph.Input)
	tv.Input = make([]iO, numInputs)
	for i := 0; i < numInputs; i++ {
		// Open the tensorproto sample file
		filepath := fmt.Sprintf("%v%v/test_data_set_0/input_%v.pb", *testdir, filename, i)
		b, err := os.ReadFile(filepath)
		if err != nil {
			return err
		}
		sampleTestData := new(ir.TensorProto)
		err = sampleTestData.XXX_Unmarshal(b)
		if err != nil {
			return err
		}
		t, err := sampleTestData.Tensor()
		if err != nil {
			// Handle empty tensors (e.g., CastLike "like" input which just carries dtype)
			if err.Error() == "No data found" {
				// Create an empty tensor with the appropriate dtype
				shape, data := createEmptyTensorCode(sampleTestData)
				tv.Input[i] = iO{
					Shape: shape,
					Data:  data,
				}
				continue
			}
			return err
		}
		data := formatTensorData(t)
		shape := fmt.Sprintf("%#v", t.Shape())
		if len(t.Shape()) == 0 {
			shape = "(1)"
		}
		tv.Input[i] = iO{
			Shape: shape,
			Data:  data,
		}
	}
	return nil
}

// processGraphOutputs reads outputs for multi-node graphs using graph outputs
func processGraphOutputs(filename string, model *ir.ModelProto, tv *testValue) error {
	numOutputs := len(model.Graph.Output)
	tv.ExpectedOutput = make([]iO, numOutputs)
	for i := 0; i < numOutputs; i++ {
		// Open the tensorproto sample file
		filepath := fmt.Sprintf("%v%v/test_data_set_0/output_%v.pb", *testdir, filename, i)
		b, err := os.ReadFile(filepath)
		if err != nil {
			return err
		}
		sampleTestData := new(ir.TensorProto)
		err = sampleTestData.XXX_Unmarshal(b)
		if err != nil {
			return err
		}
		t, err := sampleTestData.Tensor()
		if err != nil {
			return err
		}
		data := formatTensorData(t)
		shape := fmt.Sprintf("%#v", t.Shape())
		if len(t.Shape()) == 0 {
			shape = "(1)"
		}
		tv.ExpectedOutput[i] = iO{
			Shape: shape,
			Data:  data,
		}
	}
	return nil
}

func processTemplate(t *template.Template, v interface{}, output io.Writer) error {
	var buf bytes.Buffer
	if err := t.Execute(&buf, v); err != nil {
		log.Fatal(err)
	}
	p, err := format.Source(buf.Bytes())
	if err != nil {
		log.Fatal("Cannot format", err)
	}
	_, err = output.Write(p)
	return err
}

var link = regexp.MustCompile("(^[A-Za-z])|_([A-Za-z0-9])")

func toCamelCase(str string) string {
	return link.ReplaceAllStringFunc(str, func(s string) string {
		return strings.ToUpper(strings.Replace(s, "_", "", -1))
	})
}

// formatTensorData formats tensor data as a Go literal, handling NaN and Inf values
func formatTensorData(t tensor.Tensor) string {
	data := t.Data()
	switch d := data.(type) {
	case []float32:
		return formatFloat32Slice(d)
	case []float64:
		return formatFloat64Slice(d)
	case []bool:
		return fmt.Sprintf("%#v", d)
	case []int64:
		return fmt.Sprintf("%#v", d)
	case []int32:
		return fmt.Sprintf("%#v", d)
	case []int16:
		return fmt.Sprintf("%#v", d)
	case []int8:
		return fmt.Sprintf("%#v", d)
	case []uint64:
		return fmt.Sprintf("%#v", d)
	case []uint32:
		return fmt.Sprintf("%#v", d)
	case []uint16:
		return fmt.Sprintf("%#v", d)
	case []uint8:
		return fmt.Sprintf("%#v", d)
	// Handle scalar values (when tensor has shape () or (1) for a single element)
	case float32:
		return formatFloat32Slice([]float32{d})
	case float64:
		return formatFloat64Slice([]float64{d})
	case bool:
		return fmt.Sprintf("[]bool{%v}", d)
	case int64:
		return fmt.Sprintf("[]int64{%v}", d)
	case int32:
		return fmt.Sprintf("[]int32{%v}", d)
	case int16:
		return fmt.Sprintf("[]int16{%v}", d)
	case int8:
		return fmt.Sprintf("[]int8{%v}", d)
	case uint64:
		return fmt.Sprintf("[]uint64{%v}", d)
	case uint32:
		return fmt.Sprintf("[]uint32{%v}", d)
	case uint16:
		return fmt.Sprintf("[]uint16{%v}", d)
	case uint8:
		return fmt.Sprintf("[]uint8{%v}", d)
	default:
		return fmt.Sprintf("%#v", data)
	}
}

func formatFloat32Slice(data []float32) string {
	var b strings.Builder
	b.WriteString("[]float32{")
	for i, v := range data {
		if i > 0 {
			b.WriteString(", ")
		}
		if math.IsNaN(float64(v)) {
			b.WriteString("float32(math.NaN())")
		} else if math.IsInf(float64(v), 1) {
			b.WriteString("float32(math.Inf(1))")
		} else if math.IsInf(float64(v), -1) {
			b.WriteString("float32(math.Inf(-1))")
		} else {
			b.WriteString(fmt.Sprintf("%v", v))
		}
	}
	b.WriteString("}")
	return b.String()
}

func formatFloat64Slice(data []float64) string {
	var b strings.Builder
	b.WriteString("[]float64{")
	for i, v := range data {
		if i > 0 {
			b.WriteString(", ")
		}
		if math.IsNaN(v) {
			b.WriteString("math.NaN()")
		} else if math.IsInf(v, 1) {
			b.WriteString("math.Inf(1)")
		} else if math.IsInf(v, -1) {
			b.WriteString("math.Inf(-1)")
		} else {
			b.WriteString(fmt.Sprintf("%v", v))
		}
	}
	b.WriteString("}")
	return b.String()
}

// createEmptyTensorCode creates shape and data strings for an empty tensor
// based on its TensorProto dtype (used for CastLike "like" input)
func createEmptyTensorCode(tp *ir.TensorProto) (shape, data string) {
	// Shape is 0 for empty tensors
	shape = "(0)"

	// Map ONNX DataType to Go type
	switch tp.DataType {
	case 1: // FLOAT
		data = "[]float32{}"
	case 2: // UINT8
		data = "[]uint8{}"
	case 3: // INT8
		data = "[]int8{}"
	case 4: // UINT16
		data = "[]uint16{}"
	case 5: // INT16
		data = "[]int16{}"
	case 6: // INT32
		data = "[]int32{}"
	case 7: // INT64
		data = "[]int64{}"
	case 9: // BOOL
		data = "[]bool{}"
	case 10: // FLOAT16 - use uint16 as underlying type
		data = "[]uint16{}"
	case 11: // DOUBLE
		data = "[]float64{}"
	case 12: // UINT32
		data = "[]uint32{}"
	case 13: // UINT64
		data = "[]uint64{}"
	default:
		data = "[]float32{}" // fallback
	}
	return shape, data
}
