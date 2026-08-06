package onnx

import (
	"reflect"

	"github.com/gogo/protobuf/proto"
	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// Model is a wrapper around a computation graph.
// Input and Output are containing the ID of the corresponding nodes.
type Model struct {
	backend     Backend
	dbByName    map[string]graph.Node
	Input       []int64
	InputNames  []string
	Output      []int64
	OutputNames []string
}

// NewModel with dst as backend.
// dst should be a non-nil pointer.
func NewModel(dst Backend) *Model {
	return &Model{
		dbByName: make(map[string]graph.Node, 0),
		backend:  dst,
	}
}

// UnmarshalBinary decodes the binary data in onnx format into the model
func (m *Model) UnmarshalBinary(data []byte) error {
	pbModel := &ir.ModelProto{}
	err := proto.Unmarshal(data, pbModel)
	if err != nil {
		return err
	}
	return m.decodeProto(pbModel)
}

// GetNodeByName is a utility method that returns a node of the computation graph
func (m *Model) GetNodeByName(name string) (graph.Node, bool) {
	n, ok := m.dbByName[name]
	return n, ok
}

func (m *Model) processValue(io *ir.ValueInfoProto) (graph.Node, error) {
	return processValueInto(m.backend, m.dbByName, io)
}

// processValueInto is processValue generalized over any backend + name db.
func processValueInto(dst Backend, db map[string]graph.Node, io *ir.ValueInfoProto) (graph.Node, error) {
	if io == nil {
		return nil, errors.New("cannot process nil value")
	}
	var opts []tensor.ConsOpt
	n := dst.NewNode()
	if _, ok := n.(Namer); ok {
		n.(Namer).SetName(io.Name)
	}
	dst.AddNode(n)
	db[io.Name] = n
	if io.Type == nil {
		return n, nil
	}
	if _, ok := n.(DataCarrier); !ok {
		return n, nil
	}
	ttype := io.Type.GetTensorType()
	if ttype.GetShape() != nil {
		var shape []int
		for i := range ttype.Shape.Dim {
			_, ok := ttype.Shape.Dim[i].GetValue().(*ir.TensorShapeProto_Dimension_DimValue)
			if ok {
				shape = append(shape, int(ttype.Shape.Dim[i].GetDimValue()))
			}
		}
		opts = append(opts, tensor.WithShape(shape...))
	}
	dtype, err := ir.TensorProto_DataType(ttype.GetElemType()).Dtype()
	if err != nil {
		return n, err
	}
	opts = append(opts, tensor.Of(dtype))
	t := tensor.New(opts...)
	err = n.(DataCarrier).SetTensor(t)
	if err != nil {
		return n, err
	}

	return n, nil
}

// decodeProto decode a protobuf definition inside the model
func (m *Model) decodeProto(model *ir.ModelProto) error {
	rv := reflect.ValueOf(m.backend)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return &InvalidUnmarshalError{reflect.TypeOf(m.backend)}
	}

	if model == nil {
		return errModelIsNil
	}
	if model.Graph == nil {
		return errGraphIsNil
	}
	if len(model.Graph.Node) == 0 {
		return errEmptyGraph
	}
	if len(model.Graph.Output) == 0 {
		return errGraphNoIO
	}
	if len(model.Graph.Input)+len(model.Graph.Output) == 0 {
		return errGraphNoIO
	}
	err := m.applyModelProtoGraph(model)
	if err != nil {
		return err
	}
	return nil
}

// applyModelProtoGraph apply model proto graph tensors to model
func (m *Model) applyModelProtoGraph(model *ir.ModelProto) error {
	m.Input = make([]int64, len(model.Graph.Input))
	m.InputNames = make([]string, len(model.Graph.Input))
	m.Output = make([]int64, len(model.Graph.Output))
	m.OutputNames = make([]string, len(model.Graph.Output))
	m.dbByName = make(map[string]graph.Node, len(model.Graph.Output)+len(model.Graph.Input))
	// Well...
	for i, io := range model.Graph.Input {
		n, err := m.processValue(io)
		if err != nil {
			return err
		}
		m.Input[i] = n.ID()
		m.InputNames[i] = io.Name
	}
	for _, io := range model.Graph.ValueInfo {
		_, err := m.processValue(io)
		if err != nil {
			return err
		}
	}
	for i, io := range model.Graph.Output {
		n, err := m.processValue(io)
		if err != nil {
			return err
		}
		m.Output[i] = n.ID()
		m.OutputNames[i] = io.Name
	}
	err := m.applyModelProtoGraphTensors(model)
	if err != nil {
		return err
	}
	err = m.applyModelProtoGraphNodeOperations(model)
	if err != nil {
		return err
	}
	return nil
}

// applyModelProtoGraphTensors apply model proto graph tensors to model
func (m *Model) applyModelProtoGraphTensors(model *ir.ModelProto) error {
	return applyGraphTensors(m.backend, m.dbByName, model.Graph, func(n graph.Node) {
		// Remove it from the input
		// find the ID
		for i := 0; i < len(m.Input); i++ {
			if m.Input[i] == n.ID() {
				m.Input = append(m.Input[:i], m.Input[i+1:]...)
			}
		}
	})
}

// applyGraphTensors sets initializer tensors into db-registered nodes,
// creating them if absent. onInit (may be nil) is called per initializer node —
// Model uses it to prune initializers from its Input list.
func applyGraphTensors(dst Backend, db map[string]graph.Node, g *ir.GraphProto, onInit func(graph.Node)) error {
	for _, tensorProto := range g.GetInitializer() {
		name := tensorProto.GetName()
		if name == "" {
			return errors.New("initializer should have a name")
		}
		n, ok := db[name]
		if !ok {
			n = insertNode(dst, db, name)
		}
		if onInit != nil {
			onInit(n)
		}
		if _, ok := n.(DataCarrier); !ok {
			continue
		}
		t, err := tensorProto.Tensor()
		if err != nil {
			return err
		}
		err = n.(DataCarrier).SetTensor(t)
		if err != nil {
			return err
		}
		// An initializer is the only tensor in the proto that is a genuine
		// compile-time constant; record that provenance for backends that
		// fold constants.
		if cm, ok := n.(ConstMarker); ok {
			cm.MarkConst()
		}
	}
	return nil
}

// applyModelProtoGraphNodeOperations apply model proto graph node operations to model
func (m *Model) applyModelProtoGraphNodeOperations(model *ir.ModelProto) error {
	return applyGraphNodeOperations(m.backend, m.dbByName, model.Graph)
}

// applyGraphNodeOperations wires node inputs/outputs and calls dst.ApplyOperation,
// exactly as Model.applyModelProtoGraphNodeOperations does today.
func applyGraphNodeOperations(dst Backend, db map[string]graph.Node, g *ir.GraphProto) error {
	for _, node := range g.Node {
		outputNodes := make([]graph.Node, len(node.Output))
		for i, output := range node.Output {
			var ok bool
			var no graph.Node
			if no, ok = db[output]; !ok {
				no = dst.NewNode()
				if _, ok := no.(Namer); ok {
					no.(Namer).SetName(output)
				}
				dst.AddNode(no)
				db[output] = no
			}
			// If node is input-less, fake an input by creating an empty value
			if len(node.Input) == 0 {
				inputName := node.Name + "/input"
				_, err := processValueInto(dst, db, &ir.ValueInfoProto{
					Name: inputName,
				})
				if err != nil {
					return err
				}
				node.Input = append(node.Input, inputName)
			}
			// input should be ordered for non-commutatives operations
			for i, input := range node.Input {
				var ni graph.Node
				var ok bool
				if ni, ok = db[input]; !ok {
					ni = dst.NewNode()
					if _, ok := ni.(Namer); ok {
						ni.(Namer).SetName(input)
					}
					dst.AddNode(ni)
					db[input] = ni
				}
				e := dst.NewWeightedEdge(no, ni, float64(i))
				dst.SetWeightedEdge(e)
			}
			outputNodes[i] = no
		}

		// The graph can apply operations
		attrs, err := toOperationAttributes(node.GetAttribute())
		if err != nil {
			return err
		}
		err = dst.ApplyOperation(Operation{
			node.OpType,
			attrs,
		}, outputNodes...)
		if err != nil {
			return err
		}
	}
	return nil
}

func insertNode(dst Backend, db map[string]graph.Node, name string) graph.Node {
	n := dst.NewNode()
	if n, ok := n.(Namer); ok {
		n.SetName(name)
	}
	dst.AddNode(n)
	db[name] = n
	return n
}
