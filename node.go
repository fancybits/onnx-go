package onnx

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// Namer is a node that know its own name
type Namer interface {
	graph.Node
	SetName(string)
	GetName() string
}

// Documenter is an interface that describe any object able to document itself
type Documenter interface {
	graph.Node
	SetDescription(string)
	GetDescription() string
}

// DataCarrier is node with the ability to carry a tensor data
type DataCarrier interface {
	SetTensor(t tensor.Tensor) error
	GetTensor() tensor.Tensor
}

// ConstMarker is a node that records whether the tensor it carries is a
// compile-time constant. The decoder marks the nodes it fills from the
// graph's initializers; every other node carrying a tensor is a placeholder
// whose value is only known at run time — including graph inputs, whose
// tensor may already be bound before the backend builds its graph. Backends
// that fold constants need that distinction to avoid baking a run's input
// values into the graph. Implementing it is optional.
type ConstMarker interface {
	MarkConst()
}
