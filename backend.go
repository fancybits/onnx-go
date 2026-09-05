package onnx

import (
	"gonum.org/v1/gonum/graph"
)

// Backend represent any backend able to receive a computation graph
type Backend interface {
	OperationCarrier
	graph.DirectedWeightedBuilder
}

// Operation defined by its name, the operator set domain it belongs to,
// and its attributes.
//
// Domain is the ONNX operator set domain of the node ("" for the default
// ai.onnx set, "ai.onnx.ml" for the traditional-ML set). It is normalized:
// the alias "ai.onnx" is rewritten to "". Backends must dispatch on the
// (Domain, Name) pair — an operator name alone does not identify semantics,
// because the same name may be defined by more than one domain.
type Operation struct {
	Name       string
	Domain     string
	Attributes map[string]interface{}
}

// DefaultOpsetDomain is the ONNX default operator set domain. The empty
// string and "ai.onnx" both denote it.
const DefaultOpsetDomain = ""

// NormalizeOpsetDomain maps the "ai.onnx" alias onto the default domain and
// leaves every other domain untouched.
func NormalizeOpsetDomain(domain string) string {
	if domain == "ai.onnx" {
		return DefaultOpsetDomain
	}
	return domain
}

// OpsetChecker is implemented by backends that constrain which operator set
// versions they accept. Decoding calls it once per opset import declared by
// the model, before any operation is applied, so that an unsupported domain
// or version is reported at the model boundary instead of being silently
// dispatched with default-domain semantics.
//
// Backends that do not implement it accept every declared opset.
type OpsetChecker interface {
	CheckOpset(domain string, version int64) error
}

// OperationCarrier should be a method of the graph
// because the operation needs the topology of the graph
// to check the arity of the node for example
type OperationCarrier interface {
	// ApplyOperation on the graph nodes
	// graph.Node is an array because it allows to handle multiple output
	// for example a split operation returns n nodes...
	ApplyOperation(Operation, ...graph.Node) error
}
