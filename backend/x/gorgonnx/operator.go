package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
)

// opKey identifies an operator by the operator set domain that defines it and
// its name. Dispatching on the name alone would let a node from a foreign
// domain run default-domain semantics under a familiar name, which is the
// failure mode this key exists to prevent.
type opKey struct {
	domain string
	name   string
}

func (k opKey) String() string {
	if k.domain == onnx.DefaultOpsetDomain {
		return k.name
	}
	return k.domain + "::" + k.name
}

// register an operator of the default (ai.onnx) domain.
func register(optype string, op func() operator) {
	registerDomain(onnx.DefaultOpsetDomain, optype, op)
}

// registerDomain registers an operator of a named operator set domain, such
// as ai.onnx.ml.
func registerDomain(domain, optype string, op func() operator) {
	operators[opKey{onnx.NormalizeOpsetDomain(domain), optype}] = op
}

var operators = map[opKey]func() operator{}

type operator interface {
	// apply analyse the graph to find the children of the node
	// then extract its gorgonia.Node references
	// and assign the result of the operation to the node n
	apply(*Graph, ...*Node) error
	// init the operator with name and attributes as carried by the onnx.Operator
	init(o onnx.Operation) error
}

// check conditions of the children.
// It returns an error is:
//  * children's length != arity
//  * if at least one of the children's pointer fo gorgoniaNode is nil
func checkCondition(children []*Node, arity int) error {
	if len(children) != arity {
		// Include more debug info
		var childNames []string
		for _, c := range children {
			if c != nil {
				childNames = append(childNames, fmt.Sprintf("%v", c.name))
			}
		}
		return fmt.Errorf("bad arity for operation (have %v, want %v, children=%v)", len(children), arity, childNames)
	}

	return checkForNil(children)
}

// check conditions of the children.
// It returns an error is:
//  * children's length < arity
//  * if at least one of the children's pointer fo gorgoniaNode is nil
func checkMinimumCondition(children []*Node, minimumArity int) error {
	if len(children) < minimumArity {
		return fmt.Errorf("bad arity for operation (have %v, want at least %v)", len(children), minimumArity)
	}

	return checkForNil(children)
}

// check that no children is nil
func checkForNil(children []*Node) error {
	// fail fast
	for i := range children {
		if children[i].gorgoniaNode == nil {
			return fmt.Errorf("at least one of the children node is nil")
		}
	}

	return nil
}
