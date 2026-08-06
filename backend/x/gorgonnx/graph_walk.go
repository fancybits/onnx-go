package gorgonnx

import (
	"errors"
	"fmt"
	"runtime/debug"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// populateExprgraph by walking through the graph
func (g *Graph) populateExprgraph() error {
	if len(g.groups) == 0 {
		return errors.New("cannot populate the graph because ApplyOperation have not been called")
	}

	// Walk the graph
	itN := g.Nodes()
	for itN.Next() {
		// if the node is a "tensor", set it!
		n := itN.Node().(*Node)
		if n.gorgoniaNode == nil && n.operation == nil {
			if n.t != nil {
				n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, n.t, gorgonia.WithName(getUniqNodeName("node")))
			} else {
				n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, 0, gorgonia.WithName(getUniqNodeName("node")))
			}
		}
	}
	nodes := make([][]*Node, len(g.groups))
	copy(nodes, g.groups)
	for len(nodes) > 0 {
		initialLen := len(nodes)
		// Every errNotReady of the pass, kept so that a pass which makes no
		// progress at all can report why instead of "infinite loop". Keeping
		// only the last would name one stuck group and hide the others, which
		// is the wrong end of the problem to look at when several are waiting
		// on each other.
		var deferred []error
		for i := 0; i < len(nodes); i++ {
			nilChild := false
			for _, n := range nodes[i] {
				//if n.operation != nil {
				children := getOrderedChildren(g.g, n)
				for j := 0; j < len(children); j++ {
					if children[j].gorgoniaNode == nil {
						nilChild = true
						break
					}
				}
				//}
			}
			if nilChild {
				continue
			}
			err := g.applyOperation(nodes[i]...)
			if err != nil {
				// The operator depends on something the explicit edges do not
				// express — a value captured by a subgraph, which carries no
				// edge to its producer — and that something is not built yet.
				// Leave the group in place and retry it on the next pass.
				if errors.Is(err, errNotReady) {
					deferred = append(deferred, err)
					continue
				}
				return err
			}
			nodes = append(nodes[:i], nodes[i+1:]...)
		}
		if len(nodes) == initialLen {
			if len(deferred) > 0 {
				return errors.Join(deferred...)
			}
			return errors.New("infinite loop")
		}
	}
	return nil
}

// applyOperation creates a new node on the exprgraph
func (g *Graph) applyOperation(n ...*Node) error {
	// Is this node already in the ExprGraph?
	if n[0].gorgoniaNode != nil {
		return fmt.Errorf("unsupported case: node is already in the exprgraph")
	}
	var op operator
	var opC func() operator
	var ok bool
	if opC, ok = operators[n[0].operation.Name]; !ok {
		return &onnx.ErrNotImplemented{
			Operator: n[0].operation.Name,
		}
	}
	op = opC()
	err := op.init(*n[0].operation)
	if err != nil {
		return err
	}
	// Wrap apply in recover to show operator name and stack trace on panic
	defer func() {
		if r := recover(); r != nil {
			panic(fmt.Sprintf("panic in operator %s: %v\n%s", n[0].operation.Name, r, debug.Stack()))
		}
	}()
	return op.apply(g, n...)
}
