package onnx

import (
	"sort"

	"github.com/owulveryck/onnx-go/internal/onnx/ir"
	"gonum.org/v1/gonum/graph"
)

// Subgraph wraps a graph carried by a GRAPH attribute (e.g. a Loop body).
// It is delivered to backends via Operation.Attributes; backends that
// support control-flow operators decode it with Decode.
type Subgraph struct {
	g *ir.GraphProto

	// referenced caches the set of names the subgraph uses, snapshotted
	// before the first Decode call ever mutates the proto. See
	// referencedNames for why the snapshot has to be cached.
	referenced map[string]bool
}

// SubgraphInfo describes the boundary of a decoded subgraph.
type SubgraphInfo struct {
	Inputs   []graph.Node
	Outputs  []graph.Node
	Captured map[string]graph.Node
}

// Decode materializes the subgraph into dst, mirroring how Model decodes
// the top-level graph. It reports the subgraph's boundary so the caller
// can stitch it into an enclosing scope.
func (s *Subgraph) Decode(dst Backend) (*SubgraphInfo, error) {
	db := make(map[string]graph.Node)
	info := &SubgraphInfo{Captured: make(map[string]graph.Node)}

	produced := s.producedNames()
	referenced := s.referencedNames()

	for _, io := range s.g.Input {
		n, err := processValueInto(dst, db, io)
		if err != nil {
			return nil, err
		}
		info.Inputs = append(info.Inputs, n)
	}
	for _, io := range s.g.ValueInfo {
		if _, err := processValueInto(dst, db, io); err != nil {
			return nil, err
		}
	}
	for _, io := range s.g.Output {
		// Body outputs may share names with body inputs (ONNX pass-through
		// carried values). Reuse the existing node instead of creating a
		// duplicate, mirroring how initializers are handled.
		if _, ok := db[io.Name]; ok {
			continue
		}
		if _, err := processValueInto(dst, db, io); err != nil {
			return nil, err
		}
	}
	if err := applyGraphTensors(dst, db, s.g, nil); err != nil {
		return nil, err
	}
	if err := applyGraphNodeOperations(dst, db, s.g); err != nil {
		return nil, err
	}

	for name := range referenced {
		if produced[name] {
			continue
		}
		if n, ok := db[name]; ok {
			info.Captured[name] = n
		}
	}
	for _, io := range s.g.Output {
		info.Outputs = append(info.Outputs, db[io.Name])
	}
	return info, nil
}

// CapturedNames returns the names the subgraph uses but does not produce —
// the values it captures from the enclosing scope. It is derived from the
// proto alone, so a backend can check that a capture is resolvable before
// committing to decoding the subgraph.
func (s *Subgraph) CapturedNames() []string {
	produced := s.producedNames()
	var names []string
	for name := range s.referencedNames() {
		if !produced[name] {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

// producedNames are the names the subgraph defines itself: its inputs, its
// initializers and its nodes' outputs.
func (s *Subgraph) producedNames() map[string]bool {
	produced := make(map[string]bool)
	for _, io := range s.g.Input {
		produced[io.Name] = true
	}
	for _, init := range s.g.Initializer {
		produced[init.GetName()] = true
	}
	for _, n := range s.g.Node {
		for _, o := range n.Output {
			produced[o] = true
		}
	}
	return produced
}

// referencedNames are the names the subgraph uses: every node input plus every
// graph output, since a graph output naming a value the subgraph does not
// produce is a use of an enclosing-scope value just as a node input is.
//
// The set is snapshotted before applyGraphNodeOperations ever runs and cached
// on s for reuse across repeated Decode calls on this same Subgraph. A
// Subgraph is decoded once per loop iteration for control-flow bodies (see
// gorgonnx's Loop), reusing the same underlying *ir.GraphProto across calls;
// applyGraphNodeOperations mutates node.Input in place (decoder.go synthesizes
// a fake "<node.Name>/input" entry for input-less nodes such as Constant), so
// recomputing this set on every call would pick up the previous call's
// synthesized names as if the proto had referenced them.
func (s *Subgraph) referencedNames() map[string]bool {
	if s.referenced != nil {
		return s.referenced
	}
	s.referenced = make(map[string]bool)
	for _, n := range s.g.Node {
		for _, input := range n.Input {
			// An omitted optional input is spelled as the empty name; it
			// refers to nothing, least of all to the enclosing scope.
			if input == "" {
				continue
			}
			s.referenced[input] = true
		}
	}
	for _, io := range s.g.Output {
		if io.Name == "" {
			continue
		}
		s.referenced[io.Name] = true
	}
	return s.referenced
}
