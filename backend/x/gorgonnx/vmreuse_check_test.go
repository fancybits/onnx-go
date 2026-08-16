package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/tensor"
)

// chainGraph builds a straight-line graph of the named unary operations and
// returns it with its input and final output node.
//
// The choice of chain matters for the reuse test. A partially rewound VM
// re-executes only the tail of the program, so a chain of one operation --
// or of repeated Neg -- still lands on the right answer and hides the bug.
// Sqrt/Neg/Abs does not: it leaves a stale intermediate that reaches the
// output.
func chainGraph(t *testing.T, ops ...string) (*Graph, *Node, *Node) {
	t.Helper()
	g := NewGraph()
	in := g.NewNode()
	g.AddNode(in)

	prev := in
	for _, op := range ops {
		out := g.NewNode()
		g.AddNode(out)
		g.SetWeightedEdge(g.NewWeightedEdge(out, prev, 0))
		if err := g.ApplyOperation(onnx.Operation{Name: op}, out); err != nil {
			t.Fatal(err)
		}
		prev = out
	}

	return g, in.(*Node), prev.(*Node)
}

// TestVMReuseAcrossRuns guards the VM caching added alongside graph reuse: a
// second run on the same graph must recompute from the newly bound inputs,
// not replay the previous run's values. Both VM kinds are checked -- the tape
// machine is the one that gets reused, and lisp is the one that must not be,
// because lispMachine.Reset rewinds the backward pass rather than the forward
// one and a reused LispMachine silently returns stale output.
func TestVMReuseAcrossRuns(t *testing.T) {
	for _, vm := range []string{"tape", "lisp"} {
		t.Run(vm, func(t *testing.T) {
			// abs(-sqrt(x)) == sqrt(x), so squares in give their roots out.
			g, in, out := chainGraph(t, "Sqrt", "Neg", "Abs")

			run := func(values ...float32) []float32 {
				tt := tensor.New(tensor.WithShape(len(values)), tensor.WithBacking(values))
				if err := in.SetTensor(tt); err != nil {
					t.Fatalf("SetTensor: %v", err)
				}
				if err := g.RunWithVM(vm); err != nil {
					t.Fatalf("run: %v", err)
				}
				return out.GetTensor().Data().([]float32)
			}

			if got := run(1, 4, 9); got[0] != 1 || got[2] != 3 {
				t.Fatalf("run 1: got %v, want [1 2 3]", got)
			}
			if got := run(16, 25, 36); got[0] != 4 || got[2] != 6 {
				t.Errorf("run 2: got %v, want [4 5 6] (stale output from run 1?)", got)
			}
		})
	}
}

// TestSetTensorRejectsWrongShape checks that tolerating a new batch size did
// not turn every shape mismatch into a silent accept: only the leading
// dimension may move, and anything else is still reported to the caller.
func TestSetTensorRejectsWrongShape(t *testing.T) {
	g, in, _ := chainGraph(t, "Neg")

	build := tensor.New(tensor.WithShape(1, 2, 3), tensor.WithBacking(make([]float32, 6)))
	if err := in.SetTensor(build); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}

	// Rebind the built shape before each case, so every case starts from a
	// graph that is actually built for (1,2,3).
	rebuild := func() {
		if err := in.SetTensor(build); err != nil {
			t.Fatal(err)
		}
		if err := g.Run(); err != nil {
			t.Fatal(err)
		}
	}

	for _, tc := range []struct {
		name    string
		shape   []int
		wantErr bool
	}{
		{"rebatch", []int{2, 2, 3}, false},
		{"transposed", []int{1, 3, 2}, true},
		{"rank change", []int{6}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rebuild()
			size := 1
			for _, d := range tc.shape {
				size *= d
			}
			candidate := tensor.New(tensor.WithShape(tc.shape...), tensor.WithBacking(make([]float32, size)))
			err := in.SetTensor(candidate)
			if tc.wantErr && err == nil {
				t.Errorf("(1,2,3)->%v should be rejected, got nil", tc.shape)
			}
			if !tc.wantErr && err != nil {
				t.Errorf("(1,2,3)->%v should be accepted, got: %v", tc.shape, err)
			}
		})
	}
}

// TestRunWithBadVMKeepsCachedVM checks that an unknown VM name fails without
// discarding the compiled program the caller already had.
func TestRunWithBadVMKeepsCachedVM(t *testing.T) {
	g, in, _ := chainGraph(t, "Neg")

	if err := in.SetTensor(tensor.New(tensor.WithShape(3), tensor.WithBacking([]float32{1, 2, 3}))); err != nil {
		t.Fatal(err)
	}
	if err := g.Run(); err != nil {
		t.Fatal(err)
	}
	if g.m == nil {
		t.Fatal("expected a cached VM after Run")
	}

	if err := g.RunWithVM("tpae"); err == nil {
		t.Error("expected an error for an unknown VM type")
	}
	if g.m == nil {
		t.Error("a rejected VM name discarded the cached VM")
	}
}
