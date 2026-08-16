package testbackend

import (
	"math"
	"testing"
)

// recorder captures whether an assertion failed, so the tolerance itself can
// be tested without failing the test that exercises it.
type recorder struct{ failed bool }

func (r *recorder) Errorf(string, ...interface{}) { r.failed = true }

func TestAssertCloseTolerance(t *testing.T) {
	// One float32 ULP at the given magnitude, which is the difference a
	// conforming implementation can legitimately produce by accumulating in a
	// different order.
	ulp := func(x float32) float64 {
		return float64(math.Nextafter32(x, math.MaxFloat32) - x)
	}

	for _, tc := range []struct {
		name       string
		expected   []float32
		actual     []float32
		wantFailed bool
	}{
		{
			name:     "identical",
			expected: []float32{1, 2, 3},
			actual:   []float32{1, 2, 3},
		},
		{
			// The case that motivated the change: 1 ULP apart at magnitude
			// ~224, which the old absolute 1e-6 rejected.
			name:     "one ulp at large magnitude",
			expected: []float32{224.10666},
			actual:   []float32{224.10664},
		},
		{
			name:     "one ulp at unit magnitude",
			expected: []float32{1},
			actual:   []float32{1 + float32(ulp(1))},
		},
		{
			// A real error must still fail. At magnitude 224 the tolerance is
			// ~2.2e-4, so a 0.01 discrepancy is ~45x too large.
			name:       "small absolute error at large magnitude still fails",
			expected:   []float32{224.10666},
			actual:     []float32{224.11666},
			wantFailed: true,
		},
		{
			// Relative slack must not rescue a wrong small value.
			name:       "error at unit magnitude still fails",
			expected:   []float32{1},
			actual:     []float32{1.001},
			wantFailed: true,
		},
		{
			name:       "sign error fails",
			expected:   []float32{5},
			actual:     []float32{-5},
			wantFailed: true,
		},
		{
			name:       "length mismatch fails",
			expected:   []float32{1, 2},
			actual:     []float32{1},
			wantFailed: true,
		},
		{
			name:     "NaN matches NaN",
			expected: []float32{float32(math.NaN())},
			actual:   []float32{float32(math.NaN())},
		},
		{
			name:       "NaN does not match a number",
			expected:   []float32{float32(math.NaN())},
			actual:     []float32{1},
			wantFailed: true,
		},
		{
			name:       "a number does not match NaN",
			expected:   []float32{1},
			actual:     []float32{float32(math.NaN())},
			wantFailed: true,
		},
		{
			name:     "matching infinities",
			expected: []float32{float32(math.Inf(1)), float32(math.Inf(-1))},
			actual:   []float32{float32(math.Inf(1)), float32(math.Inf(-1))},
		},
		{
			name:       "opposite infinities fail",
			expected:   []float32{float32(math.Inf(1))},
			actual:     []float32{float32(math.Inf(-1))},
			wantFailed: true,
		},
		{
			name:       "infinity does not match a number",
			expected:   []float32{float32(math.Inf(1))},
			actual:     []float32{1e30},
			wantFailed: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &recorder{}
			assertClose(r, tc.expected, tc.actual)
			if r.failed != tc.wantFailed {
				t.Errorf("failed=%v, want %v (expected=%v actual=%v)",
					r.failed, tc.wantFailed, tc.expected, tc.actual)
			}
		})
	}
}

// TestAssertCloseScalars: a fully reduced tensor's Data() is a bare scalar,
// not a slice. That path must use the same tolerance -- it is the shape a
// reduce-everything fixture produces, which is where accumulation-order
// differences turn up.
func TestAssertCloseScalars(t *testing.T) {
	for _, tc := range []struct {
		name             string
		expected, actual interface{}
		wantFailed       bool
	}{
		{"identical", float32(1.5), float32(1.5), false},
		{"one ulp at large magnitude", float32(224.10666), float32(224.10664), false},
		{"real error still fails", float32(224.10666), float32(224.11666), true},
		{"unit magnitude error fails", float32(1), float32(1.001), true},
		{"NaN matches NaN", float32(math.NaN()), float32(math.NaN()), false},
		{"NaN does not match a number", float32(math.NaN()), float32(1), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &recorder{}
			assertClose(r, tc.expected, tc.actual)
			if r.failed != tc.wantFailed {
				t.Errorf("failed=%v, want %v (%v vs %v)", r.failed, tc.wantFailed, tc.expected, tc.actual)
			}
		})
	}
}

func TestAssertCloseIntegers(t *testing.T) {
	r := &recorder{}
	assertClose(r, []int32{1, 2, 3}, []int32{1, 2, 3})
	if r.failed {
		t.Error("identical integer slices should compare equal")
	}

	r = &recorder{}
	assertClose(r, []int32{1, 2, 3}, []int32{1, 2, 4})
	if !r.failed {
		t.Error("differing integer slices should fail")
	}

	// Large integers must not inherit the relative tolerance: at 2e6 it would
	// otherwise exceed 1 and swallow an off-by-one index.
	for _, tc := range []struct {
		name             string
		expected, actual interface{}
	}{
		{"int64 off by one at 2e6", []int64{2000000, 5}, []int64{2000001, 5}},
		{"int32 off by one at 1e6", []int32{1000000}, []int32{1000001}},
		{"uint64 off by one at 4e6", []uint64{4000000}, []uint64{4000001}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &recorder{}
			assertClose(r, tc.expected, tc.actual)
			if !r.failed {
				t.Errorf("%v vs %v should fail", tc.expected, tc.actual)
			}
		})
	}
}
