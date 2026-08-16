package testbackend

import (
	"math"
	"reflect"

	"github.com/stretchr/testify/assert"
)

// Tolerance for comparing a computed tensor against a test case's expected
// output, as absolute + relative*|expected| -- the same form numpy's
// assert_allclose uses, and that ONNX's own conformance runner applies to
// these very fixtures.
//
// A purely absolute tolerance cannot work here. The expected outputs were
// produced by accumulating in float32, and a conforming implementation is free
// to accumulate in a different order and land on a neighbouring float. At a
// magnitude of a few hundred, one float32 ULP is already ~1.5e-5, so demanding
// agreement to 1e-6 is demanding bit-identical arithmetic, which the ONNX spec
// does not ask for.
//
// The relative term is nonetheless far tighter than ONNX's own: it uses
// rtol=1e-3, which at that magnitude allows ~0.2. This allows roughly eight
// ULPs. The aim is to absorb rounding-order differences and nothing else, so
// that a real regression still fails.
const (
	absTolerance = 1e-6
	relTolerance = 1e-6
)

// assertClose compares expected and actual elementwise. Both are the Data() of
// a tensor, so they arrive as an untyped slice of some numeric kind.
func assertClose(t assert.TestingT, expected, actual interface{}) bool {
	ev := reflect.ValueOf(expected)
	av := reflect.ValueOf(actual)

	// A fully reduced tensor has a scalar Data(), not a slice, and that is
	// precisely where accumulation-order differences show up -- so it must go
	// through the tolerance rather than fall back to an exact compare.
	if ev.Kind() != reflect.Slice && av.Kind() != reflect.Slice {
		if want, ok := asFloat(ev); ok {
			if got, ok := asFloat(av); ok {
				return closeEnough(t, 0, want, got)
			}
		}

		return assert.Equal(t, expected, actual, "the two tensors should be equal.")
	}
	if ev.Kind() != reflect.Slice || av.Kind() != reflect.Slice {
		return assert.Equal(t, expected, actual, "the two tensors should be equal.")
	}
	if ev.Len() != av.Len() {
		return assert.Fail(t, "the two tensors should have the same number of elements.",
			"expected %d, got %d", ev.Len(), av.Len())
	}

	// Integers are exact by construction -- shape outputs, argmax indices,
	// gather indices. Giving them a relative tolerance would accept an
	// off-by-one on any value above 1e6, which is the opposite of what these
	// comparisons are for. Only floating point earns the slack.
	if isIntegerKind(ev.Type().Elem().Kind()) || isIntegerKind(av.Type().Elem().Kind()) {
		return assert.Equal(t, expected, actual, "the two tensors should be equal.")
	}

	for i := 0; i < ev.Len(); i++ {
		want, ok := asFloat(ev.Index(i))
		if !ok {
			// Not a numeric kind; fall back to exact comparison of the whole
			// slice rather than guessing elementwise.
			return assert.Equal(t, expected, actual, "the two tensors should be equal.")
		}
		got, ok := asFloat(av.Index(i))
		if !ok {
			return assert.Equal(t, expected, actual, "the two tensors should be equal.")
		}

		if !closeEnough(t, i, want, got) {
			return false
		}
	}

	return true
}

// closeEnough compares one pair of values, reporting the element index.
func closeEnough(t assert.TestingT, i int, want, got float64) bool {
	// NaN and infinities have to be settled before subtracting. NaN minus
	// anything is NaN, and infinity minus itself is NaN, and NaN compares
	// false against any threshold -- so a difference test alone would wave
	// through a NaN result, which is the opposite of what it should do.
	if math.IsNaN(want) || math.IsNaN(got) || math.IsInf(want, 0) || math.IsInf(got, 0) {
		if math.IsNaN(want) && math.IsNaN(got) {
			return true
		}
		if math.IsInf(want, 1) && math.IsInf(got, 1) {
			return true
		}
		if math.IsInf(want, -1) && math.IsInf(got, -1) {
			return true
		}

		return assert.Fail(t, "the two tensors should be equal.",
			"element %d: expected %v, got %v", i, want, got)
	}

	allowed := absTolerance + relTolerance*math.Abs(want)
	if diff := math.Abs(want - got); diff > allowed {
		return assert.Fail(t, "the two tensors should be equal.",
			"element %d: expected %v, got %v (difference %v exceeds tolerance %v)",
			i, want, got, diff, allowed)
	}

	return true
}

// asFloat widens a floating point slice element for comparison.
func asFloat(v reflect.Value) (float64, bool) {
	switch v.Kind() {
	case reflect.Float32, reflect.Float64:
		return v.Float(), true
	}

	return 0, false
}

func isIntegerKind(k reflect.Kind) bool {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return true
	}

	return false
}
