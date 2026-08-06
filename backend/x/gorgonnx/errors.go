package gorgonnx

import (
	"errors"
	"fmt"
)

// errNotReady is returned by an operator that cannot be applied yet because a
// value it needs has not been built. The graph walk defers such an operator to
// a later pass instead of failing (see populateExprgraph). Wrap it, so that the
// error reported when no pass can make progress still says what was missing.
var errNotReady = errors.New("not ready")

type errOp struct {
	op  string
	err error
}

func (e *errOp) Error() string {
	return fmt.Sprintf("%s: %v", e.op, e.err)
}
