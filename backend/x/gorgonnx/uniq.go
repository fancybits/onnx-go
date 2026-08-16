package gorgonnx

import (
	"strconv"
	"sync/atomic"
)

// Atomic: a torn increment would give two nodes the same name, which gorgonia
// hashes and compares on.
var uniq atomic.Uint64

func getUniqNodeName(prefix string) string {
	return prefix + strconv.FormatUint(uniq.Add(1), 10)
}
