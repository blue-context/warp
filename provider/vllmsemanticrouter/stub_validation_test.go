package vllmsemanticrouter

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestStubMethodsReturnWarpError verifies that unsupported methods return proper WarpError.
func TestStubMethodsReturnWarpError(t *testing.T) {
	// Create provider with test configuration
	opts := getTestOptions()
	p, err := NewProvider(opts...)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Run stub validation checks
	provider.AssertStubMethodsReturnWarpError(t, p)
}
