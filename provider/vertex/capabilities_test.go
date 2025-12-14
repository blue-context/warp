package vertex

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestVertexCapabilitiesAccuracy verifies that Supports() accurately reflects actual implementation.
func TestVertexCapabilitiesAccuracy(t *testing.T) {
	opts := getTestOptions(t)
	p, err := NewProvider(opts...)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	provider.AssertCapabilitiesAccuracy(t, p)
}
