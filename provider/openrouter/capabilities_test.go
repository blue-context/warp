package openrouter

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestOpenRouterCapabilitiesAccuracy verifies that Supports() accurately reflects actual implementation.
func TestOpenRouterCapabilitiesAccuracy(t *testing.T) {
	opts := getTestOptions()
	p, err := NewProvider(opts...)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	provider.AssertCapabilitiesAccuracy(t, p)
}
