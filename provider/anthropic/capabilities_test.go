package anthropic

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestAnthropicCapabilitiesAccuracy verifies that Supports() accurately reflects actual implementation.
func TestAnthropicCapabilitiesAccuracy(t *testing.T) {
	opts := getTestOptions()
	p, err := NewProvider(opts...)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	provider.AssertCapabilitiesAccuracy(t, p)
}
