package cohere

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestCohereCapabilitiesAccuracy verifies that Supports() accurately reflects actual implementation.
func TestCohereCapabilitiesAccuracy(t *testing.T) {
	opts := getTestOptions()
	p, err := NewProvider(opts...)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	provider.AssertCapabilitiesAccuracy(t, p)
}
