package vllm

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestProviderCompliance verifies that this provider implements the Provider interface correctly.
func TestProviderCompliance(t *testing.T) {
	// Create provider with test configuration
	opts := getTestOptions()
	p, err := NewProvider(opts...)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Run compliance checks
	provider.AssertProviderCompliance(t, p)
	provider.AssertMethodCount(t, p)
}

// getTestOptions returns options for creating a test provider instance.
// These options use test values and don't make real API calls.
func getTestOptions() []Option {
	// Provider-specific test options
	return []Option{
		WithBaseURL("http://localhost:8000"),
	}
}
