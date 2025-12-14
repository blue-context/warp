package vertex

import (
	"testing"

	"github.com/blue-context/warp/provider"
)

// TestProviderCompliance verifies that this provider implements the Provider interface correctly.
func TestProviderCompliance(t *testing.T) {
	// Create provider with test configuration
	opts := getTestOptions(t)
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
func getTestOptions(t *testing.T) []Option {
	t.Helper()

	// Generate a valid test service account key with proper PKCS#8 RSA key
	testKey := generateTestServiceAccountKey(t)

	return []Option{
		WithProjectID("test-project"),
		WithServiceAccountKey(testKey),
	}
}
