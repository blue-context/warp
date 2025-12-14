package cache

import (
	"context"
	"crypto/sha256"
	"fmt"
	"time"
)

// Cache defines the caching interface.
//
// Implementations must be thread-safe for concurrent use.
//
// Users can implement this interface with their own caching backend
// (Redis, Memcached, DynamoDB, etc.).
type Cache interface {
	// Get retrieves a value from the cache.
	//
	// Returns an error if the key is not found or expired.
	Get(ctx context.Context, key string) ([]byte, error)

	// Set stores a value in the cache with TTL.
	//
	// A TTL of 0 means no expiration.
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Delete removes a value from the cache.
	Delete(ctx context.Context, key string) error

	// Clear removes all values from the cache.
	Clear(ctx context.Context) error

	// Close closes the cache and releases resources.
	Close() error
}

// Key generates a deterministic cache key for a completion request.
//
// The key is generated from the model name and request parameters to ensure
// identical requests produce identical cache keys.
//
// Parameters:
//   - model: The model name (e.g., "gpt-4", "claude-3-sonnet")
//   - messages: JSON-encoded messages array
//   - temperature: Temperature parameter (can be nil)
//   - maxTokens: Max tokens parameter (can be nil)
//   - topP: Top-p parameter (can be nil)
//
// Returns a SHA256-based cache key prefixed with "warp:v1:".
func Key(model string, messages []byte, temperature, maxTokens, topP any) string {
	h := sha256.New()

	// Model
	h.Write([]byte(model))

	// Messages
	h.Write(messages)

	// Parameters - format them consistently
	if temperature != nil {
		switch v := temperature.(type) {
		case float64:
			h.Write([]byte(fmt.Sprintf("temp:%.2f", v)))
		case *float64:
			if v != nil {
				h.Write([]byte(fmt.Sprintf("temp:%.2f", *v)))
			}
		}
	}

	if maxTokens != nil {
		switch v := maxTokens.(type) {
		case int:
			h.Write([]byte(fmt.Sprintf("max:%d", v)))
		case *int:
			if v != nil {
				h.Write([]byte(fmt.Sprintf("max:%d", *v)))
			}
		}
	}

	if topP != nil {
		switch v := topP.(type) {
		case float64:
			h.Write([]byte(fmt.Sprintf("top_p:%.2f", v)))
		case *float64:
			if v != nil {
				h.Write([]byte(fmt.Sprintf("top_p:%.2f", *v)))
			}
		}
	}

	return fmt.Sprintf("warp:v1:%x", h.Sum(nil))
}
