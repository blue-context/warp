package cache

import (
	"context"
	"fmt"
	"time"
)

// NoopCache is a cache implementation that does nothing.
//
// Useful for disabling caching without changing client code.
// All operations are no-ops except Get, which always returns "key not found".
//
// Thread Safety: Safe for concurrent use (operations do nothing).
type NoopCache struct{}

// NewNoopCache creates a new no-op cache.
//
// Example:
//
//	cache := cache.NewNoopCache()
//	client, err := warp.NewClient(warp.WithCache(cache))
func NewNoopCache() *NoopCache {
	return &NoopCache{}
}

// Get always returns "key not found" error.
func (c *NoopCache) Get(ctx context.Context, key string) ([]byte, error) {
	return nil, fmt.Errorf("key not found")
}

// Set does nothing and returns nil.
func (c *NoopCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	return nil
}

// Delete does nothing and returns nil.
func (c *NoopCache) Delete(ctx context.Context, key string) error {
	return nil
}

// Clear does nothing and returns nil.
func (c *NoopCache) Clear(ctx context.Context) error {
	return nil
}

// Close does nothing and returns nil.
func (c *NoopCache) Close() error {
	return nil
}
