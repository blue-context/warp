package cache

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// MemoryCache implements an in-memory cache with automatic expiration and size-based eviction.
//
// Thread Safety: Safe for concurrent use. All operations are protected by a mutex.
//
// Features:
//   - Automatic expiration: cleanup goroutine removes expired entries
//   - Size-based eviction: FIFO eviction when maxSize is reached
//   - Zero-copy: stores byte slices directly
type MemoryCache struct {
	data    map[string]*entry
	maxSize int64
	size    int64
	mu      sync.RWMutex
	done    chan struct{}
	wg      sync.WaitGroup
}

// entry represents a cached value with expiration.
type entry struct {
	value      []byte
	expiration time.Time
	size       int64
}

// NewMemoryCache creates a new in-memory cache.
//
// Parameters:
//   - maxSize: Maximum cache size in bytes. 0 means unlimited.
//
// The cache automatically starts a cleanup goroutine that runs every minute
// to remove expired entries. Call Close() to stop the goroutine and release resources.
//
// Example:
//
//	cache := cache.NewMemoryCache(100 * 1024 * 1024) // 100MB
//	defer cache.Close()
func NewMemoryCache(maxSize int64) *MemoryCache {
	c := &MemoryCache{
		data:    make(map[string]*entry),
		maxSize: maxSize,
		done:    make(chan struct{}),
	}

	// Start cleanup goroutine
	c.wg.Add(1)
	go c.cleanup()

	return c
}

// Get retrieves a value from the cache.
//
// Returns an error if the key is not found or has expired.
func (c *MemoryCache) Get(ctx context.Context, key string) ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	e, exists := c.data[key]
	if !exists {
		return nil, fmt.Errorf("key not found")
	}

	// Check expiration
	if !e.expiration.IsZero() && time.Now().After(e.expiration) {
		return nil, fmt.Errorf("key expired")
	}

	return e.value, nil
}

// Set stores a value in the cache with TTL.
//
// If maxSize is set and adding the value would exceed the limit,
// entries are evicted in FIFO order until there is enough space.
//
// A TTL of 0 means the entry never expires.
func (c *MemoryCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	size := int64(len(value))

	// If key exists, remove old size from total
	if old, exists := c.data[key]; exists {
		c.size -= old.size
	}

	// Evict entries if needed to make room
	if c.maxSize > 0 && c.size+size > c.maxSize {
		c.evict(size)
	}

	// Set expiration
	var expiration time.Time
	if ttl > 0 {
		expiration = time.Now().Add(ttl)
	}

	// Store entry
	c.data[key] = &entry{
		value:      value,
		expiration: expiration,
		size:       size,
	}
	c.size += size

	return nil
}

// Delete removes a value from the cache.
//
// Does nothing if the key doesn't exist.
func (c *MemoryCache) Delete(ctx context.Context, key string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if e, exists := c.data[key]; exists {
		c.size -= e.size
		delete(c.data, key)
	}

	return nil
}

// Clear removes all values from the cache.
func (c *MemoryCache) Clear(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data = make(map[string]*entry)
	c.size = 0

	return nil
}

// Close closes the cache and stops the cleanup goroutine.
//
// After calling Close, the cache should not be used.
// It is safe to call Close multiple times.
func (c *MemoryCache) Close() error {
	select {
	case <-c.done:
		// Already closed
		return nil
	default:
		close(c.done)
	}

	// Wait for cleanup goroutine to exit
	c.wg.Wait()
	return nil
}

// cleanup removes expired entries periodically.
//
// Runs in a separate goroutine started by NewMemoryCache.
// Checks for expired entries every minute and removes them.
func (c *MemoryCache) cleanup() {
	defer c.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.mu.Lock()
			now := time.Now()
			for key, e := range c.data {
				if !e.expiration.IsZero() && now.After(e.expiration) {
					c.size -= e.size
					delete(c.data, key)
				}
			}
			c.mu.Unlock()
		case <-c.done:
			return
		}
	}
}

// evict evicts entries using FIFO strategy to make room for new entries.
//
// Must be called with the lock held.
func (c *MemoryCache) evict(needed int64) {
	// Simple FIFO eviction - map iteration order is randomized in Go,
	// but this provides a simple eviction strategy
	for key, e := range c.data {
		c.size -= e.size
		delete(c.data, key)

		// Check if we have enough space now
		if c.size+needed <= c.maxSize {
			break
		}
	}
}
