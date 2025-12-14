//go:build example
// +build example

package cache

// This file is an EXAMPLE showing how to implement the Cache interface
// with a Redis client. It is excluded from normal builds using the "example" build tag.
//
// Users should copy this pattern and use their own Redis client library.
//
// To use Redis caching:
//  1. Import your preferred Redis client (e.g., github.com/redis/go-redis/v9)
//  2. Implement the Cache interface as shown below
//  3. Pass your Redis cache to warp.WithCache()
//
// Example usage:
//
//	import (
//	    "github.com/redis/go-redis/v9"
//	    "github.com/blue-context/warp"
//	    "github.com/blue-context/warp/cache"
//	)
//
//	// Create Redis client
//	redisClient := redis.NewClient(&redis.Options{
//	    Addr: "localhost:6379",
//	})
//
//	// Create cache wrapper
//	redisCache := NewRedisCache(redisClient)
//
//	// Create Warp client with Redis cache
//	client, err := warp.NewClient(
//	    warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
//	    warp.WithCache(redisCache),
//	)

/*
Example Redis cache implementation (requires github.com/redis/go-redis/v9):

import (
	"context"
	"time"
	"github.com/redis/go-redis/v9"
)

// RedisCache implements the Cache interface using Redis as the backend.
//
// Thread Safety: Safe for concurrent use (Redis client handles concurrency).
type RedisCache struct {
	client *redis.Client
}

// NewRedisCache creates a new Redis-backed cache.
//
// Parameters:
//   - client: A configured Redis client from github.com/redis/go-redis/v9
//
// Example:
//   redisClient := redis.NewClient(&redis.Options{
//       Addr:     "localhost:6379",
//       Password: "", // no password
//       DB:       0,  // default DB
//   })
//   cache := NewRedisCache(redisClient)
func NewRedisCache(client *redis.Client) *RedisCache {
	return &RedisCache{client: client}
}

// Get retrieves a value from Redis.
//
// Returns an error if the key doesn't exist or if Redis returns an error.
func (c *RedisCache) Get(ctx context.Context, key string) ([]byte, error) {
	result, err := c.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, fmt.Errorf("key not found")
	}
	return result, err
}

// Set stores a value in Redis with TTL.
//
// If ttl is 0, the key will not expire.
func (c *RedisCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	return c.client.Set(ctx, key, value, ttl).Err()
}

// Delete removes a key from Redis.
func (c *RedisCache) Delete(ctx context.Context, key string) error {
	return c.client.Del(ctx, key).Err()
}

// Clear removes all keys from the current Redis database.
//
// WARNING: This will flush the entire database, not just Warp cache keys.
// Consider using a dedicated Redis database for Warp cache or implementing
// key-based deletion with a common prefix.
func (c *RedisCache) Clear(ctx context.Context) error {
	return c.client.FlushDB(ctx).Err()
}

// Close closes the Redis connection.
func (c *RedisCache) Close() error {
	return c.client.Close()
}

Alternative implementation with key prefix filtering:

// RedisCacheWithPrefix implements Cache with a key prefix for isolation.
type RedisCacheWithPrefix struct {
	client *redis.Client
	prefix string
}

// NewRedisCacheWithPrefix creates a Redis cache with key prefix.
//
// This allows multiple applications or cache instances to share the same
// Redis database without key conflicts.
//
// Example:
//   cache := NewRedisCacheWithPrefix(redisClient, "warp:")
func NewRedisCacheWithPrefix(client *redis.Client, prefix string) *RedisCacheWithPrefix {
	return &RedisCacheWithPrefix{
		client: client,
		prefix: prefix,
	}
}

func (c *RedisCacheWithPrefix) Get(ctx context.Context, key string) ([]byte, error) {
	result, err := c.client.Get(ctx, c.prefix+key).Bytes()
	if err == redis.Nil {
		return nil, fmt.Errorf("key not found")
	}
	return result, err
}

func (c *RedisCacheWithPrefix) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	return c.client.Set(ctx, c.prefix+key, value, ttl).Err()
}

func (c *RedisCacheWithPrefix) Delete(ctx context.Context, key string) error {
	return c.client.Del(ctx, c.prefix+key).Err()
}

func (c *RedisCacheWithPrefix) Clear(ctx context.Context) error {
	// Use SCAN to find all keys with the prefix and delete them
	iter := c.client.Scan(ctx, 0, c.prefix+"*", 0).Iterator()
	pipe := c.client.Pipeline()

	for iter.Next(ctx) {
		pipe.Del(ctx, iter.Val())
	}

	if err := iter.Err(); err != nil {
		return err
	}

	_, err := pipe.Exec(ctx)
	return err
}

func (c *RedisCacheWithPrefix) Close() error {
	return c.client.Close()
}

*/
