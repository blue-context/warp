package cache

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestMemoryCache_GetSet(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0) // Unlimited size
	defer cache.Close()

	// Test Set and Get
	key := "test-key"
	value := []byte("test-value")

	err := cache.Set(ctx, key, value, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	got, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	if string(got) != string(value) {
		t.Errorf("Get() = %s, want %s", got, value)
	}
}

func TestMemoryCache_GetNotFound(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	_, err := cache.Get(ctx, "non-existent")
	if err == nil {
		t.Error("Get() expected error for non-existent key")
	}
}

func TestMemoryCache_Expiration(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	key := "test-key"
	value := []byte("test-value")

	// Set with 100ms TTL
	err := cache.Set(ctx, key, value, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	// Should be accessible immediately
	_, err = cache.Get(ctx, key)
	if err != nil {
		t.Errorf("Get() error = %v, expected no error", err)
	}

	// Wait for expiration
	time.Sleep(150 * time.Millisecond)

	// Should be expired
	_, err = cache.Get(ctx, key)
	if err == nil {
		t.Error("Get() expected error for expired key")
	}
}

func TestMemoryCache_Delete(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	key := "test-key"
	value := []byte("test-value")

	err := cache.Set(ctx, key, value, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	err = cache.Delete(ctx, key)
	if err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	_, err = cache.Get(ctx, key)
	if err == nil {
		t.Error("Get() expected error after Delete()")
	}
}

func TestMemoryCache_Clear(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	// Set multiple keys
	for i := 0; i < 5; i++ {
		key := string(rune('a' + i))
		value := []byte("value-" + key)
		err := cache.Set(ctx, key, value, 0)
		if err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}

	err := cache.Clear(ctx)
	if err != nil {
		t.Fatalf("Clear() error = %v", err)
	}

	// All keys should be gone
	for i := 0; i < 5; i++ {
		key := string(rune('a' + i))
		_, err := cache.Get(ctx, key)
		if err == nil {
			t.Errorf("Get(%s) expected error after Clear()", key)
		}
	}
}

func TestMemoryCache_SizeLimit(t *testing.T) {
	ctx := context.Background()
	maxSize := int64(100) // 100 bytes
	cache := NewMemoryCache(maxSize)
	defer cache.Close()

	// Add entries that exceed the limit
	for i := 0; i < 10; i++ {
		key := string(rune('a' + i))
		value := make([]byte, 20) // 20 bytes each
		err := cache.Set(ctx, key, value, 0)
		if err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}

	// Check that total size doesn't exceed limit
	cache.mu.RLock()
	size := cache.size
	cache.mu.RUnlock()

	if size > maxSize {
		t.Errorf("cache size = %d, want <= %d", size, maxSize)
	}
}

func TestMemoryCache_Overwrite(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	key := "test-key"
	value1 := []byte("value1")
	value2 := []byte("value2-longer")

	err := cache.Set(ctx, key, value1, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	err = cache.Set(ctx, key, value2, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	got, err := cache.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	if string(got) != string(value2) {
		t.Errorf("Get() = %s, want %s", got, value2)
	}
}

func TestMemoryCache_CleanupGoroutine(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)

	// Set keys with short TTL
	for i := 0; i < 5; i++ {
		key := string(rune('a' + i))
		value := []byte("value-" + key)
		err := cache.Set(ctx, key, value, 50*time.Millisecond)
		if err != nil {
			t.Fatalf("Set() error = %v", err)
		}
	}

	// Wait for expiration
	time.Sleep(100 * time.Millisecond)

	// Close should wait for cleanup goroutine
	err := cache.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// Calling Close again should be safe
	err = cache.Close()
	if err != nil {
		t.Fatalf("Close() second call error = %v", err)
	}
}

func TestMemoryCache_Concurrent(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	var wg sync.WaitGroup
	numGoroutines := 10
	numOperations := 100

	// Concurrent writes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := string(rune('a' + (id % 26)))
				value := []byte("value")
				_ = cache.Set(ctx, key, value, 0)
			}
		}(i)
	}

	// Concurrent reads
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := string(rune('a' + (id % 26)))
				_, _ = cache.Get(ctx, key)
			}
		}(i)
	}

	// Concurrent deletes
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numOperations; j++ {
				key := string(rune('a' + (id % 26)))
				_ = cache.Delete(ctx, key)
			}
		}(i)
	}

	wg.Wait()
}

func TestMemoryCache_ZeroTTL(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	key := "test-key"
	value := []byte("test-value")

	// Set with 0 TTL (no expiration)
	err := cache.Set(ctx, key, value, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	// Wait a bit
	time.Sleep(100 * time.Millisecond)

	// Should still be accessible
	got, err := cache.Get(ctx, key)
	if err != nil {
		t.Errorf("Get() error = %v, expected no error for zero TTL", err)
	}

	if string(got) != string(value) {
		t.Errorf("Get() = %s, want %s", got, value)
	}
}

func TestMemoryCache_SizeTracking(t *testing.T) {
	ctx := context.Background()
	cache := NewMemoryCache(0)
	defer cache.Close()

	// Add entries and check size
	key1 := "key1"
	value1 := []byte("value1") // 6 bytes

	err := cache.Set(ctx, key1, value1, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	cache.mu.RLock()
	size1 := cache.size
	cache.mu.RUnlock()

	if size1 != 6 {
		t.Errorf("cache size = %d, want 6", size1)
	}

	// Add another entry
	key2 := "key2"
	value2 := []byte("value2value2") // 12 bytes

	err = cache.Set(ctx, key2, value2, 0)
	if err != nil {
		t.Fatalf("Set() error = %v", err)
	}

	cache.mu.RLock()
	size2 := cache.size
	cache.mu.RUnlock()

	if size2 != 18 {
		t.Errorf("cache size = %d, want 18", size2)
	}

	// Delete first entry
	err = cache.Delete(ctx, key1)
	if err != nil {
		t.Fatalf("Delete() error = %v", err)
	}

	cache.mu.RLock()
	size3 := cache.size
	cache.mu.RUnlock()

	if size3 != 12 {
		t.Errorf("cache size = %d, want 12", size3)
	}
}
