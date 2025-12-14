package cache

import (
	"context"
	"testing"
	"time"
)

func TestNoopCache_Get(t *testing.T) {
	ctx := context.Background()
	cache := NewNoopCache()

	_, err := cache.Get(ctx, "any-key")
	if err == nil {
		t.Error("Get() expected error, got nil")
	}
}

func TestNoopCache_Set(t *testing.T) {
	ctx := context.Background()
	cache := NewNoopCache()

	err := cache.Set(ctx, "key", []byte("value"), time.Hour)
	if err != nil {
		t.Errorf("Set() error = %v, want nil", err)
	}

	// Get should still return error
	_, err = cache.Get(ctx, "key")
	if err == nil {
		t.Error("Get() expected error after Set()")
	}
}

func TestNoopCache_Delete(t *testing.T) {
	ctx := context.Background()
	cache := NewNoopCache()

	err := cache.Delete(ctx, "key")
	if err != nil {
		t.Errorf("Delete() error = %v, want nil", err)
	}
}

func TestNoopCache_Clear(t *testing.T) {
	ctx := context.Background()
	cache := NewNoopCache()

	err := cache.Clear(ctx)
	if err != nil {
		t.Errorf("Clear() error = %v, want nil", err)
	}
}

func TestNoopCache_Close(t *testing.T) {
	cache := NewNoopCache()

	err := cache.Close()
	if err != nil {
		t.Errorf("Close() error = %v, want nil", err)
	}

	// Close again should be safe
	err = cache.Close()
	if err != nil {
		t.Errorf("Close() second call error = %v, want nil", err)
	}
}

func TestNoopCache_AllOperations(t *testing.T) {
	ctx := context.Background()
	cache := NewNoopCache()
	defer cache.Close()

	// All operations should succeed except Get
	err := cache.Set(ctx, "key1", []byte("value1"), time.Hour)
	if err != nil {
		t.Errorf("Set() error = %v", err)
	}

	err = cache.Set(ctx, "key2", []byte("value2"), 0)
	if err != nil {
		t.Errorf("Set() error = %v", err)
	}

	_, err = cache.Get(ctx, "key1")
	if err == nil {
		t.Error("Get() expected error")
	}

	err = cache.Delete(ctx, "key1")
	if err != nil {
		t.Errorf("Delete() error = %v", err)
	}

	err = cache.Clear(ctx)
	if err != nil {
		t.Errorf("Clear() error = %v", err)
	}
}
