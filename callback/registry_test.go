package callback

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewRegistry(t *testing.T) {
	registry := NewRegistry()

	if registry == nil {
		t.Fatal("NewRegistry() returned nil")
	}

	if registry.beforeRequest == nil {
		t.Error("beforeRequest slice should be initialized")
	}
	if registry.success == nil {
		t.Error("success slice should be initialized")
	}
	if registry.failure == nil {
		t.Error("failure slice should be initialized")
	}
	if registry.stream == nil {
		t.Error("stream slice should be initialized")
	}
}

func TestRegistry_RegisterBeforeRequest(t *testing.T) {
	tests := []struct {
		name        string
		callback    BeforeRequestCallback
		expectAdded bool
	}{
		{
			name: "register valid callback",
			callback: func(ctx context.Context, event *BeforeRequestEvent) error {
				return nil
			},
			expectAdded: true,
		},
		{
			name:        "register nil callback",
			callback:    nil,
			expectAdded: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewRegistry()
			initialCount := len(registry.beforeRequest)

			registry.RegisterBeforeRequest(tt.callback)

			finalCount := len(registry.beforeRequest)
			added := finalCount > initialCount

			if added != tt.expectAdded {
				t.Errorf("RegisterBeforeRequest() added = %v, expectAdded %v", added, tt.expectAdded)
			}
		})
	}
}

func TestRegistry_ExecuteBeforeRequest(t *testing.T) {
	tests := []struct {
		name        string
		callbacks   []BeforeRequestCallback
		expectError bool
		expectCount int
	}{
		{
			name:        "no callbacks",
			callbacks:   []BeforeRequestCallback{},
			expectError: false,
			expectCount: 0,
		},
		{
			name: "single successful callback",
			callbacks: []BeforeRequestCallback{
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return nil
				},
			},
			expectError: false,
			expectCount: 1,
		},
		{
			name: "multiple successful callbacks",
			callbacks: []BeforeRequestCallback{
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return nil
				},
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return nil
				},
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return nil
				},
			},
			expectError: false,
			expectCount: 3,
		},
		{
			name: "callback returns error",
			callbacks: []BeforeRequestCallback{
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return errors.New("callback error")
				},
			},
			expectError: true,
			expectCount: 1,
		},
		{
			name: "second callback fails, all callbacks still execute",
			callbacks: []BeforeRequestCallback{
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return nil
				},
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return errors.New("second callback error")
				},
				func(ctx context.Context, event *BeforeRequestEvent) error {
					return nil
				},
			},
			expectError: true,
			expectCount: 3, // All callbacks execute with error aggregation
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewRegistry()
			executionCount := 0

			// Wrap callbacks to count executions
			for _, cb := range tt.callbacks {
				originalCB := cb
				registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
					executionCount++
					return originalCB(ctx, event)
				})
			}

			event := &BeforeRequestEvent{
				RequestID: "test-req",
				Model:     "gpt-4",
				Provider:  "openai",
				StartTime: time.Now(),
			}

			err := registry.ExecuteBeforeRequest(context.Background(), event)

			if (err != nil) != tt.expectError {
				t.Errorf("ExecuteBeforeRequest() error = %v, expectError %v", err, tt.expectError)
			}

			if executionCount > tt.expectCount {
				t.Errorf("ExecuteBeforeRequest() executed %d callbacks, expected max %d", executionCount, tt.expectCount)
			}
		})
	}
}

func TestRegistry_ExecuteSuccess(t *testing.T) {
	tests := []struct {
		name          string
		callbackCount int
	}{
		{
			name:          "no callbacks",
			callbackCount: 0,
		},
		{
			name:          "single callback",
			callbackCount: 1,
		},
		{
			name:          "multiple callbacks",
			callbackCount: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewRegistry()
			executionCount := int32(0)

			// Register callbacks
			for i := 0; i < tt.callbackCount; i++ {
				registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {
					atomic.AddInt32(&executionCount, 1)
				})
			}

			event := &SuccessEvent{
				RequestID: "test-req",
				Model:     "gpt-4",
				Provider:  "openai",
				Cost:      0.002,
				Tokens:    150,
				Duration:  2 * time.Second,
			}

			registry.ExecuteSuccess(context.Background(), event)

			if int(executionCount) != tt.callbackCount {
				t.Errorf("ExecuteSuccess() executed %d callbacks, expected %d", executionCount, tt.callbackCount)
			}
		})
	}
}

func TestRegistry_ExecuteFailure(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Register multiple failure callbacks
	for i := 0; i < 3; i++ {
		registry.RegisterFailure(func(ctx context.Context, event *FailureEvent) {
			executionCount++
			// Verify error is present
			if event.Error == nil {
				t.Error("Expected error in FailureEvent")
			}
		})
	}

	event := &FailureEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Error:     errors.New("test error"),
		StartTime: time.Now().Add(-100 * time.Millisecond),
		EndTime:   time.Now(),
		Duration:  100 * time.Millisecond,
	}

	registry.ExecuteFailure(context.Background(), event)

	if executionCount != 3 {
		t.Errorf("ExecuteFailure() executed %d callbacks, expected 3", executionCount)
	}
}

func TestRegistry_ExecuteStream(t *testing.T) {
	registry := NewRegistry()
	chunks := make([]int, 0)

	// Register stream callback
	registry.RegisterStream(func(ctx context.Context, event *StreamEvent) {
		chunks = append(chunks, event.Index)
	})

	// Execute for multiple chunks
	for i := 0; i < 5; i++ {
		event := &StreamEvent{
			RequestID: "test-req",
			Model:     "gpt-4",
			Provider:  "openai",
			Chunk:     "chunk data",
			Index:     i,
			Timestamp: time.Now(),
		}
		registry.ExecuteStream(context.Background(), event)
	}

	if len(chunks) != 5 {
		t.Errorf("ExecuteStream() received %d chunks, expected 5", len(chunks))
	}

	// Verify chunks in order
	for i := 0; i < 5; i++ {
		if chunks[i] != i {
			t.Errorf("Chunk index %d, expected %d", chunks[i], i)
		}
	}
}

func TestRegistry_ThreadSafety(t *testing.T) {
	registry := NewRegistry()
	var wg sync.WaitGroup

	// Concurrently register callbacks
	for i := 0; i < 100; i++ {
		wg.Add(4)

		go func() {
			defer wg.Done()
			registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
				return nil
			})
		}()

		go func() {
			defer wg.Done()
			registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {})
		}()

		go func() {
			defer wg.Done()
			registry.RegisterFailure(func(ctx context.Context, event *FailureEvent) {})
		}()

		go func() {
			defer wg.Done()
			registry.RegisterStream(func(ctx context.Context, event *StreamEvent) {})
		}()
	}

	wg.Wait()

	// Verify all callbacks were registered
	registry.mu.RLock()
	beforeCount := len(registry.beforeRequest)
	successCount := len(registry.success)
	failureCount := len(registry.failure)
	streamCount := len(registry.stream)
	registry.mu.RUnlock()

	if beforeCount != 100 {
		t.Errorf("Expected 100 before-request callbacks, got %d", beforeCount)
	}
	if successCount != 100 {
		t.Errorf("Expected 100 success callbacks, got %d", successCount)
	}
	if failureCount != 100 {
		t.Errorf("Expected 100 failure callbacks, got %d", failureCount)
	}
	if streamCount != 100 {
		t.Errorf("Expected 100 stream callbacks, got %d", streamCount)
	}
}

func TestRegistry_ConcurrentExecutionAndRegistration(t *testing.T) {
	registry := NewRegistry()
	var wg sync.WaitGroup
	executionCount := int32(0)

	// Register initial callback
	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		atomic.AddInt32(&executionCount, 1)
		return nil
	})

	// Concurrently execute and register
	for i := 0; i < 50; i++ {
		wg.Add(2)

		// Execute callback
		go func() {
			defer wg.Done()
			event := &BeforeRequestEvent{
				RequestID: "test-req",
				Model:     "gpt-4",
				Provider:  "openai",
				StartTime: time.Now(),
			}
			_ = registry.ExecuteBeforeRequest(context.Background(), event)
		}()

		// Register new callback
		go func() {
			defer wg.Done()
			registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
				atomic.AddInt32(&executionCount, 1)
				return nil
			})
		}()
	}

	wg.Wait()

	// Verify no race conditions (test passes if no panic)
	if executionCount <= 0 {
		t.Error("Expected some callbacks to execute")
	}
}

// Benchmark tests
func BenchmarkRegistry_ExecuteBeforeRequest_NoCallbacks(b *testing.B) {
	registry := NewRegistry()
	event := &BeforeRequestEvent{
		RequestID: "bench-req",
		Model:     "gpt-4",
		Provider:  "openai",
		StartTime: time.Now(),
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = registry.ExecuteBeforeRequest(ctx, event)
	}
}

func BenchmarkRegistry_ExecuteBeforeRequest_SingleCallback(b *testing.B) {
	registry := NewRegistry()
	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		return nil
	})

	event := &BeforeRequestEvent{
		RequestID: "bench-req",
		Model:     "gpt-4",
		Provider:  "openai",
		StartTime: time.Now(),
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = registry.ExecuteBeforeRequest(ctx, event)
	}
}

func BenchmarkRegistry_ExecuteBeforeRequest_FiveCallbacks(b *testing.B) {
	registry := NewRegistry()
	for i := 0; i < 5; i++ {
		registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
			return nil
		})
	}

	event := &BeforeRequestEvent{
		RequestID: "bench-req",
		Model:     "gpt-4",
		Provider:  "openai",
		StartTime: time.Now(),
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = registry.ExecuteBeforeRequest(ctx, event)
	}
}

func BenchmarkRegistry_ExecuteSuccess(b *testing.B) {
	registry := NewRegistry()
	registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {
		// Minimal work
	})

	event := &SuccessEvent{
		RequestID: "bench-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Cost:      0.002,
		Tokens:    150,
		Duration:  2 * time.Second,
	}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registry.ExecuteSuccess(ctx, event)
	}
}

// Tests for panic recovery
func TestRegistry_PanicRecovery_BeforeRequest(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Register callback that panics
	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		executionCount++
		panic("test panic")
	})

	// Register second callback to verify execution continues
	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		executionCount++
		return nil
	})

	event := &BeforeRequestEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		StartTime: time.Now(),
	}

	err := registry.ExecuteBeforeRequest(context.Background(), event)

	// Should return error from panic recovery
	if err == nil {
		t.Error("Expected error from panic recovery, got nil")
	}

	// Both callbacks should have executed
	if executionCount != 2 {
		t.Errorf("Expected 2 callbacks to execute, got %d", executionCount)
	}
}

func TestRegistry_PanicRecovery_Success(t *testing.T) {
	registry := NewRegistry()
	executionCount := int32(0)

	// Register callback that panics
	registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {
		atomic.AddInt32(&executionCount, 1)
		panic("test panic")
	})

	// Register second callback to verify execution continues
	registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {
		atomic.AddInt32(&executionCount, 1)
	})

	event := &SuccessEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Cost:      0.002,
		Tokens:    150,
		Duration:  2 * time.Second,
	}

	// Should not panic
	registry.ExecuteSuccess(context.Background(), event)

	// Both callbacks should have executed
	if atomic.LoadInt32(&executionCount) != 2 {
		t.Errorf("Expected 2 callbacks to execute, got %d", executionCount)
	}
}

func TestRegistry_PanicRecovery_Failure(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Register callback that panics
	registry.RegisterFailure(func(ctx context.Context, event *FailureEvent) {
		executionCount++
		panic("test panic")
	})

	// Register second callback to verify execution continues
	registry.RegisterFailure(func(ctx context.Context, event *FailureEvent) {
		executionCount++
	})

	event := &FailureEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Error:     errors.New("test error"),
		StartTime: time.Now().Add(-100 * time.Millisecond),
		EndTime:   time.Now(),
		Duration:  100 * time.Millisecond,
	}

	// Should not panic
	registry.ExecuteFailure(context.Background(), event)

	// Both callbacks should have executed
	if executionCount != 2 {
		t.Errorf("Expected 2 callbacks to execute, got %d", executionCount)
	}
}

func TestRegistry_PanicRecovery_Stream(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Register callback that panics
	registry.RegisterStream(func(ctx context.Context, event *StreamEvent) {
		executionCount++
		panic("test panic")
	})

	// Register second callback to verify execution continues
	registry.RegisterStream(func(ctx context.Context, event *StreamEvent) {
		executionCount++
	})

	event := &StreamEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Chunk:     nil,
		Index:     0,
		Timestamp: time.Now(),
	}

	// Should not panic
	registry.ExecuteStream(context.Background(), event)

	// Both callbacks should have executed
	if executionCount != 2 {
		t.Errorf("Expected 2 callbacks to execute, got %d", executionCount)
	}
}

// Tests for context cancellation
func TestRegistry_ContextCancellation_BeforeRequest(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Register callback
	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		executionCount++
		return nil
	})

	event := &BeforeRequestEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		StartTime: time.Now(),
	}

	err := registry.ExecuteBeforeRequest(ctx, event)

	// Should return context error
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Expected context.Canceled, got %v", err)
	}

	// Callback should not have executed
	if executionCount != 0 {
		t.Errorf("Expected 0 callbacks to execute, got %d", executionCount)
	}
}

func TestRegistry_ContextCancellation_Success(t *testing.T) {
	registry := NewRegistry()
	executionCount := int32(0)

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Register callback
	registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {
		atomic.AddInt32(&executionCount, 1)
	})

	event := &SuccessEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Cost:      0.002,
		Tokens:    150,
		Duration:  2 * time.Second,
	}

	// Should return early without error
	registry.ExecuteSuccess(ctx, event)

	// Callback should not have executed
	if atomic.LoadInt32(&executionCount) != 0 {
		t.Errorf("Expected 0 callbacks to execute, got %d", executionCount)
	}
}

func TestRegistry_ContextCancellation_Failure(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Register callback
	registry.RegisterFailure(func(ctx context.Context, event *FailureEvent) {
		executionCount++
	})

	event := &FailureEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Error:     errors.New("test error"),
		StartTime: time.Now().Add(-100 * time.Millisecond),
		EndTime:   time.Now(),
		Duration:  100 * time.Millisecond,
	}

	// Should return early without error
	registry.ExecuteFailure(ctx, event)

	// Callback should not have executed
	if executionCount != 0 {
		t.Errorf("Expected 0 callbacks to execute, got %d", executionCount)
	}
}

func TestRegistry_ContextCancellation_Stream(t *testing.T) {
	registry := NewRegistry()
	executionCount := 0

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Register callback
	registry.RegisterStream(func(ctx context.Context, event *StreamEvent) {
		executionCount++
	})

	event := &StreamEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		Chunk:     nil,
		Index:     0,
		Timestamp: time.Now(),
	}

	// Should return early without error
	registry.ExecuteStream(ctx, event)

	// Callback should not have executed
	if executionCount != 0 {
		t.Errorf("Expected 0 callbacks to execute, got %d", executionCount)
	}
}

// Test error aggregation
func TestRegistry_ErrorAggregation_BeforeRequest(t *testing.T) {
	registry := NewRegistry()

	// Register multiple callbacks that return errors
	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		return errors.New("error 1")
	})

	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		return errors.New("error 2")
	})

	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
		return errors.New("error 3")
	})

	event := &BeforeRequestEvent{
		RequestID: "test-req",
		Model:     "gpt-4",
		Provider:  "openai",
		StartTime: time.Now(),
	}

	err := registry.ExecuteBeforeRequest(context.Background(), event)

	// Should return aggregated error
	if err == nil {
		t.Fatal("Expected aggregated error, got nil")
	}

	// Error message should contain all errors
	errMsg := err.Error()
	if !strings.Contains(errMsg, "error 1") || !strings.Contains(errMsg, "error 2") || !strings.Contains(errMsg, "error 3") {
		t.Errorf("Expected aggregated error message to contain all errors, got: %s", errMsg)
	}
}
