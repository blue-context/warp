package callback

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

func TestBeforeRequestCallback(t *testing.T) {
	tests := []struct {
		name        string
		callback    BeforeRequestCallback
		event       *BeforeRequestEvent
		expectError bool
	}{
		{
			name: "successful callback",
			callback: func(ctx context.Context, event *BeforeRequestEvent) error {
				if event.Model == "" {
					return errors.New("model is empty")
				}
				return nil
			},
			event: &BeforeRequestEvent{
				RequestID: "req-123",
				Model:     "gpt-4",
				Provider:  "openai",
				StartTime: time.Now(),
			},
			expectError: false,
		},
		{
			name: "callback returns error",
			callback: func(ctx context.Context, event *BeforeRequestEvent) error {
				return errors.New("validation failed")
			},
			event: &BeforeRequestEvent{
				RequestID: "req-456",
				Model:     "gpt-4",
				Provider:  "openai",
				StartTime: time.Now(),
			},
			expectError: true,
		},
		{
			name: "callback with nil event",
			callback: func(ctx context.Context, event *BeforeRequestEvent) error {
				if event == nil {
					return errors.New("event is nil")
				}
				return nil
			},
			event:       nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.callback(context.Background(), tt.event)
			if (err != nil) != tt.expectError {
				t.Errorf("callback() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

func TestSuccessCallback(t *testing.T) {
	tests := []struct {
		name     string
		callback SuccessCallback
		event    *SuccessEvent
	}{
		{
			name: "log success with cost",
			callback: func(ctx context.Context, event *SuccessEvent) {
				if event.Cost <= 0 {
					t.Error("expected positive cost")
				}
				if event.Tokens <= 0 {
					t.Error("expected positive token count")
				}
				if event.Duration <= 0 {
					t.Error("expected positive duration")
				}
			},
			event: &SuccessEvent{
				RequestID: "req-789",
				Model:     "gpt-4",
				Provider:  "openai",
				StartTime: time.Now().Add(-2 * time.Second),
				EndTime:   time.Now(),
				Duration:  2 * time.Second,
				Cost:      0.002,
				Tokens:    150,
			},
		},
		{
			name: "callback with zero cost",
			callback: func(ctx context.Context, event *SuccessEvent) {
				if event.Cost != 0 {
					t.Errorf("expected zero cost, got %f", event.Cost)
				}
			},
			event: &SuccessEvent{
				RequestID: "req-999",
				Model:     "gpt-4",
				Provider:  "openai",
				StartTime: time.Now(),
				EndTime:   time.Now(),
				Duration:  100 * time.Millisecond,
				Cost:      0,
				Tokens:    0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.callback(context.Background(), tt.event)
		})
	}
}

func TestFailureCallback(t *testing.T) {
	tests := []struct {
		name     string
		callback FailureCallback
		event    *FailureEvent
	}{
		{
			name: "log failure with error",
			callback: func(ctx context.Context, event *FailureEvent) {
				if event.Error == nil {
					t.Error("expected error to be set")
				}
				if event.Duration <= 0 {
					t.Error("expected positive duration")
				}
			},
			event: &FailureEvent{
				RequestID: "req-error",
				Model:     "gpt-4",
				Provider:  "openai",
				Error:     errors.New("rate limit exceeded"),
				StartTime: time.Now().Add(-100 * time.Millisecond),
				EndTime:   time.Now(),
				Duration:  100 * time.Millisecond,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.callback(context.Background(), tt.event)
		})
	}
}

func TestStreamCallback(t *testing.T) {
	tests := []struct {
		name     string
		callback StreamCallback
		event    *StreamEvent
	}{
		{
			name: "log stream chunk",
			callback: func(ctx context.Context, event *StreamEvent) {
				if event.Index < 0 {
					t.Error("expected non-negative index")
				}
				if event.Chunk == nil {
					t.Error("expected chunk to be set")
				}
			},
			event: &StreamEvent{
				RequestID: "req-stream",
				Model:     "gpt-4",
				Provider:  "openai",
				Chunk:     "Hello",
				Index:     5,
				Timestamp: time.Now(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.callback(context.Background(), tt.event)
		})
	}
}

func TestCallbackEventStructures(t *testing.T) {
	t.Run("BeforeRequestEvent has required fields", func(t *testing.T) {
		event := &BeforeRequestEvent{
			RequestID: "req-123",
			Model:     "gpt-4",
			Provider:  "openai",
			Request:   "test request",
			StartTime: time.Now(),
		}

		if event.RequestID == "" {
			t.Error("RequestID should not be empty")
		}
		if event.Model == "" {
			t.Error("Model should not be empty")
		}
		if event.Provider == "" {
			t.Error("Provider should not be empty")
		}
		if event.StartTime.IsZero() {
			t.Error("StartTime should not be zero")
		}
	})

	t.Run("SuccessEvent has complete metadata", func(t *testing.T) {
		startTime := time.Now().Add(-2 * time.Second)
		endTime := time.Now()

		event := &SuccessEvent{
			RequestID: "req-456",
			Model:     "gpt-4",
			Provider:  "openai",
			Request:   "test request",
			Response:  "test response",
			StartTime: startTime,
			EndTime:   endTime,
			Duration:  endTime.Sub(startTime),
			Cost:      0.002,
			Tokens:    150,
		}

		if event.Duration <= 0 {
			t.Error("Duration should be positive")
		}
		if event.Cost <= 0 {
			t.Error("Cost should be positive")
		}
		if event.Tokens <= 0 {
			t.Error("Tokens should be positive")
		}
	})

	t.Run("FailureEvent contains error", func(t *testing.T) {
		event := &FailureEvent{
			RequestID: "req-789",
			Model:     "gpt-4",
			Provider:  "openai",
			Request:   "test request",
			Error:     errors.New("test error"),
			StartTime: time.Now().Add(-100 * time.Millisecond),
			EndTime:   time.Now(),
			Duration:  100 * time.Millisecond,
		}

		if event.Error == nil {
			t.Error("Error should not be nil")
		}
	})

	t.Run("StreamEvent has chunk index", func(t *testing.T) {
		event := &StreamEvent{
			RequestID: "req-stream",
			Model:     "gpt-4",
			Provider:  "openai",
			Chunk:     "chunk data",
			Index:     10,
			Timestamp: time.Now(),
		}

		if event.Index < 0 {
			t.Error("Index should be non-negative")
		}
		if event.Timestamp.IsZero() {
			t.Error("Timestamp should not be zero")
		}
	})
}

// Benchmark callback execution overhead
func BenchmarkBeforeRequestCallback(b *testing.B) {
	callback := func(ctx context.Context, event *BeforeRequestEvent) error {
		return nil
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
		_ = callback(ctx, event)
	}
}

func BenchmarkSuccessCallback(b *testing.B) {
	callback := func(ctx context.Context, event *SuccessEvent) {
		// Simulate minimal logging
		_ = fmt.Sprintf("Cost: %.4f", event.Cost)
	}

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
		callback(ctx, event)
	}
}
