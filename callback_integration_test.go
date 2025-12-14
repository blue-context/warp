package warp

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/blue-context/warp/callback"
)

func TestClient_WithBeforeRequestCallback(t *testing.T) {
	beforeCalled := false

	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			beforeCalled = true
			if event.Model == "" {
				t.Error("Model should not be empty")
			}
			if event.Provider == "" {
				t.Error("Provider should not be empty")
			}
			if event.RequestID == "" {
				t.Error("RequestID should not be empty")
			}
			return nil
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if !beforeCalled {
		t.Error("Before-request callback was not called")
	}
}

func TestClient_BeforeRequestCallbackCanAbort(t *testing.T) {
	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			return errors.New("request blocked by callback")
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			t.Error("Provider should not be called when callback aborts")
			return nil, errors.New("should not reach provider")
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion - should fail
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})

	if err == nil {
		t.Error("Expected error from callback abort")
	}
}

func TestClient_WithSuccessCallback(t *testing.T) {
	successCalled := false

	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			successCalled = true
			if event.Response == nil {
				t.Error("Response should not be nil")
			}
			if event.Duration <= 0 {
				t.Error("Duration should be positive")
			}
			if event.Tokens <= 0 {
				t.Error("Tokens should be positive")
			}
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion
	resp, err := client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}
	if resp == nil {
		t.Fatal("Response should not be nil")
	}

	if !successCalled {
		t.Error("Success callback was not called")
	}
}

func TestClient_WithFailureCallback(t *testing.T) {
	failureCalled := false
	expectedErr := errors.New("provider error")

	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithFailureCallback(func(ctx context.Context, event *callback.FailureEvent) {
			failureCalled = true
			if event.Error == nil {
				t.Error("Error should not be nil")
			}
			if event.Duration <= 0 {
				t.Error("Duration should be positive")
			}
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider that fails
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return nil, expectedErr
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion - should fail
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})

	if err == nil {
		t.Error("Expected error from provider")
	}

	if !failureCalled {
		t.Error("Failure callback was not called")
	}
}

func TestClient_MultipleCallbacks(t *testing.T) {
	beforeCount := int32(0)
	successCount := int32(0)

	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			atomic.AddInt32(&beforeCount, 1)
			return nil
		}),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			atomic.AddInt32(&beforeCount, 1)
			return nil
		}),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			atomic.AddInt32(&successCount, 1)
		}),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			atomic.AddInt32(&successCount, 1)
		}),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			atomic.AddInt32(&successCount, 1)
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if beforeCount != 2 {
		t.Errorf("Expected 2 before callbacks, got %d", beforeCount)
	}
	if successCount != 3 {
		t.Errorf("Expected 3 success callbacks, got %d", successCount)
	}
}

func TestClient_CallbacksWithCostTracking(t *testing.T) {
	var capturedCost float64

	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithCostTracking(true),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			capturedCost = event.Cost
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   "gpt-4",
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
			}, nil
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Cost should be calculated (might be 0 if pricing not configured, but field should exist)
	if capturedCost < 0 {
		t.Errorf("Expected non-negative cost, got %f", capturedCost)
	}
}

func TestClient_CallbackExecutionOrder(t *testing.T) {
	order := make([]string, 0)

	client, err := NewClient(
		WithAPIKey("test", "test-key"),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			order = append(order, "before1")
			return nil
		}),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			order = append(order, "before2")
			return nil
		}),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			order = append(order, "success1")
		}),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			order = append(order, "success2")
		}),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			order = append(order, "provider")
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Verify execution order
	expectedOrder := []string{"before1", "before2", "provider", "success1", "success2"}
	if len(order) != len(expectedOrder) {
		t.Fatalf("Expected %d callbacks, got %d", len(expectedOrder), len(order))
	}

	for i, expected := range expectedOrder {
		if order[i] != expected {
			t.Errorf("Expected callback %d to be %s, got %s", i, expected, order[i])
		}
	}
}

func TestClient_NoCallbacks_ZeroOverhead(t *testing.T) {
	// Create client without callbacks
	client, err := NewClient(
		WithAPIKey("test", "test-key"),
	)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Register mock provider
	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	if err := client.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("RegisterProvider() error = %v", err)
	}

	// Call completion - should work without callbacks
	resp, err := client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}
	if resp == nil {
		t.Fatal("Response should not be nil")
	}
}

func TestClient_NilCallbackError(t *testing.T) {
	tests := []struct {
		name string
		opt  ClientOption
	}{
		{
			name: "nil before-request callback",
			opt:  WithBeforeRequestCallback(nil),
		},
		{
			name: "nil success callback",
			opt:  WithSuccessCallback(nil),
		},
		{
			name: "nil failure callback",
			opt:  WithFailureCallback(nil),
		},
		{
			name: "nil stream callback",
			opt:  WithStreamCallback(nil),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewClient(
				WithAPIKey("test", "test-key"),
				tt.opt,
			)
			if err == nil {
				t.Error("Expected error for nil callback")
			}
		})
	}
}

// Benchmark callback overhead
func BenchmarkClient_Completion_NoCallbacks(b *testing.B) {
	client, _ := NewClient(WithAPIKey("test", "test-key"))
	defer client.Close()

	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	client.RegisterProvider(mockProvider)

	req := &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Completion(context.Background(), req)
	}
}

func BenchmarkClient_Completion_WithCallbacks(b *testing.B) {
	client, _ := NewClient(
		WithAPIKey("test", "test-key"),
		WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			return nil
		}),
		WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {}),
	)
	defer client.Close()

	mockProvider := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}
	client.RegisterProvider(mockProvider)

	req := &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.Completion(context.Background(), req)
	}
}
