package warp

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestWithRequestID(t *testing.T) {
	ctx := context.Background()
	requestID := "req-test-123"

	ctx = WithRequestID(ctx, requestID)
	got := RequestIDFromContext(ctx)

	if got != requestID {
		t.Errorf("RequestIDFromContext() = %s, want %s", got, requestID)
	}
}

func TestRequestIDFromContext_NotFound(t *testing.T) {
	ctx := context.Background()
	got := RequestIDFromContext(ctx)

	if got != "" {
		t.Errorf("RequestIDFromContext() = %s, want empty string", got)
	}
}

func TestWithGeneratedRequestID(t *testing.T) {
	ctx := context.Background()
	ctx = WithGeneratedRequestID(ctx)

	requestID := RequestIDFromContext(ctx)
	if requestID == "" {
		t.Error("WithGeneratedRequestID() did not generate a request ID")
	}

	// Check format: req_<timestamp>_<hex>
	if !strings.HasPrefix(requestID, "req_") {
		t.Errorf("Request ID should start with 'req_', got %s", requestID)
	}

	parts := strings.Split(requestID, "_")
	if len(parts) != 3 {
		t.Errorf("Request ID should have 3 parts, got %d: %s", len(parts), requestID)
	}
}

func TestGenerateRequestID_Uniqueness(t *testing.T) {
	// Generate multiple IDs and ensure they're unique
	ids := make(map[string]bool)
	count := 1000

	for i := 0; i < count; i++ {
		id := generateRequestID()
		if ids[id] {
			t.Errorf("Duplicate request ID generated: %s", id)
		}
		ids[id] = true

		// Verify format
		if !strings.HasPrefix(id, "req_") {
			t.Errorf("Request ID should start with 'req_', got %s", id)
		}
	}

	if len(ids) != count {
		t.Errorf("Expected %d unique IDs, got %d", count, len(ids))
	}
}

func TestWithProvider(t *testing.T) {
	tests := []struct {
		name     string
		provider string
	}{
		{
			name:     "openai provider",
			provider: "openai",
		},
		{
			name:     "anthropic provider",
			provider: "anthropic",
		},
		{
			name:     "empty provider",
			provider: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ctx = WithProvider(ctx, tt.provider)
			got := ProviderFromContext(ctx)

			if got != tt.provider {
				t.Errorf("ProviderFromContext() = %s, want %s", got, tt.provider)
			}
		})
	}
}

func TestProviderFromContext_NotFound(t *testing.T) {
	ctx := context.Background()
	got := ProviderFromContext(ctx)

	if got != "" {
		t.Errorf("ProviderFromContext() = %s, want empty string", got)
	}
}

func TestWithModel(t *testing.T) {
	tests := []struct {
		name  string
		model string
	}{
		{
			name:  "gpt-4 model",
			model: "gpt-4",
		},
		{
			name:  "claude-3-sonnet model",
			model: "claude-3-sonnet",
		},
		{
			name:  "empty model",
			model: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			ctx = WithModel(ctx, tt.model)
			got := ModelFromContext(ctx)

			if got != tt.model {
				t.Errorf("ModelFromContext() = %s, want %s", got, tt.model)
			}
		})
	}
}

func TestModelFromContext_NotFound(t *testing.T) {
	ctx := context.Background()
	got := ModelFromContext(ctx)

	if got != "" {
		t.Errorf("ModelFromContext() = %s, want empty string", got)
	}
}

func TestWithStartTime(t *testing.T) {
	ctx := context.Background()
	startTime := time.Now()

	ctx = WithStartTime(ctx, startTime)
	got := StartTimeFromContext(ctx)

	if !got.Equal(startTime) {
		t.Errorf("StartTimeFromContext() = %v, want %v", got, startTime)
	}
}

func TestStartTimeFromContext_NotFound(t *testing.T) {
	ctx := context.Background()
	got := StartTimeFromContext(ctx)

	if !got.IsZero() {
		t.Errorf("StartTimeFromContext() = %v, want zero time", got)
	}
}

func TestMultipleContextValues(t *testing.T) {
	ctx := context.Background()
	requestID := "req-multi-123"
	provider := "openai"
	model := "gpt-4"
	startTime := time.Now()

	// Add all values
	ctx = WithRequestID(ctx, requestID)
	ctx = WithProvider(ctx, provider)
	ctx = WithModel(ctx, model)
	ctx = WithStartTime(ctx, startTime)

	// Verify all values
	if got := RequestIDFromContext(ctx); got != requestID {
		t.Errorf("RequestIDFromContext() = %s, want %s", got, requestID)
	}
	if got := ProviderFromContext(ctx); got != provider {
		t.Errorf("ProviderFromContext() = %s, want %s", got, provider)
	}
	if got := ModelFromContext(ctx); got != model {
		t.Errorf("ModelFromContext() = %s, want %s", got, model)
	}
	if got := StartTimeFromContext(ctx); !got.Equal(startTime) {
		t.Errorf("StartTimeFromContext() = %v, want %v", got, startTime)
	}
}

func TestContextValueOverwrite(t *testing.T) {
	ctx := context.Background()

	// Set initial value
	ctx = WithRequestID(ctx, "req-first")
	if got := RequestIDFromContext(ctx); got != "req-first" {
		t.Errorf("Initial RequestIDFromContext() = %s, want req-first", got)
	}

	// Overwrite with new value
	ctx = WithRequestID(ctx, "req-second")
	if got := RequestIDFromContext(ctx); got != "req-second" {
		t.Errorf("Updated RequestIDFromContext() = %s, want req-second", got)
	}
}

func TestContextChaining(t *testing.T) {
	ctx := context.Background()

	// Chain multiple context helpers
	ctx = WithGeneratedRequestID(
		WithProvider(
			WithModel(
				WithStartTime(ctx, time.Now()),
				"gpt-4",
			),
			"openai",
		),
	)

	// Verify all values are set
	if RequestIDFromContext(ctx) == "" {
		t.Error("RequestID should be set")
	}
	if ProviderFromContext(ctx) != "openai" {
		t.Error("Provider should be 'openai'")
	}
	if ModelFromContext(ctx) != "gpt-4" {
		t.Error("Model should be 'gpt-4'")
	}
	if StartTimeFromContext(ctx).IsZero() {
		t.Error("StartTime should be set")
	}
}

func TestContextInheritance(t *testing.T) {
	parent := context.Background()
	parent = WithRequestID(parent, "req-parent")

	// Create child context
	child, cancel := context.WithTimeout(parent, 5*time.Second)
	defer cancel()

	// Child should inherit parent's values
	if got := RequestIDFromContext(child); got != "req-parent" {
		t.Errorf("Child context RequestID = %s, want req-parent", got)
	}

	// Add value to child
	child = WithProvider(child, "openai")

	// Parent should not have child's values
	if got := ProviderFromContext(parent); got != "" {
		t.Errorf("Parent context should not have provider, got %s", got)
	}

	// Child should have both values
	if got := RequestIDFromContext(child); got != "req-parent" {
		t.Errorf("Child context RequestID = %s, want req-parent", got)
	}
	if got := ProviderFromContext(child); got != "openai" {
		t.Errorf("Child context Provider = %s, want openai", got)
	}
}

func TestStartTimeLatencyCalculation(t *testing.T) {
	ctx := context.Background()
	startTime := time.Now()

	ctx = WithStartTime(ctx, startTime)

	// Simulate some work
	time.Sleep(10 * time.Millisecond)

	retrieved := StartTimeFromContext(ctx)
	duration := time.Since(retrieved)

	if duration < 10*time.Millisecond {
		t.Errorf("Duration should be at least 10ms, got %v", duration)
	}

	if duration > 100*time.Millisecond {
		t.Errorf("Duration should be less than 100ms, got %v", duration)
	}
}

func TestContextKeyIsolation(t *testing.T) {
	ctx := context.Background()

	// Set a value using context.WithValue with a custom key
	type customKey string
	ctx = context.WithValue(ctx, customKey("litellm_request_id"), "custom-value")

	// Our RequestIDFromContext should not retrieve this value
	// because it uses a different key type (contextKey vs customKey)
	if got := RequestIDFromContext(ctx); got != "" {
		t.Errorf("RequestIDFromContext() should not find custom key value, got %s", got)
	}

	// Now set it properly
	ctx = WithRequestID(ctx, "proper-value")

	// Should get the proper value
	if got := RequestIDFromContext(ctx); got != "proper-value" {
		t.Errorf("RequestIDFromContext() = %s, want proper-value", got)
	}
}

func BenchmarkWithRequestID(b *testing.B) {
	ctx := context.Background()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ctx = WithRequestID(ctx, "req-bench-123")
	}
}

func BenchmarkGenerateRequestID(b *testing.B) {
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = generateRequestID()
	}
}

func BenchmarkWithGeneratedRequestID(b *testing.B) {
	ctx := context.Background()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		ctx = WithGeneratedRequestID(ctx)
	}
}

func BenchmarkContextFromContext(b *testing.B) {
	ctx := context.Background()
	ctx = WithRequestID(ctx, "req-bench-123")
	ctx = WithProvider(ctx, "openai")
	ctx = WithModel(ctx, "gpt-4")
	ctx = WithStartTime(ctx, time.Now())

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = RequestIDFromContext(ctx)
		_ = ProviderFromContext(ctx)
		_ = ModelFromContext(ctx)
		_ = StartTimeFromContext(ctx)
	}
}
