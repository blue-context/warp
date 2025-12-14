package warp

import (
	"context"
	"crypto/rand"
	"fmt"
	mathrand "math/rand"
	"time"
)

// contextKey is a private type for context keys to avoid collisions.
type contextKey string

const (
	contextKeyRequestID contextKey = "litellm_request_id"
	contextKeyProvider  contextKey = "litellm_provider"
	contextKeyModel     contextKey = "litellm_model"
	contextKeyStartTime contextKey = "litellm_start_time"
)

// WithRequestID adds a request ID to the context.
//
// The request ID is used to track requests through the system for debugging
// and logging purposes.
//
// Example:
//
//	ctx = warp.WithRequestID(ctx, "req-123")
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, contextKeyRequestID, id)
}

// RequestIDFromContext retrieves the request ID from the context.
//
// Returns an empty string if no request ID is found.
//
// Example:
//
//	requestID := warp.RequestIDFromContext(ctx)
//	if requestID != "" {
//	    log.Printf("Processing request %s", requestID)
//	}
func RequestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(contextKeyRequestID).(string); ok {
		return id
	}
	return ""
}

// WithGeneratedRequestID adds a generated request ID to the context.
//
// The generated ID includes a timestamp and random component to ensure uniqueness.
//
// Example:
//
//	ctx = warp.WithGeneratedRequestID(ctx)
//	requestID := warp.RequestIDFromContext(ctx)
func WithGeneratedRequestID(ctx context.Context) context.Context {
	id := generateRequestID()
	return WithRequestID(ctx, id)
}

// generateRequestID generates a unique request ID.
//
// The ID format is: req_<timestamp>_<random_hex>
// This ensures uniqueness across concurrent requests.
//
// If crypto/rand fails (rare but possible), falls back to math/rand
// to ensure IDs are always unique.
func generateRequestID() string {
	timestamp := time.Now().UnixNano()
	randomBytes := make([]byte, 4)

	// Try crypto/rand first for cryptographically secure randomness
	_, err := rand.Read(randomBytes)
	if err != nil {
		// Fallback to math/rand if crypto/rand fails
		// This is less secure but ensures IDs are always unique
		for i := range randomBytes {
			randomBytes[i] = byte(mathrand.Intn(256))
		}
	}

	return fmt.Sprintf("req_%d_%x", timestamp, randomBytes)
}

// WithProvider adds the provider name to the context.
//
// The provider name identifies which LLM provider is being used
// (e.g., "openai", "anthropic").
//
// Example:
//
//	ctx = warp.WithProvider(ctx, "openai")
func WithProvider(ctx context.Context, provider string) context.Context {
	return context.WithValue(ctx, contextKeyProvider, provider)
}

// ProviderFromContext retrieves the provider name from the context.
//
// Returns an empty string if no provider is found.
//
// Example:
//
//	provider := warp.ProviderFromContext(ctx)
//	if provider != "" {
//	    log.Printf("Using provider: %s", provider)
//	}
func ProviderFromContext(ctx context.Context) string {
	if provider, ok := ctx.Value(contextKeyProvider).(string); ok {
		return provider
	}
	return ""
}

// WithModel adds the model name to the context.
//
// The model name identifies which specific model is being used
// (e.g., "gpt-4", "claude-3-sonnet").
//
// Example:
//
//	ctx = warp.WithModel(ctx, "gpt-4")
func WithModel(ctx context.Context, model string) context.Context {
	return context.WithValue(ctx, contextKeyModel, model)
}

// ModelFromContext retrieves the model name from the context.
//
// Returns an empty string if no model is found.
//
// Example:
//
//	model := warp.ModelFromContext(ctx)
//	if model != "" {
//	    log.Printf("Using model: %s", model)
//	}
func ModelFromContext(ctx context.Context) string {
	if model, ok := ctx.Value(contextKeyModel).(string); ok {
		return model
	}
	return ""
}

// WithStartTime adds the request start time to the context.
//
// This is used internally to track request latency and duration.
//
// Example:
//
//	ctx = warp.WithStartTime(ctx, time.Now())
func WithStartTime(ctx context.Context, t time.Time) context.Context {
	return context.WithValue(ctx, contextKeyStartTime, t)
}

// StartTimeFromContext retrieves the start time from the context.
//
// Returns the zero time if no start time is found.
// Use time.Time.IsZero() to check if a valid time was retrieved.
//
// Example:
//
//	startTime := warp.StartTimeFromContext(ctx)
//	if !startTime.IsZero() {
//	    duration := time.Since(startTime)
//	    log.Printf("Request took %v", duration)
//	}
func StartTimeFromContext(ctx context.Context) time.Time {
	if t, ok := ctx.Value(contextKeyStartTime).(time.Time); ok {
		return t
	}
	return time.Time{}
}
