// Package callback provides lifecycle hooks for Warp requests.
//
// Callbacks allow you to track, log, and monitor API requests at different
// stages of the request lifecycle: before request, after success, after failure,
// and during streaming.
//
// Thread Safety: All callback types must be safe for concurrent calls.
// The Registry manages callbacks thread-safely using the snapshot pattern.
package callback

import (
	"context"
	"time"
)

// BeforeRequestCallback is called before sending a request to the LLM provider.
//
// The callback can inspect and modify the event, or return an error to abort
// the request. If an error is returned, the request will not be sent and the
// error will be returned to the caller.
//
// Thread Safety: Must be safe for concurrent calls.
//
// Example:
//
//	func logRequest(ctx context.Context, event *BeforeRequestEvent) error {
//	    log.Printf("Request to %s/%s", event.Provider, event.Model)
//	    return nil
//	}
type BeforeRequestCallback func(ctx context.Context, event *BeforeRequestEvent) error

// SuccessCallback is called after a successful response from the LLM provider.
//
// Success callbacks are informational only. Errors returned from success
// callbacks are logged but do not fail the request, since the API call
// already succeeded.
//
// Thread Safety: Must be safe for concurrent calls.
//
// Example:
//
//	func trackCost(ctx context.Context, event *SuccessEvent) {
//	    log.Printf("Cost: $%.4f, Tokens: %d", event.Cost, event.Tokens)
//	}
type SuccessCallback func(ctx context.Context, event *SuccessEvent)

// FailureCallback is called after a failed request.
//
// Failure callbacks are informational only. They cannot modify the error
// or prevent it from being returned to the caller.
//
// Thread Safety: Must be safe for concurrent calls.
//
// Example:
//
//	func logError(ctx context.Context, event *FailureEvent) {
//	    log.Printf("Request failed: %v", event.Error)
//	}
type FailureCallback func(ctx context.Context, event *FailureEvent)

// StreamCallback is called for each streaming chunk.
//
// Stream callbacks receive each chunk as it arrives from the provider.
// Errors are logged but do not interrupt the stream.
//
// Thread Safety: Must be safe for concurrent calls.
//
// Example:
//
//	func logChunk(ctx context.Context, event *StreamEvent) {
//	    log.Printf("Chunk %d: %s", event.Index, event.Chunk.Choices[0].Delta.Content)
//	}
type StreamCallback func(ctx context.Context, event *StreamEvent)

// BeforeRequestEvent contains data for before-request callbacks.
type BeforeRequestEvent struct {
	// RequestID uniquely identifies this request
	RequestID string

	// Model is the model name (without provider prefix)
	Model string

	// Provider is the provider name (e.g., "openai", "anthropic")
	Provider string

	// Request is the completion request being sent.
	// Type: *warp.CompletionRequest (interface{} to avoid circular import)
	Request interface{}

	// StartTime is when the request started
	StartTime time.Time
}

// SuccessEvent contains data for success callbacks.
type SuccessEvent struct {
	// RequestID uniquely identifies this request
	RequestID string

	// Model is the model name (without provider prefix)
	Model string

	// Provider is the provider name (e.g., "openai", "anthropic")
	Provider string

	// Request is the completion request that was sent.
	// Type: *warp.CompletionRequest (interface{} to avoid circular import)
	Request interface{}

	// Response is the completion response received.
	// Type: *warp.CompletionResponse (interface{} to avoid circular import)
	Response interface{}

	// StartTime is when the request started
	StartTime time.Time

	// EndTime is when the request completed
	EndTime time.Time

	// Duration is the total request duration (EndTime - StartTime)
	Duration time.Duration

	// Cost is the estimated cost in USD (0 if not available)
	Cost float64

	// Tokens is the total number of tokens used (0 if not available)
	Tokens int
}

// FailureEvent contains data for failure callbacks.
type FailureEvent struct {
	// RequestID uniquely identifies this request
	RequestID string

	// Model is the model name (without provider prefix)
	Model string

	// Provider is the provider name (e.g., "openai", "anthropic")
	Provider string

	// Request is the completion request that was sent.
	// Type: *warp.CompletionRequest (interface{} to avoid circular import)
	Request interface{}

	// Error is the error that occurred
	Error error

	// StartTime is when the request started
	StartTime time.Time

	// EndTime is when the request failed
	EndTime time.Time

	// Duration is the total request duration until failure
	Duration time.Duration
}

// StreamEvent contains data for streaming callbacks.
type StreamEvent struct {
	// RequestID uniquely identifies this request
	RequestID string

	// Model is the model name (without provider prefix)
	Model string

	// Provider is the provider name (e.g., "openai", "anthropic")
	Provider string

	// Chunk is the streaming chunk received.
	// Type: *warp.CompletionChunk (interface{} to avoid circular import)
	Chunk interface{}

	// Index is the zero-based index of this chunk in the stream
	Index int

	// Timestamp is when this chunk was received
	Timestamp time.Time
}
