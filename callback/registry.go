package callback

import (
	"context"
	"fmt"
	"sync"
)

// Registry manages registered callbacks for the request lifecycle.
//
// The Registry is thread-safe and uses the snapshot pattern for execution:
// callbacks are copied under a read lock before execution, preventing lock
// contention during callback execution.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	registry := callback.NewRegistry()
//	registry.RegisterBeforeRequest(myCallback)
//	err := registry.ExecuteBeforeRequest(ctx, event)
type Registry struct {
	beforeRequest []BeforeRequestCallback
	success       []SuccessCallback
	failure       []FailureCallback
	stream        []StreamCallback
	mu            sync.RWMutex
}

// NewRegistry creates a new callback registry.
//
// The registry is initialized with empty callback lists.
//
// Example:
//
//	registry := callback.NewRegistry()
func NewRegistry() *Registry {
	return &Registry{
		beforeRequest: make([]BeforeRequestCallback, 0),
		success:       make([]SuccessCallback, 0),
		failure:       make([]FailureCallback, 0),
		stream:        make([]StreamCallback, 0),
	}
}

// RegisterBeforeRequest registers a before-request callback.
//
// The callback will be executed before sending requests to the provider.
// Callbacks are executed in registration order.
//
// If the callback is nil, this method is a no-op.
//
// Example:
//
//	registry.RegisterBeforeRequest(func(ctx context.Context, event *BeforeRequestEvent) error {
//	    log.Printf("Request to %s", event.Model)
//	    return nil
//	})
func (r *Registry) RegisterBeforeRequest(cb BeforeRequestCallback) {
	if cb == nil {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.beforeRequest = append(r.beforeRequest, cb)
}

// RegisterSuccess registers a success callback.
//
// The callback will be executed after successful responses.
// Callbacks are executed in registration order.
//
// If the callback is nil, this method is a no-op.
//
// Example:
//
//	registry.RegisterSuccess(func(ctx context.Context, event *SuccessEvent) {
//	    log.Printf("Success! Cost: $%.4f", event.Cost)
//	})
func (r *Registry) RegisterSuccess(cb SuccessCallback) {
	if cb == nil {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.success = append(r.success, cb)
}

// RegisterFailure registers a failure callback.
//
// The callback will be executed after failed requests.
// Callbacks are executed in registration order.
//
// If the callback is nil, this method is a no-op.
//
// Example:
//
//	registry.RegisterFailure(func(ctx context.Context, event *FailureEvent) {
//	    log.Printf("Failure: %v", event.Error)
//	})
func (r *Registry) RegisterFailure(cb FailureCallback) {
	if cb == nil {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.failure = append(r.failure, cb)
}

// RegisterStream registers a streaming callback.
//
// The callback will be executed for each streaming chunk.
// Callbacks are executed in registration order.
//
// If the callback is nil, this method is a no-op.
//
// Example:
//
//	registry.RegisterStream(func(ctx context.Context, event *StreamEvent) {
//	    log.Printf("Chunk %d received", event.Index)
//	})
func (r *Registry) RegisterStream(cb StreamCallback) {
	if cb == nil {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.stream = append(r.stream, cb)
}

// ExecuteBeforeRequest executes all before-request callbacks.
//
// Callbacks are executed sequentially in registration order.
// If any callback returns an error, execution continues but all errors
// are collected and returned as an aggregated error.
//
// Context cancellation is checked before each callback execution.
// Panics in callbacks are recovered and converted to errors.
//
// Returns nil if all callbacks succeed or if no callbacks are registered.
//
// Example:
//
//	err := registry.ExecuteBeforeRequest(ctx, &BeforeRequestEvent{
//	    RequestID: "req-123",
//	    Model: "gpt-4",
//	    Provider: "openai",
//	    StartTime: time.Now(),
//	})
//	if err != nil {
//	    // Request was aborted by callback
//	    return err
//	}
func (r *Registry) ExecuteBeforeRequest(ctx context.Context, event *BeforeRequestEvent) error {
	// Snapshot callbacks under read lock
	r.mu.RLock()
	callbacks := make([]BeforeRequestCallback, len(r.beforeRequest))
	copy(callbacks, r.beforeRequest)
	r.mu.RUnlock()

	// Early return if no callbacks (zero overhead)
	if len(callbacks) == 0 {
		return nil
	}

	// Collect all errors from callbacks
	var errs []error

	// Execute callbacks sequentially, collect all errors
	for _, cb := range callbacks {
		// Check context cancellation before each callback
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Execute callback with panic recovery
		if err := func() (callbackErr error) {
			defer func() {
				if r := recover(); r != nil {
					callbackErr = fmt.Errorf("callback panic: %v", r)
				}
			}()

			return cb(ctx, event)
		}(); err != nil {
			errs = append(errs, err)
		}
	}

	// Aggregate errors if any occurred
	if len(errs) > 0 {
		return fmt.Errorf("before-request callbacks failed: %v", errs)
	}

	return nil
}

// ExecuteSuccess executes all success callbacks.
//
// Callbacks are executed sequentially in registration order.
// Errors and panics from callbacks are ignored since the request has already succeeded.
// Context cancellation is checked before each callback execution.
// This is informational only and cannot fail the request.
//
// Example:
//
//	registry.ExecuteSuccess(ctx, &SuccessEvent{
//	    RequestID: "req-123",
//	    Model: "gpt-4",
//	    Provider: "openai",
//	    Cost: 0.002,
//	    Tokens: 150,
//	    Duration: 2 * time.Second,
//	})
func (r *Registry) ExecuteSuccess(ctx context.Context, event *SuccessEvent) {
	// Snapshot callbacks under read lock
	r.mu.RLock()
	callbacks := make([]SuccessCallback, len(r.success))
	copy(callbacks, r.success)
	r.mu.RUnlock()

	// Early return if no callbacks (zero overhead)
	if len(callbacks) == 0 {
		return
	}

	// Execute all callbacks
	for _, cb := range callbacks {
		// Check context cancellation before each callback
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Execute callback with panic recovery
		func() {
			defer func() {
				if r := recover(); r != nil {
					// Log panic but don't crash (informational callbacks only)
					// In production, this would use a logger
					_ = r
				}
			}()

			cb(ctx, event)
		}()
	}
}

// ExecuteFailure executes all failure callbacks.
//
// Callbacks are executed sequentially in registration order.
// Errors and panics from callbacks are ignored since the request has already failed.
// Context cancellation is checked before each callback execution.
// This is informational only and cannot change the error.
//
// Example:
//
//	registry.ExecuteFailure(ctx, &FailureEvent{
//	    RequestID: "req-123",
//	    Model: "gpt-4",
//	    Provider: "openai",
//	    Error: fmt.Errorf("rate limit exceeded"),
//	    Duration: 100 * time.Millisecond,
//	})
func (r *Registry) ExecuteFailure(ctx context.Context, event *FailureEvent) {
	// Snapshot callbacks under read lock
	r.mu.RLock()
	callbacks := make([]FailureCallback, len(r.failure))
	copy(callbacks, r.failure)
	r.mu.RUnlock()

	// Early return if no callbacks (zero overhead)
	if len(callbacks) == 0 {
		return
	}

	// Execute all callbacks
	for _, cb := range callbacks {
		// Check context cancellation before each callback
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Execute callback with panic recovery
		func() {
			defer func() {
				if r := recover(); r != nil {
					// Log panic but don't crash (informational callbacks only)
					// In production, this would use a logger
					_ = r
				}
			}()

			cb(ctx, event)
		}()
	}
}

// ExecuteStream executes all streaming callbacks.
//
// Callbacks are executed sequentially in registration order.
// Errors and panics from callbacks are ignored to avoid interrupting the stream.
// Context cancellation is checked before each callback execution.
//
// Example:
//
//	registry.ExecuteStream(ctx, &StreamEvent{
//	    RequestID: "req-123",
//	    Model: "gpt-4",
//	    Provider: "openai",
//	    Chunk: chunk,
//	    Index: 5,
//	    Timestamp: time.Now(),
//	})
func (r *Registry) ExecuteStream(ctx context.Context, event *StreamEvent) {
	// Snapshot callbacks under read lock
	r.mu.RLock()
	callbacks := make([]StreamCallback, len(r.stream))
	copy(callbacks, r.stream)
	r.mu.RUnlock()

	// Early return if no callbacks (zero overhead)
	if len(callbacks) == 0 {
		return
	}

	// Execute all callbacks
	for _, cb := range callbacks {
		// Check context cancellation before each callback
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Execute callback with panic recovery
		func() {
			defer func() {
				if r := recover(); r != nil {
					// Log panic but don't crash (informational callbacks only)
					// In production, this would use a logger
					_ = r
				}
			}()

			cb(ctx, event)
		}()
	}
}
