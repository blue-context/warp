package warp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/blue-context/warp/cache"
	"github.com/blue-context/warp/callback"
)

// Completion creates a chat completion.
//
// The model field in the request must be in the format "provider/model-name",
// for example "openai/gpt-4" or "anthropic/claude-3-sonnet".
//
// Example:
//
//	resp, err := client.Completion(ctx, &warp.CompletionRequest{
//	    Model: "openai/gpt-4",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
func (c *client) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Record start time
	startTime := time.Now()
	ctx = WithStartTime(ctx, startTime)

	// Parse model string
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Add provider and model to context
	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Execute before-request callbacks
	if c.callbacks != nil {
		beforeEvent := &callback.BeforeRequestEvent{
			RequestID: RequestIDFromContext(ctx),
			Model:     modelName,
			Provider:  providerName,
			Request:   req,
			StartTime: startTime,
		}
		if err := c.callbacks.ExecuteBeforeRequest(ctx, beforeEvent); err != nil {
			return nil, fmt.Errorf("before-request callback failed: %w", err)
		}
	}

	// Get provider
	p, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found (did you register it?)", providerName)
	}

	// Check cache before API call
	if c.cache != nil {
		// Marshal messages for cache key generation
		messagesJSON, err := json.Marshal(req.Messages)
		if err == nil {
			cacheKey := cache.Key(modelName, messagesJSON, req.Temperature, req.MaxTokens, req.TopP)

			// Try to get from cache
			if cached, err := c.cache.Get(ctx, cacheKey); err == nil {
				var resp CompletionResponse
				if json.Unmarshal(cached, &resp) == nil {
					return &resp, nil
				}
			}
		}
	}

	// Apply timeout if specified
	if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	} else if c.config.DefaultTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	}

	// Create a copy of the request with provider prefix removed
	providerReq := *req
	providerReq.Model = modelName

	// Call provider with retries
	var resp *CompletionResponse
	err = c.withRetry(ctx, func() error {
		var callErr error
		resp, callErr = p.Completion(ctx, &providerReq)
		return callErr
	})

	// Record end time
	endTime := time.Now()
	duration := endTime.Sub(startTime)

	if err != nil {
		// Execute failure callbacks
		if c.callbacks != nil {
			failureEvent := &callback.FailureEvent{
				RequestID: RequestIDFromContext(ctx),
				Model:     modelName,
				Provider:  providerName,
				Request:   req,
				Error:     err,
				StartTime: startTime,
				EndTime:   endTime,
				Duration:  duration,
			}
			c.callbacks.ExecuteFailure(ctx, failureEvent)
		}
		return nil, err
	}

	// Store successful response in cache
	if c.cache != nil && resp != nil {
		// Marshal messages for cache key generation
		if messagesJSON, err := json.Marshal(req.Messages); err == nil {
			cacheKey := cache.Key(modelName, messagesJSON, req.Temperature, req.MaxTokens, req.TopP)

			// Store in cache with 1 hour TTL
			if data, err := json.Marshal(resp); err == nil {
				// Ignore cache errors - don't fail the request if caching fails
				_ = c.cache.Set(ctx, cacheKey, data, 1*time.Hour)
			}
		}
	}

	// Execute success callbacks
	if c.callbacks != nil {
		// Calculate cost if available
		cost := 0.0
		if c.costCalc != nil {
			if calculatedCost, err := c.costCalc.CalculateCompletion(resp); err == nil {
				cost = calculatedCost
			}
		}

		// Get token count
		tokens := 0
		if resp.Usage != nil {
			tokens = resp.Usage.TotalTokens
		}

		successEvent := &callback.SuccessEvent{
			RequestID: RequestIDFromContext(ctx),
			Model:     modelName,
			Provider:  providerName,
			Request:   req,
			Response:  resp,
			StartTime: startTime,
			EndTime:   endTime,
			Duration:  duration,
			Cost:      cost,
			Tokens:    tokens,
		}
		c.callbacks.ExecuteSuccess(ctx, successEvent)
	}

	return resp, nil
}

// CompletionStream creates a streaming chat completion.
//
// The returned Stream must be closed by the caller to release resources.
//
// Example:
//
//	stream, err := client.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "openai/gpt-4",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Write a poem"},
//	    },
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer stream.Close()
//
//	for {
//	    chunk, err := stream.Recv()
//	    if err == io.EOF {
//	        break
//	    }
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    fmt.Print(chunk.Choices[0].Delta.Content)
//	}
func (c *client) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Record start time
	startTime := time.Now()
	ctx = WithStartTime(ctx, startTime)

	// Parse model string
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Add provider and model to context
	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Execute before-request callbacks
	if c.callbacks != nil {
		beforeEvent := &callback.BeforeRequestEvent{
			RequestID: RequestIDFromContext(ctx),
			Model:     modelName,
			Provider:  providerName,
			Request:   req,
			StartTime: startTime,
		}
		if err := c.callbacks.ExecuteBeforeRequest(ctx, beforeEvent); err != nil {
			return nil, fmt.Errorf("before-request callback failed: %w", err)
		}
	}

	// Get provider
	p, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found (did you register it?)", providerName)
	}

	// Apply timeout if specified
	if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	} else if c.config.DefaultTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	}

	// Create a copy of the request with provider prefix removed
	providerReq := *req
	providerReq.Model = modelName

	// Call provider (no retry for streaming)
	stream, err := p.CompletionStream(ctx, &providerReq)
	if err != nil {
		// Execute failure callbacks
		if c.callbacks != nil {
			endTime := time.Now()
			failureEvent := &callback.FailureEvent{
				RequestID: RequestIDFromContext(ctx),
				Model:     modelName,
				Provider:  providerName,
				Request:   req,
				Error:     err,
				StartTime: startTime,
				EndTime:   endTime,
				Duration:  endTime.Sub(startTime),
			}
			c.callbacks.ExecuteFailure(ctx, failureEvent)
		}
		return nil, err
	}

	// Wrap stream with callback execution if callbacks are registered
	if c.callbacks != nil {
		return newCallbackStream(ctx, stream, c.callbacks, req, modelName, providerName, startTime), nil
	}

	return stream, nil
}

// callbackStream wraps a Stream to execute callbacks for each chunk and on completion.
type callbackStream struct {
	underlying Stream
	ctx        context.Context
	callbacks  *callback.Registry
	req        *CompletionRequest
	model      string
	provider   string
	startTime  time.Time
	chunkIndex int
	finalErr   error
	closed     bool
}

// newCallbackStream creates a new callback-aware stream wrapper.
func newCallbackStream(
	ctx context.Context,
	underlying Stream,
	callbacks *callback.Registry,
	req *CompletionRequest,
	model, provider string,
	startTime time.Time,
) Stream {
	return &callbackStream{
		underlying: underlying,
		ctx:        ctx,
		callbacks:  callbacks,
		req:        req,
		model:      model,
		provider:   provider,
		startTime:  startTime,
		chunkIndex: 0,
	}
}

// Recv receives the next chunk and executes stream callbacks.
func (s *callbackStream) Recv() (*CompletionChunk, error) {
	chunk, err := s.underlying.Recv()

	// Handle EOF (stream completed successfully)
	if err == io.EOF {
		s.executeSuccessCallback()
		return nil, io.EOF
	}

	// Handle error
	if err != nil {
		s.finalErr = err
		s.executeFailureCallback(err)
		return nil, err
	}

	// Execute stream callback for this chunk
	if chunk != nil {
		streamEvent := &callback.StreamEvent{
			RequestID: RequestIDFromContext(s.ctx),
			Model:     s.model,
			Provider:  s.provider,
			Chunk:     chunk,
			Index:     s.chunkIndex,
			Timestamp: time.Now(),
		}
		s.callbacks.ExecuteStream(s.ctx, streamEvent)
		s.chunkIndex++
	}

	return chunk, nil
}

// Close closes the underlying stream and executes completion callbacks if needed.
func (s *callbackStream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true

	// Close underlying stream
	closeErr := s.underlying.Close()

	// If we haven't seen EOF or an error yet, this is abnormal termination
	// Don't execute success/failure callbacks as the stream may have been interrupted
	return closeErr
}

// executeSuccessCallback executes success callbacks after stream completion.
func (s *callbackStream) executeSuccessCallback() {
	endTime := time.Now()
	successEvent := &callback.SuccessEvent{
		RequestID: RequestIDFromContext(s.ctx),
		Model:     s.model,
		Provider:  s.provider,
		Request:   s.req,
		Response:  nil, // Streaming doesn't have a single response object
		StartTime: s.startTime,
		EndTime:   endTime,
		Duration:  endTime.Sub(s.startTime),
		Cost:      0, // Cost calculation not available for streaming
		Tokens:    0, // Token count not available until final chunk
	}
	s.callbacks.ExecuteSuccess(s.ctx, successEvent)
}

// executeFailureCallback executes failure callbacks after stream error.
func (s *callbackStream) executeFailureCallback(err error) {
	endTime := time.Now()
	failureEvent := &callback.FailureEvent{
		RequestID: RequestIDFromContext(s.ctx),
		Model:     s.model,
		Provider:  s.provider,
		Request:   s.req,
		Error:     err,
		StartTime: s.startTime,
		EndTime:   endTime,
		Duration:  endTime.Sub(s.startTime),
	}
	s.callbacks.ExecuteFailure(s.ctx, failureEvent)
}
