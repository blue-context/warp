package warp

import (
	"context"
	"fmt"
	"time"
)

// Embedding creates embeddings for the given input.
//
// Example:
//
//	resp, err := client.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "openai/text-embedding-ada-002",
//	    Input: "Hello, world!",
//	})
func (c *client) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Add start time to context
	ctx = WithStartTime(ctx, time.Now())

	// Parse model string
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Add provider and model to context
	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	p, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found (did you register it?)", providerName)
	}

	// Update model name in request
	req.Model = modelName

	// Apply timeout
	if c.config.DefaultTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	}

	// Call provider with retries
	var resp *EmbeddingResponse
	err = c.withRetry(ctx, func() error {
		var callErr error
		resp, callErr = p.Embedding(ctx, req)
		return callErr
	})

	if err != nil {
		return nil, err
	}

	return resp, nil
}
