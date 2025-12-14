package warp

import (
	"context"
	"fmt"
	"time"
)

// Moderation checks content for policy violations.
//
// The input can be a single string or an array of strings.
// The model field defaults to "openai/text-moderation-latest" if not specified.
//
// Example (single text):
//
//	resp, err := client.Moderation(ctx, &warp.ModerationRequest{
//	    Input: "I want to hurt someone",
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	if resp.Results[0].Flagged {
//	    fmt.Println("Content flagged:", resp.Results[0].Categories)
//	}
//
// Example (multiple texts):
//
//	resp, err := client.Moderation(ctx, &warp.ModerationRequest{
//	    Model: "openai/text-moderation-stable",
//	    Input: []string{
//	        "This is fine",
//	        "I want to hurt someone",
//	    },
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	for i, result := range resp.Results {
//	    if result.Flagged {
//	        fmt.Printf("Text %d flagged\n", i)
//	    }
//	}
func (c *client) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	if req.Input == nil {
		return nil, fmt.Errorf("input is required")
	}

	// Default model if not specified
	if req.Model == "" {
		req.Model = "openai/text-moderation-latest"
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

	// Check if provider supports moderation
	// Note: Supports() returns interface{} to avoid import cycle
	// We use type assertion to check the Moderation field
	supportsVal := p.Supports()

	// Try to check if moderation is supported
	// The provider.Capabilities struct has a Moderation field
	supportsModeration := false

	// Use reflection-free struct field access
	if v, ok := supportsVal.(struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
	}); ok {
		supportsModeration = v.Moderation
	}

	if !supportsModeration {
		return nil, fmt.Errorf("provider %q does not support moderation", providerName)
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
	var resp *ModerationResponse
	err = c.withRetry(ctx, func() error {
		var callErr error
		resp, callErr = p.Moderation(ctx, req)
		return callErr
	})

	if err != nil {
		return nil, err
	}

	resp.Provider = providerName
	return resp, nil
}
