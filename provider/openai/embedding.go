package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Embedding sends an embedding request to OpenAI.
//
// This method generates vector embeddings for the provided text input.
// It supports both single text and batch processing.
//
// Example (single text):
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "text-embedding-ada-002",
//	    Input: "Hello, world!",
//	})
//
// Example (batch):
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "text-embedding-ada-002",
//	    Input: []string{"Hello", "World"},
//	})
//
// Example (with dimensions - for text-embedding-3-* models):
//
//	dimensions := 1536
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model:      "text-embedding-3-small",
//	    Input:      "Hello, world!",
//	    Dimensions: &dimensions,
//	})
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	// Transform request to OpenAI format
	openaiReq := map[string]any{
		"model": req.Model,
		"input": req.Input,
	}

	// Optional parameters
	if req.EncodingFormat != "" {
		openaiReq["encoding_format"] = req.EncodingFormat
	}
	if req.Dimensions != nil {
		openaiReq["dimensions"] = *req.Dimensions
	}
	if req.User != "" {
		openaiReq["user"] = req.User
	}

	// Marshal to JSON
	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("openai", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var resp warp.EmbeddingResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}
