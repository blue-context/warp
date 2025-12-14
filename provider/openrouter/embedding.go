package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Embedding sends an embedding request to OpenRouter.
//
// OpenRouter supports multiple embedding models from OpenAI and open source
// providers like sentence-transformers. The API format is OpenAI-compatible.
//
// This method generates vector embeddings for the provided text input.
// It supports both single text and batch processing.
//
// Example (single text):
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "openai/text-embedding-ada-002",
//	    Input: "Hello, world!",
//	})
//
// Example (batch):
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "sentence-transformers/all-mpnet-base-v2",
//	    Input: []string{"Hello", "World"},
//	})
//
// Example (with dimensions - for OpenAI text-embedding-3-* models):
//
//	dimensions := 1536
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model:      "openai/text-embedding-3-small",
//	    Input:      "Hello, world!",
//	    Dimensions: &dimensions,
//	})
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	// Validate model name
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	// Transform request to OpenRouter format (OpenAI-compatible)
	openrouterReq := map[string]any{
		"model": req.Model,
		"input": req.Input,
	}

	// Optional parameters
	if req.EncodingFormat != "" {
		openrouterReq["encoding_format"] = req.EncodingFormat
	}
	if req.Dimensions != nil {
		openrouterReq["dimensions"] = *req.Dimensions
	}
	if req.User != "" {
		openrouterReq["user"] = req.User
	}

	// Marshal to JSON
	body, err := json.Marshal(openrouterReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request for model %s: %w", req.Model, err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding request for model %s: %w", req.Model, err)
	}

	// Set standard headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	// Set optional custom headers for rankings and analytics
	if p.httpReferer != "" {
		httpReq.Header.Set("HTTP-Referer", p.httpReferer)
	}
	if p.appTitle != "" {
		httpReq.Header.Set("X-Title", p.appTitle)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send embedding request to model %s: %w", req.Model, err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("openrouter", httpResp.StatusCode, body, nil)
	}

	// Parse response (OpenAI-compatible format)
	var resp warp.EmbeddingResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode embedding response from model %s: %w", req.Model, err)
	}

	return &resp, nil
}
