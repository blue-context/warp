package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to Ollama.
//
// This method handles the complete request/response cycle including:
// - Request transformation to Ollama format
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "llama3",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Transform request to Ollama format (non-streaming)
	ollamaReq := transformToOllamaRequest(req, false)

	// Marshal to JSON
	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers (no auth needed for Ollama)
	httpReq.Header.Set("Content-Type", "application/json")

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("ollama", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var ollamaResp ollamaResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	resp := transformFromOllamaResponse(&ollamaResp, req)

	return resp, nil
}

// Embedding is not supported by this Ollama provider implementation.
//
// Ollama has a separate /api/embeddings endpoint with different request/response format.
// Use the Ollama embeddings API directly if you need embedding support.
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	return nil, &warp.WarpError{
		Message:  "embeddings are not supported by this Ollama provider implementation",
		Provider: "ollama",
	}
}
