package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Embedding sends an embedding request to Azure OpenAI.
//
// This method generates vector embeddings for the provided text input.
// It supports both single text and batch processing. Azure uses the same
// embedding request/response format as OpenAI.
//
// Note: The deployment used must be an embedding model deployment
// (e.g., text-embedding-ada-002), not a chat model deployment.
//
// Example (single text):
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "text-embedding-ada-002", // Model is ignored; deployment name is used
//	    Input: "Hello, world!",
//	})
//
// Example (batch):
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "text-embedding-ada-002", // Model is ignored; deployment name is used
//	    Input: []string{"Hello", "World"},
//	})
//
// Example (with dimensions - for text-embedding-3-* models):
//
//	dimensions := 1536
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model:      "text-embedding-3-small", // Model is ignored; deployment name is used
//	    Input:      "Hello, world!",
//	    Dimensions: &dimensions,
//	})
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	// Build Azure URL with deployment name and API version
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		p.apiBase, p.deployment, p.apiVersion)

	// Transform request to Azure format (same as OpenAI)
	azureReq := map[string]any{
		"input": req.Input,
	}

	// Optional parameters
	if req.EncodingFormat != "" {
		azureReq["encoding_format"] = req.EncodingFormat
	}
	if req.Dimensions != nil {
		azureReq["dimensions"] = *req.Dimensions
	}
	if req.User != "" {
		azureReq["user"] = req.User
	}

	// Marshal to JSON
	body, err := json.Marshal(azureReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set Azure-specific headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", p.apiKey) // Azure uses api-key, not Bearer token

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("azure", httpResp.StatusCode, body, nil)
	}

	// Parse response (same format as OpenAI)
	var resp warp.EmbeddingResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}
