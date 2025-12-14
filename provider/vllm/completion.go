package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/token"
)

// Completion sends a chat completion request to vLLM.
//
// This method handles the complete request/response cycle including:
// - Request transformation to vLLM native format
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// Uses vLLM's native /inference/v1/generate endpoint.
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "meta-llama/Llama-2-7b-hf",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Check context cancellation before starting
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Transform request to vLLM format (non-streaming)
	vllmReq := transformToVLLMRequest(req, false)

	// Marshal to JSON
	body, err := json.Marshal(vllmReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request to vLLM's native generate endpoint
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/inference/v1/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")

	// Add optional API key if configured (for custom deployments with auth)
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(httpResp.Body)
		if err != nil {
			body = []byte("failed to read error response")
		}
		return nil, warp.ParseProviderError("vllm", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var vllmResp vllmResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&vllmResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	resp := transformFromVLLMResponse(&vllmResp)

	// If usage information is missing, estimate it
	if resp.Usage == nil && len(resp.Choices) > 0 {
		resp.Usage = estimateTokenUsage(req, resp.Choices[0].Message.Content.(string))
	}

	return resp, nil
}

// Embedding sends an embedding request to vLLM.
//
// Uses vLLM's /pooling endpoint for generating embeddings.
//
// Note: This requires vLLM to be running with a pooling-compatible model
// (e.g., intfloat/e5-small, sentence-transformers models).
//
// Example:
//
//	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
//	    Model: "intfloat/e5-small",
//	    Input: "Hello, world!",
//	})
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	// Check context cancellation before starting
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Transform input to array format
	var inputs []string
	switch input := req.Input.(type) {
	case string:
		inputs = []string{input}
	case []string:
		inputs = input
	default:
		return nil, fmt.Errorf("invalid input type: expected string or []string")
	}

	// Create vLLM pooling request
	poolingReq := map[string]any{
		"model": req.Model,
		"input": inputs,
		"task":  "embed",
	}

	// Marshal to JSON
	body, err := json.Marshal(poolingReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request to vLLM's pooling endpoint
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/pooling", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")

	// Add optional API key if configured
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(httpResp.Body)
		if err != nil {
			body = []byte("failed to read error response")
		}
		return nil, warp.ParseProviderError("vllm", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var poolingResp struct {
		Data []struct {
			Data []float64 `json:"data"`
		} `json:"data"`
	}
	if err := json.NewDecoder(httpResp.Body).Decode(&poolingResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	embeddings := make([]warp.Embedding, len(poolingResp.Data))
	for i, data := range poolingResp.Data {
		embeddings[i] = warp.Embedding{
			Object:    "embedding",
			Embedding: data.Data,
			Index:     i,
		}
	}

	// Estimate token usage (vLLM pooling doesn't provide token counts)
	counter := token.NewCounter()
	promptTokens := 0
	for _, input := range inputs {
		promptTokens += counter.CountText(input)
	}

	return &warp.EmbeddingResponse{
		Object: "list",
		Data:   embeddings,
		Model:  req.Model,
		Usage: &warp.EmbeddingUsage{
			PromptTokens: promptTokens,
			TotalTokens:  promptTokens,
		},
	}, nil
}

// Rerank ranks documents by relevance to a query.
//
// Uses vLLM's /rerank endpoint for document reranking.
//
// Note: This requires vLLM to be running with a reranking-compatible model.
//
// Example:
//
//	resp, err := provider.Rerank(ctx, &warp.RerankRequest{
//	    Model: "BAAI/bge-reranker-large",
//	    Query: "What is the capital of France?",
//	    Documents: []string{
//	        "Paris is the capital of France",
//	        "London is the capital of England",
//	    },
//	    TopN: warp.IntPtr(1),
//	})
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	// Check context cancellation before starting
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Create vLLM rerank request
	rerankReq := map[string]any{
		"model":     req.Model,
		"query":     req.Query,
		"documents": req.Documents,
	}

	if req.TopN != nil {
		rerankReq["top_n"] = *req.TopN
	}

	if req.ReturnDocuments != nil {
		rerankReq["return_documents"] = *req.ReturnDocuments
	}

	// Marshal to JSON
	body, err := json.Marshal(rerankReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request to vLLM's rerank endpoint
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")

	// Add optional API key if configured
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(httpResp.Body)
		if err != nil {
			body = []byte("failed to read error response")
		}
		return nil, warp.ParseProviderError("vllm", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var rerankResp struct {
		Results []struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
			Document       string  `json:"document,omitempty"`
		} `json:"results"`
	}
	if err := json.NewDecoder(httpResp.Body).Decode(&rerankResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	results := make([]warp.RerankResult, len(rerankResp.Results))
	for i, result := range rerankResp.Results {
		results[i] = warp.RerankResult{
			Index:          result.Index,
			RelevanceScore: result.RelevanceScore,
			Document:       result.Document,
		}
	}

	return &warp.RerankResponse{
		ID:       "vllm-rerank",
		Results:  results,
		Provider: "vllm",
		Model:    req.Model,
	}, nil
}
