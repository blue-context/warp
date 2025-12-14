package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Rerank ranks documents using Cohere's rerank API.
//
// Cohere's rerank models are optimized for ranking documents by relevance
// to a search query. This is particularly useful in RAG (Retrieval-Augmented
// Generation) pipelines where you want to rerank retrieved documents before
// feeding them to an LLM.
//
// Supported models:
//   - rerank-english-v3.0: Optimized for English text
//   - rerank-multilingual-v3.0: Supports multiple languages
//
// Example:
//
//	resp, err := provider.Rerank(ctx, &warp.RerankRequest{
//	    Model: "rerank-english-v3.0",
//	    Query: "What is the capital of France?",
//	    Documents: []string{
//	        "Paris is the capital of France",
//	        "London is the capital of England",
//	        "Berlin is the capital of Germany",
//	    },
//	    TopN: warp.IntPtr(2),
//	    ReturnDocuments: warp.BoolPtr(true),
//	})
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	// Build Cohere request
	cohereReq := map[string]any{
		"model":     req.Model,
		"query":     req.Query,
		"documents": req.Documents,
	}

	// Add optional parameters
	if req.TopN != nil {
		cohereReq["top_n"] = *req.TopN
	}
	if req.ReturnDocuments != nil {
		cohereReq["return_documents"] = *req.ReturnDocuments
	}
	if req.MaxChunksPerDoc != nil {
		cohereReq["max_chunks_per_doc"] = *req.MaxChunksPerDoc
	}

	// Marshal request
	body, err := json.Marshal(cohereReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/rerank", bytes.NewReader(body))
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
		bodyBytes, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("cohere", httpResp.StatusCode, bodyBytes, nil)
	}

	// Parse Cohere response
	var cohereResp struct {
		ID      string `json:"id"`
		Results []struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
			Document       *struct {
				Text string `json:"text"`
			} `json:"document,omitempty"`
		} `json:"results"`
		Meta *warp.RerankMeta `json:"meta,omitempty"`
	}

	if err := json.NewDecoder(httpResp.Body).Decode(&cohereResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	resp := &warp.RerankResponse{
		ID:      cohereResp.ID,
		Results: make([]warp.RerankResult, len(cohereResp.Results)),
		Meta:    cohereResp.Meta,
	}

	for i, r := range cohereResp.Results {
		result := warp.RerankResult{
			Index:          r.Index,
			RelevanceScore: r.RelevanceScore,
		}
		if r.Document != nil {
			result.Document = r.Document.Text
		}
		resp.Results[i] = result
	}

	return resp, nil
}
