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

// ImageGeneration generates images using DALL-E.
//
// Supports DALL-E 2 and DALL-E 3 models with various sizes, qualities, and styles.
//
// Example:
//
//	resp, err := provider.ImageGeneration(ctx, &warp.ImageGenerationRequest{
//	    Model:  "dall-e-3",
//	    Prompt: "A cute baby sea otter",
//	    Size:   "1024x1024",
//	    Quality: "hd",
//	})
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}

	// Build OpenAI request
	openaiReq := map[string]any{
		"model":  req.Model,
		"prompt": req.Prompt,
	}

	// Add optional parameters
	if req.N != nil {
		openaiReq["n"] = *req.N
	}
	if req.Size != "" {
		openaiReq["size"] = req.Size
	}
	if req.Quality != "" {
		openaiReq["quality"] = req.Quality
	}
	if req.Style != "" {
		openaiReq["style"] = req.Style
	}
	if req.ResponseFormat != "" {
		openaiReq["response_format"] = req.ResponseFormat
	}
	if req.User != "" {
		openaiReq["user"] = req.User
	}

	// Marshal request
	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Determine API key and base URL
	apiKey := p.apiKey
	if req.APIKey != "" {
		apiKey = req.APIKey
	}

	apiBase := p.apiBase
	if req.APIBase != "" {
		apiBase = req.APIBase
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiBase+"/images/generations", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		return nil, warp.ParseProviderError("openai", httpResp.StatusCode, respBody, nil)
	}

	// Parse response
	var resp warp.ImageGenerationResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}
