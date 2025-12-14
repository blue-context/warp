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

// Moderation checks content for policy violations using OpenAI's moderation API.
//
// This method analyzes text for harmful content across 11 categories including
// sexual content, hate speech, harassment, self-harm, and violence.
//
// Example (single text):
//
//	resp, err := provider.Moderation(ctx, &warp.ModerationRequest{
//	    Model: "text-moderation-latest",
//	    Input: "I want to hurt someone",
//	})
//	if err != nil {
//	    return err
//	}
//
//	if resp.Results[0].Flagged {
//	    fmt.Println("Content flagged for:", resp.Results[0].Categories)
//	}
//
// Example (multiple texts):
//
//	resp, err := provider.Moderation(ctx, &warp.ModerationRequest{
//	    Model: "text-moderation-stable",
//	    Input: []string{
//	        "This is fine",
//	        "I want to hurt someone",
//	    },
//	})
//	if err != nil {
//	    return err
//	}
//
//	for i, result := range resp.Results {
//	    if result.Flagged {
//	        fmt.Printf("Text %d flagged\n", i)
//	    }
//	}
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	// Transform request to OpenAI format
	openaiReq := map[string]any{
		"input": req.Input,
	}

	// Add model if specified
	if req.Model != "" {
		openaiReq["model"] = req.Model
	}

	// Marshal to JSON
	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/moderations", bytes.NewReader(body))
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
	var resp warp.ModerationResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}
