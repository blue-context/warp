package vllmsemanticrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to vLLM Semantic Router.
//
// This method handles the complete request/response cycle including:
// - Request transformation to OpenAI-compatible format
// - Support for special "auto" model for semantic routing
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// The "auto" model parameter enables intelligent semantic routing to
// automatically select the best backend model for the request.
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "auto", // Semantic routing
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Explain quantum computing"},
//	    },
//	    Temperature: warp.Float64Ptr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}

	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	// Transform request to OpenAI-compatible format
	routerReq := transformRequest(req)

	// Marshal to JSON
	body, err := json.Marshal(routerReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
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
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("vllm-semantic-router", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var resp warp.CompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// transformRequest transforms a Warp request to OpenAI-compatible format.
//
// vLLM Semantic Router uses OpenAI-compatible format, with special support
// for the "auto" model parameter which enables semantic routing.
func transformRequest(req *warp.CompletionRequest) map[string]any {
	routerReq := map[string]any{
		"model":    req.Model,
		"messages": transformMessages(req.Messages),
	}

	// Optional parameters
	if req.Temperature != nil {
		routerReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		routerReq["max_tokens"] = *req.MaxTokens
	}
	if req.TopP != nil {
		routerReq["top_p"] = *req.TopP
	}
	if req.FrequencyPenalty != nil {
		routerReq["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		routerReq["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		routerReq["stop"] = req.Stop
	}
	if req.N != nil {
		routerReq["n"] = *req.N
	}

	// Function calling
	if len(req.Tools) > 0 {
		routerReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		routerReq["tool_choice"] = req.ToolChoice
	}

	// Response format
	if req.ResponseFormat != nil {
		routerReq["response_format"] = req.ResponseFormat
	}

	return routerReq
}

// transformMessages transforms Warp messages to OpenAI-compatible format.
//
// This function handles all message types including text and multimodal content.
func transformMessages(messages []warp.Message) []map[string]any {
	routerMessages := make([]map[string]any, len(messages))

	for i, msg := range messages {
		routerMsg := map[string]any{
			"role": msg.Role,
		}

		// Handle content (can be string or []ContentPart for multimodal)
		switch content := msg.Content.(type) {
		case string:
			routerMsg["content"] = content
		case []warp.ContentPart:
			// Multimodal content
			parts := make([]map[string]any, len(content))
			for j, part := range content {
				parts[j] = map[string]any{
					"type": part.Type,
				}
				if part.Text != "" {
					parts[j]["text"] = part.Text
				}
				if part.ImageURL != nil {
					parts[j]["image_url"] = part.ImageURL
				}
			}
			routerMsg["content"] = parts
		}

		// Optional fields
		if msg.Name != "" {
			routerMsg["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			routerMsg["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallID != "" {
			routerMsg["tool_call_id"] = msg.ToolCallID
		}

		routerMessages[i] = routerMsg
	}

	return routerMessages
}
