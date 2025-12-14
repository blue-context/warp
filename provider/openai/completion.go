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

// Completion sends a chat completion request to OpenAI.
//
// This method handles the complete request/response cycle including:
// - Request transformation to OpenAI format
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "gpt-4",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    Temperature: warp.Float64Ptr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Transform request to OpenAI format
	openaiReq := transformRequest(req)

	// Marshal to JSON
	body, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat/completions", bytes.NewReader(body))
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
	var resp warp.CompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// transformRequest transforms a Warp request to OpenAI format.
//
// This function handles all optional parameters and converts them to the
// format expected by OpenAI's API.
func transformRequest(req *warp.CompletionRequest) map[string]any {
	openaiReq := map[string]any{
		"model":    req.Model,
		"messages": transformMessages(req.Messages),
	}

	// Optional parameters
	if req.Temperature != nil {
		openaiReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		openaiReq["max_tokens"] = *req.MaxTokens
	}
	if req.TopP != nil {
		openaiReq["top_p"] = *req.TopP
	}
	if req.FrequencyPenalty != nil {
		openaiReq["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		openaiReq["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		openaiReq["stop"] = req.Stop
	}
	// Note: Stream is not set here - non-streaming mode
	if req.N != nil {
		openaiReq["n"] = *req.N
	}

	// Function calling
	if len(req.Tools) > 0 {
		openaiReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		openaiReq["tool_choice"] = req.ToolChoice
	}

	// Response format
	if req.ResponseFormat != nil {
		openaiReq["response_format"] = req.ResponseFormat
	}

	return openaiReq
}

// transformMessages transforms Warp messages to OpenAI format.
//
// This function handles both simple text content and multimodal content
// (text + images) according to OpenAI's message format.
func transformMessages(messages []warp.Message) []map[string]any {
	openaiMessages := make([]map[string]any, len(messages))

	for i, msg := range messages {
		openaiMsg := map[string]any{
			"role": msg.Role,
		}

		// Handle content (can be string or []ContentPart for multimodal)
		switch content := msg.Content.(type) {
		case string:
			openaiMsg["content"] = content
		case []warp.ContentPart:
			// Multimodal content (text + images)
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
			openaiMsg["content"] = parts
		}

		// Optional fields
		if msg.Name != "" {
			openaiMsg["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			openaiMsg["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallID != "" {
			openaiMsg["tool_call_id"] = msg.ToolCallID
		}

		openaiMessages[i] = openaiMsg
	}

	return openaiMessages
}
