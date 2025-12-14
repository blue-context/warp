package together

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to Together AI.
//
// This method handles the complete request/response cycle including:
// - Request transformation to Together AI format (OpenAI-compatible)
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "meta-llama/Llama-3-70b-chat-hf",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    Temperature: floatPtr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Transform request to Together AI format (OpenAI-compatible)
	togetherReq := transformRequest(req)

	// Marshal to JSON
	body, err := json.Marshal(togetherReq)
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
		return nil, warp.ParseProviderError("together", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var resp warp.CompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// transformRequest transforms a Warp request to Together AI format.
//
// Together AI uses OpenAI-compatible format, so this is essentially the same
// as the OpenAI transformation.
func transformRequest(req *warp.CompletionRequest) map[string]any {
	togetherReq := map[string]any{
		"model":    req.Model,
		"messages": transformMessages(req.Messages),
	}

	// Optional parameters
	if req.Temperature != nil {
		togetherReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		togetherReq["max_tokens"] = *req.MaxTokens
	}
	if req.TopP != nil {
		togetherReq["top_p"] = *req.TopP
	}
	if req.FrequencyPenalty != nil {
		togetherReq["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		togetherReq["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		togetherReq["stop"] = req.Stop
	}
	// Note: Stream is not set here - handled separately by Completion vs CompletionStream
	if req.N != nil {
		togetherReq["n"] = *req.N
	}

	// Function calling
	if len(req.Tools) > 0 {
		togetherReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		togetherReq["tool_choice"] = req.ToolChoice
	}

	// Response format
	if req.ResponseFormat != nil {
		togetherReq["response_format"] = req.ResponseFormat
	}

	return togetherReq
}

// transformMessages transforms Warp messages to Together AI format.
//
// Together AI uses OpenAI-compatible message format.
func transformMessages(messages []warp.Message) []map[string]any {
	togetherMessages := make([]map[string]any, len(messages))

	for i, msg := range messages {
		togetherMsg := map[string]any{
			"role": msg.Role,
		}

		// Handle content (can be string or []ContentPart for multimodal)
		switch content := msg.Content.(type) {
		case string:
			togetherMsg["content"] = content
		case []warp.ContentPart:
			// Multimodal content (though Together AI doesn't support vision yet)
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
			togetherMsg["content"] = parts
		}

		// Optional fields
		if msg.Name != "" {
			togetherMsg["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			togetherMsg["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallID != "" {
			togetherMsg["tool_call_id"] = msg.ToolCallID
		}

		togetherMessages[i] = togetherMsg
	}

	return togetherMessages
}
