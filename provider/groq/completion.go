package groq

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to Groq.
//
// This method handles the complete request/response cycle including:
// - Request transformation to Groq format (OpenAI-compatible)
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "llama3-70b-8192",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    Temperature: warp.Float64Ptr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Transform request to Groq format (OpenAI-compatible)
	groqReq := transformRequest(req)

	// Marshal to JSON
	body, err := json.Marshal(groqReq)
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
		return nil, warp.ParseProviderError("groq", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var resp warp.CompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// transformRequest transforms a Warp request to Groq format.
//
// Groq uses OpenAI-compatible format, so this is essentially the same
// as the OpenAI transformation with minor adjustments.
func transformRequest(req *warp.CompletionRequest) map[string]any {
	groqReq := map[string]any{
		"model":    req.Model,
		"messages": transformMessages(req.Messages),
	}

	// Optional parameters
	if req.Temperature != nil {
		groqReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		groqReq["max_tokens"] = *req.MaxTokens
	}
	if req.TopP != nil {
		groqReq["top_p"] = *req.TopP
	}
	if req.FrequencyPenalty != nil {
		groqReq["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		groqReq["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		groqReq["stop"] = req.Stop
	}
	if req.N != nil {
		groqReq["n"] = *req.N
	}

	// Function calling
	if len(req.Tools) > 0 {
		groqReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		groqReq["tool_choice"] = req.ToolChoice
	}

	// Response format
	if req.ResponseFormat != nil {
		groqReq["response_format"] = req.ResponseFormat
	}

	return groqReq
}

// transformMessages transforms Warp messages to Groq format.
//
// Groq uses OpenAI-compatible message format.
func transformMessages(messages []warp.Message) []map[string]any {
	groqMessages := make([]map[string]any, len(messages))

	for i, msg := range messages {
		groqMsg := map[string]any{
			"role": msg.Role,
		}

		// Handle content (can be string or []ContentPart for multimodal)
		switch content := msg.Content.(type) {
		case string:
			groqMsg["content"] = content
		case []warp.ContentPart:
			// Multimodal content (though Groq doesn't support vision yet)
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
			groqMsg["content"] = parts
		}

		// Optional fields
		if msg.Name != "" {
			groqMsg["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			groqMsg["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallID != "" {
			groqMsg["tool_call_id"] = msg.ToolCallID
		}

		groqMessages[i] = groqMsg
	}

	return groqMessages
}
