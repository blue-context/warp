package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to OpenRouter.
//
// OpenRouter uses an OpenAI-compatible API format, so this implementation
// reuses the same request/response structure with custom headers for
// site identification and rankings.
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "openai/gpt-4o",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    Temperature: warp.Float64Ptr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Validate model name
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	// Transform request to OpenRouter format (OpenAI-compatible)
	openrouterReq := transformRequest(req)

	// Marshal to JSON
	body, err := json.Marshal(openrouterReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request for model %s: %w", req.Model, err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request for model %s: %w", req.Model, err)
	}

	// Set standard headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	// Set optional custom headers for rankings and analytics
	if p.httpReferer != "" {
		httpReq.Header.Set("HTTP-Referer", p.httpReferer)
	}
	if p.appTitle != "" {
		httpReq.Header.Set("X-Title", p.appTitle)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to model %s: %w", req.Model, err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("openrouter", httpResp.StatusCode, body, nil)
	}

	// Parse response (OpenAI-compatible format)
	var resp warp.CompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response from model %s: %w", req.Model, err)
	}

	return &resp, nil
}

// transformRequest transforms a Warp request to OpenRouter format.
//
// Since OpenRouter is OpenAI-compatible, we use the same transformation
// logic as the OpenAI provider. This handles all optional parameters and
// converts them to the format expected by OpenRouter's API.
func transformRequest(req *warp.CompletionRequest) map[string]any {
	openrouterReq := map[string]any{
		"model":    req.Model,
		"messages": transformMessages(req.Messages),
	}

	// Optional parameters
	if req.Temperature != nil {
		openrouterReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		openrouterReq["max_tokens"] = *req.MaxTokens
	}
	if req.TopP != nil {
		openrouterReq["top_p"] = *req.TopP
	}
	if req.FrequencyPenalty != nil {
		openrouterReq["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		openrouterReq["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		openrouterReq["stop"] = req.Stop
	}
	// Note: Stream is not set here - handled separately by Completion vs CompletionStream
	if req.N != nil {
		openrouterReq["n"] = *req.N
	}

	// Function calling
	if len(req.Tools) > 0 {
		openrouterReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		openrouterReq["tool_choice"] = req.ToolChoice
	}

	// Response format (JSON mode)
	if req.ResponseFormat != nil {
		openrouterReq["response_format"] = req.ResponseFormat
	}

	return openrouterReq
}

// transformMessages transforms Warp messages to OpenRouter format.
//
// This function handles both simple text content and multimodal content
// (text + images) according to OpenRouter's OpenAI-compatible message format.
func transformMessages(messages []warp.Message) []map[string]any {
	openrouterMessages := make([]map[string]any, len(messages))

	for i, msg := range messages {
		openrouterMsg := map[string]any{
			"role": msg.Role,
		}

		// Handle content (can be string or []ContentPart for multimodal)
		switch content := msg.Content.(type) {
		case string:
			openrouterMsg["content"] = content
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
			openrouterMsg["content"] = parts
		}

		// Optional fields
		if msg.Name != "" {
			openrouterMsg["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			openrouterMsg["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallID != "" {
			openrouterMsg["tool_call_id"] = msg.ToolCallID
		}

		openrouterMessages[i] = openrouterMsg
	}

	return openrouterMessages
}
