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

// Completion sends a chat completion request to Azure OpenAI.
//
// This method handles the complete request/response cycle including:
// - Azure-specific URL construction with deployment name and API version
// - Request transformation to OpenAI format (Azure uses same format)
// - HTTP request/response handling with Azure authentication
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "gpt-4", // Model is ignored; deployment name is used instead
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    Temperature: warp.Float64Ptr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Build Azure URL with deployment name and API version
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.apiBase, p.deployment, p.apiVersion)

	// Transform request to Azure format (same as OpenAI)
	azureReq := transformRequest(req)

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
	var resp warp.CompletionResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// transformRequest transforms a Warp request to Azure OpenAI format.
//
// Azure OpenAI uses the same request format as OpenAI, so we reuse the
// OpenAI transformation logic. The model field is excluded from the request
// since Azure uses deployment names instead.
func transformRequest(req *warp.CompletionRequest) map[string]any {
	azureReq := map[string]any{
		"messages": transformMessages(req.Messages),
	}

	// Optional parameters (same as OpenAI)
	if req.Temperature != nil {
		azureReq["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		azureReq["max_tokens"] = *req.MaxTokens
	}
	if req.TopP != nil {
		azureReq["top_p"] = *req.TopP
	}
	if req.FrequencyPenalty != nil {
		azureReq["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		azureReq["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		azureReq["stop"] = req.Stop
	}
	// Note: Stream is not set here - handled separately by Completion vs CompletionStream
	if req.N != nil {
		azureReq["n"] = *req.N
	}

	// Function calling
	if len(req.Tools) > 0 {
		azureReq["tools"] = req.Tools
	}
	if req.ToolChoice != nil {
		azureReq["tool_choice"] = req.ToolChoice
	}

	// Response format
	if req.ResponseFormat != nil {
		azureReq["response_format"] = req.ResponseFormat
	}

	return azureReq
}

// transformMessages transforms Warp messages to Azure OpenAI format.
//
// This function handles both simple text content and multimodal content
// (text + images) according to OpenAI's message format, which Azure also uses.
func transformMessages(messages []warp.Message) []map[string]any {
	azureMessages := make([]map[string]any, len(messages))

	for i, msg := range messages {
		azureMsg := map[string]any{
			"role": msg.Role,
		}

		// Handle content (can be string or []ContentPart for multimodal)
		switch content := msg.Content.(type) {
		case string:
			azureMsg["content"] = content
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
			azureMsg["content"] = parts
		}

		// Optional fields
		if msg.Name != "" {
			azureMsg["name"] = msg.Name
		}
		if len(msg.ToolCalls) > 0 {
			azureMsg["tool_calls"] = msg.ToolCalls
		}
		if msg.ToolCallID != "" {
			azureMsg["tool_call_id"] = msg.ToolCallID
		}

		azureMessages[i] = azureMsg
	}

	return azureMessages
}
