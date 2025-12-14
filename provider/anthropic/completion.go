package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/blue-context/warp"
)

// anthropicRequest represents an Anthropic API request.
type anthropicRequest struct {
	Model         string               `json:"model"`
	Messages      []anthropicMessage   `json:"messages"`
	System        string               `json:"system,omitempty"`
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature,omitempty"`
	TopP          *float64             `json:"top_p,omitempty"`
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Stream        bool                 `json:"stream,omitempty"`
	Tools         []anthropicTool      `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice,omitempty"`
	Metadata      map[string]any       `json:"metadata,omitempty"`
}

// anthropicMessage represents a message in Anthropic format.
type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // Can be string or []anthropicContentBlock
}

// anthropicContentBlock represents a content block in Anthropic format.
type anthropicContentBlock struct {
	Type   string                `json:"type"`
	Text   string                `json:"text,omitempty"`
	Source *anthropicImageSource `json:"source,omitempty"`
}

// anthropicImageSource represents an image source in Anthropic format.
type anthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// anthropicTool represents a tool in Anthropic format.
type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// anthropicToolChoice represents tool choice in Anthropic format.
type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

// anthropicResponse represents an Anthropic API response.
type anthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Model        string                  `json:"model"`
	Content      []anthropicContentBlock `json:"content"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

// anthropicUsage represents token usage in Anthropic format.
type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Completion sends a chat completion request to Anthropic.
//
// This method handles the complete request/response cycle including:
// - Request transformation to Anthropic format (system message extraction, max_tokens requirement)
// - HTTP request/response handling with correct headers
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "claude-3-opus-20240229",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	    Temperature: warp.Float64Ptr(0.7),
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Transform request to Anthropic format
	anthropicReq, err := transformRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to transform request: %w", err)
	}

	// Marshal to JSON
	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set Anthropic-specific headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.apiVersion)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("anthropic", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var anthropicResp anthropicResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	return transformResponse(&anthropicResp), nil
}

// transformRequest transforms a Warp request to Anthropic format.
//
// Key transformations:
// - Extract system messages to separate "system" parameter
// - Ensure max_tokens is always set (required by Anthropic, default 1024)
// - Convert stop to stop_sequences
// - Handle multimodal content (images)
// - Transform tools to Anthropic format
func transformRequest(req *warp.CompletionRequest) (*anthropicRequest, error) {
	anthropicReq := &anthropicRequest{
		Model: req.Model,
	}

	// Extract system message and filter messages
	var systemMessage string
	messages := make([]anthropicMessage, 0, len(req.Messages))

	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Extract system content
			switch content := msg.Content.(type) {
			case string:
				systemMessage = content
			default:
				return nil, fmt.Errorf("system message must have string content")
			}
		} else {
			// Transform message content
			anthropicMsg := anthropicMessage{
				Role: msg.Role,
			}

			switch content := msg.Content.(type) {
			case string:
				anthropicMsg.Content = content
			case []warp.ContentPart:
				// Multimodal content
				blocks := make([]anthropicContentBlock, len(content))
				for i, part := range content {
					if part.Type == "text" {
						blocks[i] = anthropicContentBlock{
							Type: "text",
							Text: part.Text,
						}
					} else if part.Type == "image_url" && part.ImageURL != nil {
						// Note: Anthropic expects base64-encoded images
						// For now, pass through - client should provide base64 data URIs
						blocks[i] = anthropicContentBlock{
							Type: "image",
							Source: &anthropicImageSource{
								Type:      "base64",
								MediaType: "image/jpeg", // Default, should be inferred from data URI
								Data:      part.ImageURL.URL,
							},
						}
					}
				}
				anthropicMsg.Content = blocks
			}

			messages = append(messages, anthropicMsg)
		}
	}

	anthropicReq.Messages = messages

	// Set system message if present
	if systemMessage != "" {
		anthropicReq.System = systemMessage
	}

	// max_tokens is REQUIRED by Anthropic
	if req.MaxTokens != nil {
		anthropicReq.MaxTokens = *req.MaxTokens
	} else {
		anthropicReq.MaxTokens = 1024 // Default
	}

	// Optional parameters
	if req.Temperature != nil {
		anthropicReq.Temperature = req.Temperature
	}
	if req.TopP != nil {
		anthropicReq.TopP = req.TopP
	}
	if len(req.Stop) > 0 {
		anthropicReq.StopSequences = req.Stop
	}
	// Note: Stream is not set here - handled separately by Completion vs CompletionStream

	// Transform tools
	if len(req.Tools) > 0 {
		anthropicReq.Tools = make([]anthropicTool, len(req.Tools))
		for i, tool := range req.Tools {
			anthropicReq.Tools[i] = anthropicTool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				InputSchema: tool.Function.Parameters,
			}
		}
	}

	// Transform tool choice
	if req.ToolChoice != nil {
		anthropicReq.ToolChoice = &anthropicToolChoice{
			Type: req.ToolChoice.Type,
		}
		if req.ToolChoice.Function != nil {
			anthropicReq.ToolChoice.Name = req.ToolChoice.Function.Name
		}
	}

	return anthropicReq, nil
}

// transformResponse transforms an Anthropic response to Warp format.
//
// This function maps Anthropic's response structure to OpenAI-compatible format:
// - Converts content blocks to message content
// - Maps stop_reason to finish_reason
// - Transforms usage information
func transformResponse(resp *anthropicResponse) *warp.CompletionResponse {
	// Extract text content from content blocks
	var content string
	for _, block := range resp.Content {
		if block.Type == "text" {
			content += block.Text
		}
	}

	// Map stop_reason to OpenAI finish_reason
	finishReason := mapStopReason(resp.StopReason)

	return &warp.CompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    resp.Role,
					Content: content,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &warp.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
}

// mapStopReason maps Anthropic's stop_reason to OpenAI's finish_reason.
func mapStopReason(stopReason string) string {
	switch stopReason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "stop_sequence":
		return "stop"
	default:
		return stopReason
	}
}
