package vllm

import (
	"github.com/blue-context/warp"
	"github.com/blue-context/warp/token"
)

// vllmRequest represents a vLLM native generate request.
//
// vLLM uses the /inference/v1/generate endpoint with a format similar to OpenAI
// but with vLLM-specific parameters.
type vllmRequest struct {
	Model              string   `json:"model"`
	Prompt             string   `json:"prompt"`
	MaxTokens          *int     `json:"max_tokens,omitempty"`
	Temperature        *float64 `json:"temperature,omitempty"`
	TopP               *float64 `json:"top_p,omitempty"`
	N                  *int     `json:"n,omitempty"`
	BestOf             *int     `json:"best_of,omitempty"`
	PresencePenalty    *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty   *float64 `json:"frequency_penalty,omitempty"`
	RepetitionPenalty  *float64 `json:"repetition_penalty,omitempty"`
	Stop               []string `json:"stop,omitempty"`
	Stream             bool     `json:"stream,omitempty"`
	Logprobs           *int     `json:"logprobs,omitempty"`
	ResponseFormat     *string  `json:"response_format,omitempty"` // For JSON mode
}

// vllmResponse represents a vLLM native generate response.
//
// vLLM returns an OpenAI-compatible response format even from the native endpoint.
type vllmResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []vllmChoice   `json:"choices"`
	Usage   *vllmUsage     `json:"usage,omitempty"`
}

// vllmChoice represents a single completion choice in vLLM response.
type vllmChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	Logprobs     any     `json:"logprobs,omitempty"`
	FinishReason string  `json:"finish_reason"`
}

// vllmUsage represents token usage in vLLM response.
type vllmUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// transformToVLLMRequest transforms a Warp request to vLLM native format.
//
// vLLM's native endpoint uses a prompt-based format rather than messages.
// We convert the messages to a prompt by concatenating them with role prefixes.
// The stream parameter is passed explicitly by caller (Completion vs CompletionStream).
func transformToVLLMRequest(req *warp.CompletionRequest, stream bool) *vllmRequest {
	vllmReq := &vllmRequest{
		Model:             req.Model,
		Prompt:            messagesToPrompt(req.Messages),
		Stream:            stream,
		MaxTokens:         req.MaxTokens,
		Temperature:       req.Temperature,
		TopP:              req.TopP,
		PresencePenalty:   req.PresencePenalty,
		FrequencyPenalty:  req.FrequencyPenalty,
		Stop:              req.Stop,
	}

	// Set N parameter (number of completions)
	if req.N != nil && *req.N > 1 {
		vllmReq.N = req.N
	}

	// Handle JSON mode via response_format
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		jsonFormat := "json"
		vllmReq.ResponseFormat = &jsonFormat
	}

	return vllmReq
}

// messagesToPrompt converts a message array to a single prompt string.
//
// vLLM's native endpoint expects a prompt rather than messages.
// We format messages with role prefixes to maintain conversation structure.
func messagesToPrompt(messages []warp.Message) string {
	if len(messages) == 0 {
		return ""
	}

	var prompt string
	for i, msg := range messages {
		content := extractTextContent(msg.Content)

		// Format with role prefix
		switch msg.Role {
		case "system":
			prompt += "System: " + content
		case "user":
			prompt += "User: " + content
		case "assistant":
			prompt += "Assistant: " + content
		default:
			prompt += content
		}

		// Add newline between messages (except after last message)
		if i < len(messages)-1 {
			prompt += "\n\n"
		}
	}

	// Add "Assistant: " prefix to prompt response continuation
	if len(messages) > 0 && messages[len(messages)-1].Role == "user" {
		prompt += "\n\nAssistant:"
	}

	return prompt
}

// extractTextContent extracts text content from a message.
//
// Handles both string content and multimodal content (extracts text only).
func extractTextContent(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []warp.ContentPart:
		// For multimodal content, concatenate text parts
		var text string
		for _, part := range c {
			if part.Type == "text" {
				text += part.Text
			}
		}
		return text
	default:
		return ""
	}
}

// transformFromVLLMResponse transforms a vLLM response to Warp format.
//
// vLLM returns an OpenAI-compatible response, but we need to convert
// the text completion format to chat completion format.
func transformFromVLLMResponse(vllmResp *vllmResponse) *warp.CompletionResponse {
	// Transform choices
	choices := make([]warp.Choice, len(vllmResp.Choices))
	for i, vllmChoice := range vllmResp.Choices {
		choices[i] = warp.Choice{
			Index: vllmChoice.Index,
			Message: warp.Message{
				Role:    "assistant",
				Content: vllmChoice.Text,
			},
			FinishReason: vllmChoice.FinishReason,
		}
	}

	// Transform usage
	var usage *warp.Usage
	if vllmResp.Usage != nil {
		usage = &warp.Usage{
			PromptTokens:     vllmResp.Usage.PromptTokens,
			CompletionTokens: vllmResp.Usage.CompletionTokens,
			TotalTokens:      vllmResp.Usage.TotalTokens,
		}
	}

	return &warp.CompletionResponse{
		ID:      vllmResp.ID,
		Object:  "chat.completion",
		Created: vllmResp.Created,
		Model:   vllmResp.Model,
		Choices: choices,
		Usage:   usage,
	}
}

// estimateTokenUsage estimates token usage when vLLM doesn't provide it.
//
// vLLM should provide token counts, but this is a fallback for cases
// where usage information is missing.
func estimateTokenUsage(req *warp.CompletionRequest, responseText string) *warp.Usage {
	counter := token.NewCounter()

	// Estimate prompt tokens from request messages
	promptTokens := 0
	if req != nil {
		promptTokens = counter.CountMessages(req.Messages)
	}

	// Estimate completion tokens from response content
	completionTokens := counter.CountText(responseText)

	totalTokens := promptTokens + completionTokens

	return &warp.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}
}
