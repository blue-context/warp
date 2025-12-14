package ollama

import (
	"github.com/blue-context/warp"
	"github.com/blue-context/warp/token"
)

// ollamaRequest represents an Ollama chat request.
//
// Ollama uses a simpler format than OpenAI.
type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream,omitempty"`
	Options  *ollamaOptions  `json:"options,omitempty"`
}

// ollamaMessage represents a single message in Ollama format.
type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ollamaOptions contains generation options for Ollama.
type ollamaOptions struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"` // max_tokens equivalent
	TopP             *float64 `json:"top_p,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// ollamaResponse represents an Ollama chat response.
type ollamaResponse struct {
	Model     string        `json:"model"`
	CreatedAt string        `json:"created_at"`
	Message   ollamaMessage `json:"message"`
	Done      bool          `json:"done"`
}

// transformToOllamaRequest transforms a Warp request to Ollama format.
//
// Ollama has a simpler API than OpenAI, so we need to transform:
// - Messages: Extract text content only
// - Options: Map to Ollama's options format
// - Stream: Passed explicitly by caller (Completion vs CompletionStream)
func transformToOllamaRequest(req *warp.CompletionRequest, stream bool) *ollamaRequest {
	ollamaReq := &ollamaRequest{
		Model:    req.Model,
		Messages: make([]ollamaMessage, len(req.Messages)),
		Stream:   stream,
	}

	// Transform messages
	for i, msg := range req.Messages {
		ollamaReq.Messages[i] = ollamaMessage{
			Role:    msg.Role,
			Content: extractTextContent(msg.Content),
		}
	}

	// Transform options
	if req.Temperature != nil || req.MaxTokens != nil || req.TopP != nil ||
		req.FrequencyPenalty != nil || req.PresencePenalty != nil || len(req.Stop) > 0 {
		ollamaReq.Options = &ollamaOptions{
			Temperature:      req.Temperature,
			NumPredict:       req.MaxTokens,
			TopP:             req.TopP,
			FrequencyPenalty: req.FrequencyPenalty,
			PresencePenalty:  req.PresencePenalty,
			Stop:             req.Stop,
		}
	}

	return ollamaReq
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

// transformFromOllamaResponse transforms an Ollama response to Warp format.
//
// Ollama returns a simpler response than OpenAI, so we need to:
// - Create a CompletionResponse structure
// - Map Ollama fields to Warp fields
// - Estimate token usage using token counter (Ollama doesn't provide actual counts)
func transformFromOllamaResponse(ollamaResp *ollamaResponse, req *warp.CompletionRequest) *warp.CompletionResponse {
	// Estimate token counts using token counter
	// Ollama doesn't provide actual token counts, so we approximate
	counter := token.NewCounter()

	// Estimate prompt tokens from request messages
	promptTokens := 0
	if req != nil {
		promptTokens = counter.CountMessages(req.Messages)
	}

	// Estimate completion tokens from response content
	completionTokens := counter.CountText(ollamaResp.Message.Content)

	totalTokens := promptTokens + completionTokens

	return &warp.CompletionResponse{
		ID:      "ollama-" + ollamaResp.CreatedAt, // Ollama doesn't provide ID
		Object:  "chat.completion",
		Created: 0, // Ollama doesn't provide Unix timestamp
		Model:   ollamaResp.Model,
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    ollamaResp.Message.Role,
					Content: ollamaResp.Message.Content,
				},
				FinishReason: "stop",
			},
		},
		Usage: &warp.Usage{
			// Ollama doesn't provide token counts, so we estimate them using approximation
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      totalTokens,
		},
	}
}
