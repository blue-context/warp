package cohere

import (
	"github.com/blue-context/warp"
)

// cohereRequest represents a Cohere chat request.
//
// Cohere uses a different format from OpenAI:
// - "message" field for the latest user message
// - "chat_history" for previous messages
// - Different parameter names
type cohereRequest struct {
	Message       string          `json:"message"`
	ChatHistory   []cohereMessage `json:"chat_history,omitempty"`
	Model         string          `json:"model,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	MaxTokens     *int            `json:"max_tokens,omitempty"`
	P             *float64        `json:"p,omitempty"` // top_p equivalent
	K             *int            `json:"k,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
}

// cohereMessage represents a message in Cohere's chat history.
type cohereMessage struct {
	Role    string `json:"role"`    // USER, CHATBOT, or SYSTEM
	Message string `json:"message"` // Cohere uses "message" not "content"
}

// cohereResponse represents a Cohere chat response.
type cohereResponse struct {
	ResponseID   string          `json:"response_id"`
	Text         string          `json:"text"`
	GenerationID string          `json:"generation_id"`
	ChatHistory  []cohereMessage `json:"chat_history"`
	FinishReason string          `json:"finish_reason"`
	Meta         cohereMeta      `json:"meta"`
}

// cohereMeta contains metadata about the response.
type cohereMeta struct {
	APIVersion  cohereBilledUnits `json:"api_version"`
	BilledUnits cohereBilledUnits `json:"billed_units"`
	Tokens      coheretokens      `json:"tokens"`
}

// cohereBilledUnits contains token usage information.
type cohereBilledUnits struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// coheretokens contains detailed token counts.
type coheretokens struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// transformToCohereRequest transforms a Warp request to Cohere format.
//
// Cohere's API is different from OpenAI:
// - Last message becomes "message"
// - Previous messages become "chat_history"
// - Role names are uppercase (USER, CHATBOT, SYSTEM)
func transformToCohereRequest(req *warp.CompletionRequest) *cohereRequest {
	cohereReq := &cohereRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		P:           req.TopP,
	}

	if len(req.Stop) > 0 {
		cohereReq.StopSequences = req.Stop
	}

	// Split messages into chat history and current message
	if len(req.Messages) > 0 {
		// Get the last user message as the current message
		lastIdx := len(req.Messages) - 1
		lastMsg := req.Messages[lastIdx]
		cohereReq.Message = extractTextContent(lastMsg.Content)

		// Convert previous messages to chat history
		if lastIdx > 0 {
			cohereReq.ChatHistory = make([]cohereMessage, lastIdx)
			for i := 0; i < lastIdx; i++ {
				cohereReq.ChatHistory[i] = cohereMessage{
					Role:    convertRoleToCohere(req.Messages[i].Role),
					Message: extractTextContent(req.Messages[i].Content),
				}
			}
		}
	}

	return cohereReq
}

// convertRoleToCohere converts OpenAI role names to Cohere format.
//
// Cohere uses uppercase role names:
// - "user" -> "USER"
// - "assistant" -> "CHATBOT"
// - "system" -> "SYSTEM"
func convertRoleToCohere(role string) string {
	switch role {
	case "user":
		return "USER"
	case "assistant":
		return "CHATBOT"
	case "system":
		return "SYSTEM"
	default:
		return "USER"
	}
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

// transformFromCohereResponse transforms a Cohere response to Warp format.
//
// Cohere returns a different structure than OpenAI, so we need to:
// - Map text to message content
// - Convert role names back to OpenAI format
// - Extract token usage from meta
func transformFromCohereResponse(cohereResp *cohereResponse) *warp.CompletionResponse {
	return &warp.CompletionResponse{
		ID:      cohereResp.GenerationID,
		Object:  "chat.completion",
		Created: 0,  // Cohere doesn't provide timestamp
		Model:   "", // Cohere doesn't echo the model in response
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    "assistant",
					Content: cohereResp.Text,
				},
				FinishReason: mapCohereFinishReason(cohereResp.FinishReason),
			},
		},
		Usage: &warp.Usage{
			PromptTokens:     cohereResp.Meta.BilledUnits.InputTokens,
			CompletionTokens: cohereResp.Meta.BilledUnits.OutputTokens,
			TotalTokens:      cohereResp.Meta.BilledUnits.InputTokens + cohereResp.Meta.BilledUnits.OutputTokens,
		},
	}
}

// mapCohereFinishReason maps Cohere finish reasons to OpenAI format.
func mapCohereFinishReason(reason string) string {
	switch reason {
	case "COMPLETE":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "ERROR":
		return "error"
	default:
		return "stop"
	}
}
