package vertex

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/blue-context/warp"
)

// vertexRequest represents a Vertex AI generateContent request.
//
// Vertex AI uses a different format than OpenAI:
// - "contents" instead of "messages"
// - "parts" within each content item
// - Different role naming ("user" and "model" instead of "assistant")
// - Generation config as separate object
type vertexRequest struct {
	Contents         []vertexContent         `json:"contents"`
	GenerationConfig *vertexGenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings   []vertexSafetySetting   `json:"safetySettings,omitempty"`
	Tools            []vertexTool            `json:"tools,omitempty"`
}

// vertexContent represents a message in Vertex AI format.
type vertexContent struct {
	Role  string       `json:"role"`
	Parts []vertexPart `json:"parts"`
}

// vertexPart represents a content part (text, image, function call, etc.).
type vertexPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *vertexInlineData       `json:"inlineData,omitempty"`
	FunctionCall     *vertexFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *vertexFunctionResponse `json:"functionResponse,omitempty"`
}

// vertexInlineData represents inline image/file data.
type vertexInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // Base64-encoded
}

// vertexFunctionCall represents a function call from the model.
type vertexFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

// vertexFunctionResponse represents a function call response.
type vertexFunctionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}

// vertexGenerationConfig represents generation parameters.
type vertexGenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
	CandidateCount  *int     `json:"candidateCount,omitempty"`
}

// vertexSafetySetting represents a safety setting.
type vertexSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// vertexTool represents function calling tool.
type vertexTool struct {
	FunctionDeclarations []vertexFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// vertexFunctionDeclaration represents a function declaration.
type vertexFunctionDeclaration struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// vertexResponse represents a Vertex AI generateContent response.
type vertexResponse struct {
	Candidates     []vertexCandidate     `json:"candidates,omitempty"`
	PromptFeedback *vertexPromptFeedback `json:"promptFeedback,omitempty"`
	UsageMetadata  *vertexUsageMetadata  `json:"usageMetadata,omitempty"`
}

// vertexCandidate represents a completion candidate.
type vertexCandidate struct {
	Content       vertexContent        `json:"content"`
	FinishReason  string               `json:"finishReason,omitempty"`
	SafetyRatings []vertexSafetyRating `json:"safetyRatings,omitempty"`
	Index         int                  `json:"index"`
}

// vertexPromptFeedback represents feedback about the prompt.
type vertexPromptFeedback struct {
	BlockReason   string               `json:"blockReason,omitempty"`
	SafetyRatings []vertexSafetyRating `json:"safetyRatings,omitempty"`
}

// vertexSafetyRating represents a safety rating.
type vertexSafetyRating struct {
	Category    string `json:"category"`
	Probability string `json:"probability"`
	Blocked     bool   `json:"blocked,omitempty"`
}

// vertexUsageMetadata represents token usage metadata.
type vertexUsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// transformRequest converts a Warp CompletionRequest to Vertex AI format.
//
// Key transformations:
//   - messages -> contents (with role mapping)
//   - message content -> parts array
//   - "assistant" role -> "model"
//   - "system" messages -> prepended to first user message
//   - temperature, maxTokens, etc. -> generationConfig
//   - tools -> functionDeclarations
func transformRequest(req *warp.CompletionRequest) (*vertexRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if len(req.Messages) == 0 {
		return nil, fmt.Errorf("messages cannot be empty")
	}

	vReq := &vertexRequest{
		Contents: make([]vertexContent, 0, len(req.Messages)),
	}

	// Transform messages to contents
	var systemPrompt string
	for _, msg := range req.Messages {
		// Handle system messages separately
		if msg.Role == "system" {
			// Collect system message content
			content := extractTextContent(msg.Content)
			if systemPrompt != "" {
				systemPrompt += "\n\n"
			}
			systemPrompt += content
			continue
		}

		// Transform message to Vertex content
		content, err := transformMessage(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to transform message: %w", err)
		}
		vReq.Contents = append(vReq.Contents, content)
	}

	// Prepend system prompt to first user message if present
	if systemPrompt != "" && len(vReq.Contents) > 0 {
		// Find first user message
		for i := range vReq.Contents {
			if vReq.Contents[i].Role == "user" {
				// Prepend system prompt to first part
				if len(vReq.Contents[i].Parts) > 0 {
					originalText := vReq.Contents[i].Parts[0].Text
					vReq.Contents[i].Parts[0].Text = systemPrompt + "\n\n" + originalText
				} else {
					vReq.Contents[i].Parts = append([]vertexPart{{Text: systemPrompt}}, vReq.Contents[i].Parts...)
				}
				break
			}
		}
	}

	// Transform generation config
	if req.Temperature != nil || req.MaxTokens != nil || req.TopP != nil || len(req.Stop) > 0 || req.N != nil {
		vReq.GenerationConfig = &vertexGenerationConfig{
			Temperature:     req.Temperature,
			TopP:            req.TopP,
			MaxOutputTokens: req.MaxTokens,
			CandidateCount:  req.N,
		}
		if len(req.Stop) > 0 {
			vReq.GenerationConfig.StopSequences = req.Stop
		}
	}

	// Transform tools
	if len(req.Tools) > 0 {
		vReq.Tools = transformTools(req.Tools)
	}

	return vReq, nil
}

// transformMessage converts a single message to Vertex content.
func transformMessage(msg warp.Message) (vertexContent, error) {
	content := vertexContent{
		Role:  transformRole(msg.Role),
		Parts: make([]vertexPart, 0),
	}

	// Handle content (can be string or multimodal array)
	switch v := msg.Content.(type) {
	case string:
		if v != "" {
			content.Parts = append(content.Parts, vertexPart{Text: v})
		}

	case []interface{}:
		// Multimodal content
		for _, item := range v {
			part, err := transformContentPart(item)
			if err != nil {
				return content, err
			}
			content.Parts = append(content.Parts, part)
		}

	case []warp.ContentPart:
		// Typed multimodal content
		for _, item := range v {
			part, err := transformTypedContentPart(item)
			if err != nil {
				return content, err
			}
			content.Parts = append(content.Parts, part)
		}

	default:
		return content, fmt.Errorf("unsupported content type: %T", v)
	}

	// Handle tool calls
	if len(msg.ToolCalls) > 0 {
		for _, tc := range msg.ToolCalls {
			if tc.Type == "function" {
				// Parse arguments JSON
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					return content, fmt.Errorf("failed to parse function arguments: %w", err)
				}

				content.Parts = append(content.Parts, vertexPart{
					FunctionCall: &vertexFunctionCall{
						Name: tc.Function.Name,
						Args: args,
					},
				})
			}
		}
	}

	// Handle tool responses
	if msg.Role == "tool" && msg.ToolCallID != "" {
		// Parse content as function response
		var response map[string]interface{}
		if err := json.Unmarshal([]byte(extractTextContent(msg.Content)), &response); err != nil {
			// If not JSON, wrap as string response
			response = map[string]interface{}{
				"result": extractTextContent(msg.Content),
			}
		}

		content.Parts = append(content.Parts, vertexPart{
			FunctionResponse: &vertexFunctionResponse{
				Name:     msg.Name, // Function name should be in Name field
				Response: response,
			},
		})
	}

	return content, nil
}

// transformRole converts OpenAI role to Vertex AI role.
func transformRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	case "tool":
		return "function" // Vertex uses "function" for tool responses
	default:
		return role // "user" stays "user"
	}
}

// transformContentPart transforms a content part from interface{}.
func transformContentPart(item interface{}) (vertexPart, error) {
	part := vertexPart{}

	itemMap, ok := item.(map[string]interface{})
	if !ok {
		return part, fmt.Errorf("content part must be object, got %T", item)
	}

	partType, _ := itemMap["type"].(string)
	switch partType {
	case "text":
		text, _ := itemMap["text"].(string)
		part.Text = text

	case "image_url":
		imageURL, ok := itemMap["image_url"].(map[string]interface{})
		if !ok {
			return part, fmt.Errorf("image_url must be object")
		}
		url, _ := imageURL["url"].(string)

		// Extract base64 data from data URI
		mimeType, data, err := parseDataURI(url)
		if err != nil {
			return part, fmt.Errorf("failed to parse image data URI: %w", err)
		}

		part.InlineData = &vertexInlineData{
			MimeType: mimeType,
			Data:     data,
		}

	default:
		return part, fmt.Errorf("unsupported content part type: %s", partType)
	}

	return part, nil
}

// transformTypedContentPart transforms a typed ContentPart.
func transformTypedContentPart(item warp.ContentPart) (vertexPart, error) {
	part := vertexPart{}

	switch item.Type {
	case "text":
		part.Text = item.Text

	case "image_url":
		if item.ImageURL == nil {
			return part, fmt.Errorf("image_url is nil")
		}

		// Extract base64 data from data URI
		mimeType, data, err := parseDataURI(item.ImageURL.URL)
		if err != nil {
			return part, fmt.Errorf("failed to parse image data URI: %w", err)
		}

		part.InlineData = &vertexInlineData{
			MimeType: mimeType,
			Data:     data,
		}

	default:
		return part, fmt.Errorf("unsupported content part type: %s", item.Type)
	}

	return part, nil
}

// transformTools converts Warp tools to Vertex function declarations.
func transformTools(tools []warp.Tool) []vertexTool {
	if len(tools) == 0 {
		return nil
	}

	declarations := make([]vertexFunctionDeclaration, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" {
			declarations = append(declarations, vertexFunctionDeclaration{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			})
		}
	}

	if len(declarations) == 0 {
		return nil
	}

	return []vertexTool{{FunctionDeclarations: declarations}}
}

// transformResponse converts a Vertex AI response to Warp format.
func transformResponse(vResp *vertexResponse, model string) (*warp.CompletionResponse, error) {
	if vResp == nil {
		return nil, fmt.Errorf("vertex response is nil")
	}

	resp := &warp.CompletionResponse{
		ID:      generateResponseID(),
		Object:  "chat.completion",
		Created: currentUnixTime(),
		Model:   model,
		Choices: make([]warp.Choice, 0, len(vResp.Candidates)),
	}

	// Transform candidates to choices
	for _, candidate := range vResp.Candidates {
		choice := warp.Choice{
			Index:        candidate.Index,
			Message:      transformContent(candidate.Content),
			FinishReason: transformFinishReason(candidate.FinishReason),
		}
		resp.Choices = append(resp.Choices, choice)
	}

	// Transform usage metadata
	if vResp.UsageMetadata != nil {
		resp.Usage = &warp.Usage{
			PromptTokens:     vResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: vResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      vResp.UsageMetadata.TotalTokenCount,
		}
	}

	return resp, nil
}

// transformContent converts Vertex content to Warp message.
func transformContent(content vertexContent) warp.Message {
	msg := warp.Message{
		Role: inverseTransformRole(content.Role),
	}

	// Extract text parts
	var textParts []string
	var toolCalls []warp.ToolCall

	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}

		if part.FunctionCall != nil {
			// Convert args to JSON string
			argsJSON, _ := json.Marshal(part.FunctionCall.Args)
			toolCalls = append(toolCalls, warp.ToolCall{
				ID:   generateToolCallID(),
				Type: "function",
				Function: warp.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Set content
	if len(textParts) > 0 {
		msg.Content = joinTextParts(textParts)
	}

	// Set tool calls
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	return msg
}

// inverseTransformRole converts Vertex role back to OpenAI role.
func inverseTransformRole(role string) string {
	switch role {
	case "model":
		return "assistant"
	case "function":
		return "tool"
	default:
		return role
	}
}

// transformFinishReason converts Vertex finish reason to OpenAI format.
func transformFinishReason(reason string) string {
	switch reason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY":
		return "content_filter"
	case "RECITATION":
		return "content_filter"
	default:
		return reason
	}
}

// Helper functions

// extractTextContent extracts text content from message content (handles string or array).
func extractTextContent(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		for _, item := range v {
			if itemMap, ok := item.(map[string]interface{}); ok {
				if itemMap["type"] == "text" {
					if text, ok := itemMap["text"].(string); ok {
						return text
					}
				}
			}
		}
	case []warp.ContentPart:
		for _, part := range v {
			if part.Type == "text" {
				return part.Text
			}
		}
	}
	return ""
}

// parseDataURI parses a data URI and returns mime type and base64 data.
//
// Expected format: data:image/jpeg;base64,/9j/4AAQSkZJRg...
func parseDataURI(uri string) (mimeType string, data string, err error) {
	// Simple data URI parser (minimal implementation)
	if len(uri) < 5 || uri[:5] != "data:" {
		return "", "", fmt.Errorf("invalid data URI: must start with 'data:'")
	}

	// Find semicolon (separates mime type from encoding)
	semicolonIdx := -1
	for i := 5; i < len(uri); i++ {
		if uri[i] == ';' {
			semicolonIdx = i
			break
		}
	}

	if semicolonIdx == -1 {
		return "", "", fmt.Errorf("invalid data URI: missing semicolon")
	}

	mimeType = uri[5:semicolonIdx]

	// Find comma (separates encoding from data)
	commaIdx := -1
	for i := semicolonIdx; i < len(uri); i++ {
		if uri[i] == ',' {
			commaIdx = i
			break
		}
	}

	if commaIdx == -1 {
		return "", "", fmt.Errorf("invalid data URI: missing comma")
	}

	// Extract data (after comma)
	data = uri[commaIdx+1:]

	return mimeType, data, nil
}

// joinTextParts joins multiple text parts with newlines.
func joinTextParts(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 {
		return parts[0]
	}

	result := parts[0]
	for i := 1; i < len(parts); i++ {
		result += "\n" + parts[i]
	}
	return result
}

// generateResponseID generates a unique response ID.
func generateResponseID() string {
	return fmt.Sprintf("chatcmpl-vertex-%d", currentUnixTime())
}

// generateToolCallID generates a unique tool call ID.
func generateToolCallID() string {
	return fmt.Sprintf("call_%d", currentUnixTime())
}

// currentUnixTime returns current Unix timestamp.
func currentUnixTime() int64 {
	return time.Now().Unix()
}
