package bedrock

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/blue-context/warp"
)

// transformClaudeRequest transforms a Warp request to Bedrock Claude format.
//
// Bedrock Claude API uses a format similar to Anthropic's API but with some differences:
//   - Messages array for conversation
//   - System message separate from messages array
//   - Tools array for function calling
//   - Anthropic API version header
//
// Example Bedrock Claude request:
//
//	{
//	  "anthropic_version": "bedrock-2023-05-31",
//	  "messages": [{"role": "user", "content": "Hello"}],
//	  "max_tokens": 1024,
//	  "temperature": 0.7
//	}
func transformClaudeRequest(req *warp.CompletionRequest) map[string]interface{} {
	bedrockReq := map[string]interface{}{
		"anthropic_version": "bedrock-2023-05-31",
	}

	// Extract system message and regular messages
	var systemMessage string
	var messages []map[string]interface{}

	for _, msg := range req.Messages {
		if msg.Role == "system" {
			// Extract system message content
			if content, ok := msg.Content.(string); ok {
				systemMessage = content
			}
			continue
		}

		// Transform message
		bedrockMsg := map[string]interface{}{
			"role": msg.Role,
		}

		// Handle content (can be string or multimodal)
		switch content := msg.Content.(type) {
		case string:
			bedrockMsg["content"] = content
		case []interface{}:
			// Multimodal content (text + images)
			bedrockMsg["content"] = content
		default:
			// Fallback: convert to string
			bedrockMsg["content"] = fmt.Sprintf("%v", content)
		}

		messages = append(messages, bedrockMsg)
	}

	bedrockReq["messages"] = messages

	// Add system message if present
	if systemMessage != "" {
		bedrockReq["system"] = systemMessage
	}

	// Add max_tokens (required for Claude)
	maxTokens := 1024 // Default
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	bedrockReq["max_tokens"] = maxTokens

	// Add optional parameters
	if req.Temperature != nil {
		bedrockReq["temperature"] = *req.Temperature
	}

	if req.TopP != nil {
		bedrockReq["top_p"] = *req.TopP
	}

	if len(req.Stop) > 0 {
		bedrockReq["stop_sequences"] = req.Stop
	}

	// Add tools if present
	if len(req.Tools) > 0 {
		tools := make([]map[string]interface{}, 0, len(req.Tools))
		for _, tool := range req.Tools {
			tools = append(tools, map[string]interface{}{
				"name":         tool.Function.Name,
				"description":  tool.Function.Description,
				"input_schema": tool.Function.Parameters,
			})
		}
		bedrockReq["tools"] = tools
	}

	return bedrockReq
}

// transformLlamaRequest transforms a Warp request to Bedrock Llama format.
//
// Bedrock Llama API uses a different format from Claude:
//   - Single "prompt" string instead of messages array
//   - Different parameter names
//
// Example Bedrock Llama request:
//
//	{
//	  "prompt": "<s>[INST] Hello [/INST]",
//	  "max_gen_len": 512,
//	  "temperature": 0.7
//	}
func transformLlamaRequest(req *warp.CompletionRequest) map[string]interface{} {
	// Convert messages to Llama prompt format
	prompt := convertMessagesToLlamaPrompt(req.Messages)

	bedrockReq := map[string]interface{}{
		"prompt": prompt,
	}

	// Add max_gen_len (max tokens for Llama)
	maxTokens := 512 // Default
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	bedrockReq["max_gen_len"] = maxTokens

	// Add optional parameters
	if req.Temperature != nil {
		bedrockReq["temperature"] = *req.Temperature
	}

	if req.TopP != nil {
		bedrockReq["top_p"] = *req.TopP
	}

	return bedrockReq
}

// transformTitanRequest transforms a Warp request to Bedrock Titan format.
//
// Bedrock Titan API uses yet another format:
//   - inputText field for the prompt
//   - textGenerationConfig for parameters
//
// Example Bedrock Titan request:
//
//	{
//	  "inputText": "Hello, world!",
//	  "textGenerationConfig": {
//	    "maxTokenCount": 512,
//	    "temperature": 0.7
//	  }
//	}
func transformTitanRequest(req *warp.CompletionRequest) map[string]interface{} {
	// Convert messages to single prompt text
	prompt := convertMessagesToText(req.Messages)

	bedrockReq := map[string]interface{}{
		"inputText": prompt,
	}

	// Build textGenerationConfig
	config := make(map[string]interface{})

	maxTokens := 512 // Default
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	config["maxTokenCount"] = maxTokens

	if req.Temperature != nil {
		config["temperature"] = *req.Temperature
	}

	if req.TopP != nil {
		config["topP"] = *req.TopP
	}

	if len(req.Stop) > 0 {
		config["stopSequences"] = req.Stop
	}

	bedrockReq["textGenerationConfig"] = config

	return bedrockReq
}

// transformCohereRequest transforms a Warp request to Bedrock Cohere format.
//
// Bedrock Cohere API format:
//   - prompt field for input text
//   - Generation parameters at top level
//
// Example:
//
//	{
//	  "prompt": "Hello, world!",
//	  "max_tokens": 512,
//	  "temperature": 0.7
//	}
func transformCohereRequest(req *warp.CompletionRequest) map[string]interface{} {
	// Convert messages to prompt text
	prompt := convertMessagesToText(req.Messages)

	bedrockReq := map[string]interface{}{
		"prompt": prompt,
	}

	// Add parameters
	maxTokens := 512
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	bedrockReq["max_tokens"] = maxTokens

	if req.Temperature != nil {
		bedrockReq["temperature"] = *req.Temperature
	}

	if req.TopP != nil {
		bedrockReq["p"] = *req.TopP
	}

	if len(req.Stop) > 0 {
		bedrockReq["stop_sequences"] = req.Stop
	}

	return bedrockReq
}

// parseClaudeResponse parses a Bedrock Claude response to Warp format.
//
// Bedrock Claude response format:
//
//	{
//	  "id": "msg_...",
//	  "type": "message",
//	  "role": "assistant",
//	  "content": [{"type": "text", "text": "Hello!"}],
//	  "stop_reason": "end_turn",
//	  "usage": {"input_tokens": 10, "output_tokens": 20}
//	}
func parseClaudeResponse(body io.Reader) (*warp.CompletionResponse, error) {
	var bedrockResp struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(body).Decode(&bedrockResp); err != nil {
		return nil, fmt.Errorf("failed to decode Claude response: %w", err)
	}

	// Extract text from content array
	var text string
	if len(bedrockResp.Content) > 0 {
		text = bedrockResp.Content[0].Text
	}

	// Map stop_reason to finish_reason
	finishReason := "stop"
	switch bedrockResp.StopReason {
	case "end_turn":
		finishReason = "stop"
	case "max_tokens":
		finishReason = "length"
	case "stop_sequence":
		finishReason = "stop"
	case "tool_use":
		finishReason = "tool_calls"
	}

	// Build response
	resp := &warp.CompletionResponse{
		ID:      bedrockResp.ID,
		Object:  "chat.completion",
		Created: 0, // Bedrock doesn't provide timestamp
		Model:   "",
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    bedrockResp.Role,
					Content: text,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &warp.Usage{
			PromptTokens:     bedrockResp.Usage.InputTokens,
			CompletionTokens: bedrockResp.Usage.OutputTokens,
			TotalTokens:      bedrockResp.Usage.InputTokens + bedrockResp.Usage.OutputTokens,
		},
	}

	return resp, nil
}

// parseLlamaResponse parses a Bedrock Llama response to Warp format.
//
// Bedrock Llama response format:
//
//	{
//	  "generation": "Hello, world!",
//	  "prompt_token_count": 10,
//	  "generation_token_count": 20,
//	  "stop_reason": "stop"
//	}
func parseLlamaResponse(body io.Reader) (*warp.CompletionResponse, error) {
	var bedrockResp struct {
		Generation           string `json:"generation"`
		PromptTokenCount     int    `json:"prompt_token_count"`
		GenerationTokenCount int    `json:"generation_token_count"`
		StopReason           string `json:"stop_reason"`
	}

	if err := json.NewDecoder(body).Decode(&bedrockResp); err != nil {
		return nil, fmt.Errorf("failed to decode Llama response: %w", err)
	}

	// Map stop_reason to finish_reason
	finishReason := "stop"
	if bedrockResp.StopReason == "length" {
		finishReason = "length"
	}

	resp := &warp.CompletionResponse{
		ID:      "", // Llama doesn't provide ID
		Object:  "chat.completion",
		Created: 0,
		Model:   "",
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    "assistant",
					Content: bedrockResp.Generation,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &warp.Usage{
			PromptTokens:     bedrockResp.PromptTokenCount,
			CompletionTokens: bedrockResp.GenerationTokenCount,
			TotalTokens:      bedrockResp.PromptTokenCount + bedrockResp.GenerationTokenCount,
		},
	}

	return resp, nil
}

// parseTitanResponse parses a Bedrock Titan response to Warp format.
//
// Bedrock Titan response format:
//
//	{
//	  "results": [
//	    {
//	      "tokenCount": 20,
//	      "outputText": "Hello, world!",
//	      "completionReason": "FINISH"
//	    }
//	  ],
//	  "inputTextTokenCount": 10
//	}
func parseTitanResponse(body io.Reader) (*warp.CompletionResponse, error) {
	var bedrockResp struct {
		Results []struct {
			TokenCount       int    `json:"tokenCount"`
			OutputText       string `json:"outputText"`
			CompletionReason string `json:"completionReason"`
		} `json:"results"`
		InputTextTokenCount int `json:"inputTextTokenCount"`
	}

	if err := json.NewDecoder(body).Decode(&bedrockResp); err != nil {
		return nil, fmt.Errorf("failed to decode Titan response: %w", err)
	}

	if len(bedrockResp.Results) == 0 {
		return nil, fmt.Errorf("no results in Titan response")
	}

	result := bedrockResp.Results[0]

	// Map completion_reason to finish_reason
	finishReason := "stop"
	switch result.CompletionReason {
	case "FINISH":
		finishReason = "stop"
	case "LENGTH":
		finishReason = "length"
	}

	resp := &warp.CompletionResponse{
		ID:      "",
		Object:  "chat.completion",
		Created: 0,
		Model:   "",
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    "assistant",
					Content: result.OutputText,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &warp.Usage{
			PromptTokens:     bedrockResp.InputTextTokenCount,
			CompletionTokens: result.TokenCount,
			TotalTokens:      bedrockResp.InputTextTokenCount + result.TokenCount,
		},
	}

	return resp, nil
}

// parseCohereResponse parses a Bedrock Cohere response to Warp format.
func parseCohereResponse(body io.Reader) (*warp.CompletionResponse, error) {
	var bedrockResp struct {
		Generations []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"generations"`
	}

	if err := json.NewDecoder(body).Decode(&bedrockResp); err != nil {
		return nil, fmt.Errorf("failed to decode Cohere response: %w", err)
	}

	if len(bedrockResp.Generations) == 0 {
		return nil, fmt.Errorf("no generations in Cohere response")
	}

	gen := bedrockResp.Generations[0]

	resp := &warp.CompletionResponse{
		ID:      "",
		Object:  "chat.completion",
		Created: 0,
		Model:   "",
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    "assistant",
					Content: gen.Text,
				},
				FinishReason: gen.FinishReason,
			},
		},
	}

	return resp, nil
}

// convertMessagesToLlamaPrompt converts messages to Llama prompt format.
//
// Llama uses a specific prompt format with special tokens:
//
//	<s>[INST] user message [/INST] assistant response </s><s>[INST] user message [/INST]
func convertMessagesToLlamaPrompt(messages []warp.Message) string {
	var prompt string

	for i, msg := range messages {
		content := ""
		switch c := msg.Content.(type) {
		case string:
			content = c
		default:
			content = fmt.Sprintf("%v", c)
		}

		switch msg.Role {
		case "system":
			// System message is prepended to first user message
			prompt += content + "\n\n"
		case "user":
			if i == 0 || messages[i-1].Role != "assistant" {
				prompt += "<s>[INST] " + content + " [/INST]"
			} else {
				prompt += "<s>[INST] " + content + " [/INST]"
			}
		case "assistant":
			prompt += " " + content + " </s>"
		}
	}

	return prompt
}

// convertMessagesToText converts messages to plain text.
//
// Used for models that don't support structured message formats.
func convertMessagesToText(messages []warp.Message) string {
	var text string

	for _, msg := range messages {
		content := ""
		switch c := msg.Content.(type) {
		case string:
			content = c
		default:
			content = fmt.Sprintf("%v", c)
		}

		switch msg.Role {
		case "system":
			text += "System: " + content + "\n\n"
		case "user":
			text += "User: " + content + "\n\n"
		case "assistant":
			text += "Assistant: " + content + "\n\n"
		}
	}

	return text
}
