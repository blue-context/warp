package bedrock

import (
	"bytes"
	"io"
	"testing"

	"github.com/blue-context/warp"
)

func TestTransformLlamaRequest(t *testing.T) {
	maxTokens := 512
	temperature := 0.8

	req := &warp.CompletionRequest{
		Model: "llama3-70b",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
	}

	bedrockReq := transformLlamaRequest(req)

	// Check required fields
	if _, ok := bedrockReq["prompt"].(string); !ok {
		t.Error("missing prompt field")
	}

	if bedrockReq["max_gen_len"] != 512 {
		t.Error("max_gen_len not set correctly")
	}

	if bedrockReq["temperature"] != 0.8 {
		t.Error("temperature not set correctly")
	}
}

func TestTransformTitanRequest(t *testing.T) {
	maxTokens := 256
	temperature := 0.5
	topP := 0.9

	req := &warp.CompletionRequest{
		Model: "titan-text-express",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		TopP:        &topP,
		Stop:        []string{"END"},
	}

	bedrockReq := transformTitanRequest(req)

	// Check inputText
	if _, ok := bedrockReq["inputText"].(string); !ok {
		t.Error("missing inputText field")
	}

	// Check textGenerationConfig
	config, ok := bedrockReq["textGenerationConfig"].(map[string]interface{})
	if !ok {
		t.Fatal("missing textGenerationConfig")
	}

	if config["maxTokenCount"] != 256 {
		t.Error("maxTokenCount not set correctly")
	}

	if config["temperature"] != 0.5 {
		t.Error("temperature not set correctly")
	}

	if config["topP"] != 0.9 {
		t.Error("topP not set correctly")
	}

	stopSeqs, ok := config["stopSequences"].([]string)
	if !ok || len(stopSeqs) != 1 || stopSeqs[0] != "END" {
		t.Error("stopSequences not set correctly")
	}
}

func TestTransformCohereRequest(t *testing.T) {
	maxTokens := 300
	temperature := 0.6
	topP := 0.85

	req := &warp.CompletionRequest{
		Model: "command-r",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
		TopP:        &topP,
		Stop:        []string{"STOP"},
	}

	bedrockReq := transformCohereRequest(req)

	if _, ok := bedrockReq["prompt"].(string); !ok {
		t.Error("missing prompt field")
	}

	if bedrockReq["max_tokens"] != 300 {
		t.Error("max_tokens not set correctly")
	}

	if bedrockReq["temperature"] != 0.6 {
		t.Error("temperature not set correctly")
	}

	if bedrockReq["p"] != 0.85 {
		t.Error("p (topP) not set correctly")
	}

	stopSeqs, ok := bedrockReq["stop_sequences"].([]string)
	if !ok || len(stopSeqs) != 1 || stopSeqs[0] != "STOP" {
		t.Error("stop_sequences not set correctly")
	}
}

func TestParseLlamaResponse(t *testing.T) {
	response := `{
		"generation": "Hello! How can I help you?",
		"prompt_token_count": 15,
		"generation_token_count": 8,
		"stop_reason": "stop"
	}`

	resp, err := parseLlamaResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseLlamaResponse() error = %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("no choices in response")
	}

	choice := resp.Choices[0]
	if choice.Message.Content != "Hello! How can I help you?" {
		t.Errorf("content = %q, want %q", choice.Message.Content, "Hello! How can I help you?")
	}

	if choice.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want %q", choice.FinishReason, "stop")
	}

	if resp.Usage.PromptTokens != 15 {
		t.Errorf("prompt_tokens = %d, want 15", resp.Usage.PromptTokens)
	}

	if resp.Usage.CompletionTokens != 8 {
		t.Errorf("completion_tokens = %d, want 8", resp.Usage.CompletionTokens)
	}
}

func TestParseLlamaResponseLengthLimit(t *testing.T) {
	response := `{
		"generation": "Response text",
		"prompt_token_count": 10,
		"generation_token_count": 5,
		"stop_reason": "length"
	}`

	resp, err := parseLlamaResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseLlamaResponse() error = %v", err)
	}

	if resp.Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "length")
	}
}

func TestParseTitanResponse(t *testing.T) {
	response := `{
		"results": [
			{
				"tokenCount": 12,
				"outputText": "Hello from Titan!",
				"completionReason": "FINISH"
			}
		],
		"inputTextTokenCount": 5
	}`

	resp, err := parseTitanResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseTitanResponse() error = %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("no choices in response")
	}

	choice := resp.Choices[0]
	if choice.Message.Content != "Hello from Titan!" {
		t.Errorf("content = %q, want %q", choice.Message.Content, "Hello from Titan!")
	}

	if choice.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want %q", choice.FinishReason, "stop")
	}

	if resp.Usage.PromptTokens != 5 {
		t.Errorf("prompt_tokens = %d, want 5", resp.Usage.PromptTokens)
	}

	if resp.Usage.CompletionTokens != 12 {
		t.Errorf("completion_tokens = %d, want 12", resp.Usage.CompletionTokens)
	}
}

func TestParseTitanResponseLengthLimit(t *testing.T) {
	response := `{
		"results": [
			{
				"tokenCount": 100,
				"outputText": "Response",
				"completionReason": "LENGTH"
			}
		],
		"inputTextTokenCount": 10
	}`

	resp, err := parseTitanResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseTitanResponse() error = %v", err)
	}

	if resp.Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "length")
	}
}

func TestParseTitanResponseNoResults(t *testing.T) {
	response := `{
		"results": [],
		"inputTextTokenCount": 5
	}`

	_, err := parseTitanResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err == nil {
		t.Error("expected error for response with no results")
	}
}

func TestParseCohereResponse(t *testing.T) {
	response := `{
		"generations": [
			{
				"text": "Hello from Cohere!",
				"finish_reason": "COMPLETE"
			}
		]
	}`

	resp, err := parseCohereResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseCohereResponse() error = %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("no choices in response")
	}

	choice := resp.Choices[0]
	if choice.Message.Content != "Hello from Cohere!" {
		t.Errorf("content = %q, want %q", choice.Message.Content, "Hello from Cohere!")
	}

	if choice.FinishReason != "COMPLETE" {
		t.Errorf("finish_reason = %q, want %q", choice.FinishReason, "COMPLETE")
	}
}

func TestParseCohereResponseNoGenerations(t *testing.T) {
	response := `{
		"generations": []
	}`

	_, err := parseCohereResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err == nil {
		t.Error("expected error for response with no generations")
	}
}

func TestParseClaudeResponseToolUse(t *testing.T) {
	response := `{
		"id": "msg_123",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "text", "text": "Using tool..."}],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 10}
	}`

	resp, err := parseClaudeResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseClaudeResponse() error = %v", err)
	}

	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "tool_calls")
	}
}

func TestParseClaudeResponseMaxTokens(t *testing.T) {
	response := `{
		"id": "msg_123",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "text", "text": "Response"}],
		"stop_reason": "max_tokens",
		"usage": {"input_tokens": 10, "output_tokens": 100}
	}`

	resp, err := parseClaudeResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseClaudeResponse() error = %v", err)
	}

	if resp.Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "length")
	}
}

func TestParseClaudeResponseStopSequence(t *testing.T) {
	response := `{
		"id": "msg_123",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "text", "text": "Response"}],
		"stop_reason": "stop_sequence",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	resp, err := parseClaudeResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err != nil {
		t.Fatalf("parseClaudeResponse() error = %v", err)
	}

	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
}

func TestTransformClaudeRequestWithTools(t *testing.T) {
	maxTokens := 1024

	req := &warp.CompletionRequest{
		Model: "claude-3-opus",
		Messages: []warp.Message{
			{Role: "user", Content: "Use this tool"},
		},
		MaxTokens: &maxTokens,
		Tools: []warp.Tool{
			{
				Type: "function",
				Function: warp.Function{
					Name:        "get_weather",
					Description: "Get weather information",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type": "string",
							},
						},
					},
				},
			},
		},
	}

	bedrockReq := transformClaudeRequest(req)

	tools, ok := bedrockReq["tools"].([]map[string]interface{})
	if !ok {
		t.Fatal("tools field not set")
	}

	if len(tools) != 1 {
		t.Errorf("tools length = %d, want 1", len(tools))
	}

	if tools[0]["name"] != "get_weather" {
		t.Error("tool name not set correctly")
	}
}

func TestTransformClaudeRequestWithStopSequences(t *testing.T) {
	maxTokens := 512

	req := &warp.CompletionRequest{
		Model: "claude-3-opus",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: &maxTokens,
		Stop:      []string{"STOP1", "STOP2"},
	}

	bedrockReq := transformClaudeRequest(req)

	stopSeqs, ok := bedrockReq["stop_sequences"].([]string)
	if !ok {
		t.Fatal("stop_sequences not set")
	}

	if len(stopSeqs) != 2 {
		t.Errorf("stop_sequences length = %d, want 2", len(stopSeqs))
	}
}

func TestTransformClaudeRequestMultimodal(t *testing.T) {
	maxTokens := 1024

	req := &warp.CompletionRequest{
		Model: "claude-3-opus",
		Messages: []warp.Message{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "text",
						"text": "What's in this image?",
					},
					map[string]interface{}{
						"type": "image_url",
						"image_url": map[string]interface{}{
							"url": "https://example.com/image.jpg",
						},
					},
				},
			},
		},
		MaxTokens: &maxTokens,
	}

	bedrockReq := transformClaudeRequest(req)

	messages, ok := bedrockReq["messages"].([]map[string]interface{})
	if !ok {
		t.Fatal("messages not set")
	}

	if len(messages) != 1 {
		t.Fatalf("messages length = %d, want 1", len(messages))
	}

	// Content should be preserved as-is for multimodal
	_, ok = messages[0]["content"].([]interface{})
	if !ok {
		t.Error("multimodal content not preserved")
	}
}

func TestTransformClaudeRequestNonStringContent(t *testing.T) {
	maxTokens := 512

	// Test with numeric content (edge case)
	req := &warp.CompletionRequest{
		Model: "claude-3-opus",
		Messages: []warp.Message{
			{Role: "user", Content: 12345},
		},
		MaxTokens: &maxTokens,
	}

	bedrockReq := transformClaudeRequest(req)

	messages, ok := bedrockReq["messages"].([]map[string]interface{})
	if !ok {
		t.Fatal("messages not set")
	}

	if len(messages) != 1 {
		t.Fatalf("messages length = %d, want 1", len(messages))
	}

	// Should convert to string
	content, ok := messages[0]["content"].(string)
	if !ok {
		t.Error("content not converted to string")
	}

	if content != "12345" {
		t.Errorf("content = %q, want %q", content, "12345")
	}
}

func TestParseClaudeResponseInvalidJSON(t *testing.T) {
	response := `invalid json`

	_, err := parseClaudeResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseLlamaResponseInvalidJSON(t *testing.T) {
	response := `invalid json`

	_, err := parseLlamaResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseTitanResponseInvalidJSON(t *testing.T) {
	response := `invalid json`

	_, err := parseTitanResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseCohereResponseInvalidJSON(t *testing.T) {
	response := `invalid json`

	_, err := parseCohereResponse(io.NopCloser(bytes.NewBufferString(response)))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestTransformRequestDefaults(t *testing.T) {
	// Test that defaults are applied when optional fields are not set

	t.Run("claude defaults", func(t *testing.T) {
		req := &warp.CompletionRequest{
			Model: "claude-3-opus",
			Messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
		}

		bedrockReq := transformClaudeRequest(req)

		// Default max_tokens should be 1024
		if bedrockReq["max_tokens"] != 1024 {
			t.Errorf("default max_tokens = %v, want 1024", bedrockReq["max_tokens"])
		}
	})

	t.Run("llama defaults", func(t *testing.T) {
		req := &warp.CompletionRequest{
			Model: "llama3-70b",
			Messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
		}

		bedrockReq := transformLlamaRequest(req)

		// Default max_gen_len should be 512
		if bedrockReq["max_gen_len"] != 512 {
			t.Errorf("default max_gen_len = %v, want 512", bedrockReq["max_gen_len"])
		}
	})

	t.Run("titan defaults", func(t *testing.T) {
		req := &warp.CompletionRequest{
			Model: "titan-text-express",
			Messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
		}

		bedrockReq := transformTitanRequest(req)

		config := bedrockReq["textGenerationConfig"].(map[string]interface{})

		// Default maxTokenCount should be 512
		if config["maxTokenCount"] != 512 {
			t.Errorf("default maxTokenCount = %v, want 512", config["maxTokenCount"])
		}
	})

	t.Run("cohere defaults", func(t *testing.T) {
		req := &warp.CompletionRequest{
			Model: "command-r",
			Messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
		}

		bedrockReq := transformCohereRequest(req)

		// Default max_tokens should be 512
		if bedrockReq["max_tokens"] != 512 {
			t.Errorf("default max_tokens = %v, want 512", bedrockReq["max_tokens"])
		}
	})
}
