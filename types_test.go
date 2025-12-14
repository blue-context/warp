package warp

import (
	"encoding/json"
	"testing"
	"time"
)

func TestCompletionRequestJSON(t *testing.T) {
	tests := []struct {
		name    string
		req     CompletionRequest
		wantErr bool
	}{
		{
			name: "minimal request",
			req: CompletionRequest{
				Model: "openai/gpt-4",
				Messages: []Message{
					{Role: "user", Content: "Hello"},
				},
			},
			wantErr: false,
		},
		{
			name: "full request with all fields",
			req: CompletionRequest{
				Model: "openai/gpt-4",
				Messages: []Message{
					{Role: "system", Content: "You are helpful"},
					{Role: "user", Content: "Hello"},
				},
				Temperature:      ptrFloat64(0.7),
				MaxTokens:        ptrInt(100),
				TopP:             ptrFloat64(0.9),
				FrequencyPenalty: ptrFloat64(0.0),
				PresencePenalty:  ptrFloat64(0.0),
				Stop:             []string{"END"},
				N:                ptrInt(1),
				Tools: []Tool{
					{
						Type: "function",
						Function: Function{
							Name:        "get_weather",
							Description: "Get weather",
							Parameters:  map[string]any{"type": "object"},
						},
					},
				},
				APIKey:     "test-key",
				Metadata:   map[string]any{"user_id": "123"},
				NumRetries: 3,
				Fallbacks:  []string{"openai/gpt-3.5-turbo"},
				Timeout:    30 * time.Second,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal to JSON
			data, err := json.Marshal(tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("json.Marshal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			// Unmarshal back
			var got CompletionRequest
			if err := json.Unmarshal(data, &got); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify model and messages are preserved
			if got.Model != tt.req.Model {
				t.Errorf("Model = %v, want %v", got.Model, tt.req.Model)
			}

			if len(got.Messages) != len(tt.req.Messages) {
				t.Errorf("Messages length = %v, want %v", len(got.Messages), len(tt.req.Messages))
			}
		})
	}
}

func TestMessageContent(t *testing.T) {
	tests := []struct {
		name    string
		msg     Message
		wantErr bool
	}{
		{
			name: "simple text content",
			msg: Message{
				Role:    "user",
				Content: "Hello, world!",
			},
			wantErr: false,
		},
		{
			name: "multimodal content with text and image",
			msg: Message{
				Role: "user",
				Content: []ContentPart{
					{
						Type: "text",
						Text: "What's in this image?",
					},
					{
						Type: "image_url",
						ImageURL: &ImageURL{
							URL:    "https://example.com/image.jpg",
							Detail: "high",
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "assistant message with tool calls",
			msg: Message{
				Role:    "assistant",
				Content: "",
				ToolCalls: []ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location":"San Francisco"}`,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "tool response message",
			msg: Message{
				Role:       "tool",
				Content:    `{"temperature": 72, "condition": "sunny"}`,
				ToolCallID: "call_123",
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal to JSON
			data, err := json.Marshal(tt.msg)
			if (err != nil) != tt.wantErr {
				t.Errorf("json.Marshal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			// Unmarshal back
			var got Message
			if err := json.Unmarshal(data, &got); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify role is preserved
			if got.Role != tt.msg.Role {
				t.Errorf("Role = %v, want %v", got.Role, tt.msg.Role)
			}

			// Verify tool call ID if present
			if tt.msg.ToolCallID != "" && got.ToolCallID != tt.msg.ToolCallID {
				t.Errorf("ToolCallID = %v, want %v", got.ToolCallID, tt.msg.ToolCallID)
			}
		})
	}
}

func TestCompletionResponseJSON(t *testing.T) {
	tests := []struct {
		name    string
		resp    CompletionResponse
		wantErr bool
	}{
		{
			name: "basic response",
			resp: CompletionResponse{
				ID:      "chatcmpl-123",
				Object:  "chat.completion",
				Created: 1677652288,
				Model:   "gpt-4",
				Choices: []Choice{
					{
						Index: 0,
						Message: Message{
							Role:    "assistant",
							Content: "Hello! How can I help you today?",
						},
						FinishReason: "stop",
					},
				},
				Usage: &Usage{
					PromptTokens:     10,
					CompletionTokens: 9,
					TotalTokens:      19,
				},
			},
			wantErr: false,
		},
		{
			name: "response with tool calls",
			resp: CompletionResponse{
				ID:      "chatcmpl-456",
				Object:  "chat.completion",
				Created: 1677652288,
				Model:   "gpt-4",
				Choices: []Choice{
					{
						Index: 0,
						Message: Message{
							Role:    "assistant",
							Content: "",
							ToolCalls: []ToolCall{
								{
									ID:   "call_abc123",
									Type: "function",
									Function: FunctionCall{
										Name:      "get_weather",
										Arguments: `{"location":"Boston","unit":"celsius"}`,
									},
								},
							},
						},
						FinishReason: "tool_calls",
					},
				},
				Usage: &Usage{
					PromptTokens:     20,
					CompletionTokens: 15,
					TotalTokens:      35,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal to JSON
			data, err := json.Marshal(tt.resp)
			if (err != nil) != tt.wantErr {
				t.Errorf("json.Marshal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			// Unmarshal back
			var got CompletionResponse
			if err := json.Unmarshal(data, &got); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify basic fields
			if got.ID != tt.resp.ID {
				t.Errorf("ID = %v, want %v", got.ID, tt.resp.ID)
			}

			if got.Model != tt.resp.Model {
				t.Errorf("Model = %v, want %v", got.Model, tt.resp.Model)
			}

			if len(got.Choices) != len(tt.resp.Choices) {
				t.Errorf("Choices length = %v, want %v", len(got.Choices), len(tt.resp.Choices))
			}

			// Verify usage if present
			if tt.resp.Usage != nil {
				if got.Usage == nil {
					t.Error("Usage is nil, want non-nil")
				} else if got.Usage.TotalTokens != tt.resp.Usage.TotalTokens {
					t.Errorf("Usage.TotalTokens = %v, want %v", got.Usage.TotalTokens, tt.resp.Usage.TotalTokens)
				}
			}
		})
	}
}

func TestCompletionChunkJSON(t *testing.T) {
	tests := []struct {
		name    string
		chunk   CompletionChunk
		wantErr bool
	}{
		{
			name: "streaming chunk with content",
			chunk: CompletionChunk{
				ID:      "chatcmpl-123",
				Object:  "chat.completion.chunk",
				Created: 1677652288,
				Model:   "gpt-4",
				Choices: []ChunkChoice{
					{
						Index: 0,
						Delta: MessageDelta{
							Content: "Hello",
						},
						FinishReason: nil,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "final chunk with finish reason",
			chunk: CompletionChunk{
				ID:      "chatcmpl-123",
				Object:  "chat.completion.chunk",
				Created: 1677652288,
				Model:   "gpt-4",
				Choices: []ChunkChoice{
					{
						Index: 0,
						Delta: MessageDelta{
							Content: "",
						},
						FinishReason: ptrString("stop"),
					},
				},
				Usage: &Usage{
					PromptTokens:     10,
					CompletionTokens: 20,
					TotalTokens:      30,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal to JSON
			data, err := json.Marshal(tt.chunk)
			if (err != nil) != tt.wantErr {
				t.Errorf("json.Marshal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			// Unmarshal back
			var got CompletionChunk
			if err := json.Unmarshal(data, &got); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Verify basic fields
			if got.ID != tt.chunk.ID {
				t.Errorf("ID = %v, want %v", got.ID, tt.chunk.ID)
			}

			if len(got.Choices) != len(tt.chunk.Choices) {
				t.Errorf("Choices length = %v, want %v", len(got.Choices), len(tt.chunk.Choices))
			}
		})
	}
}

func TestToolsAndFunctions(t *testing.T) {
	tool := Tool{
		Type: "function",
		Function: Function{
			Name:        "get_weather",
			Description: "Get the current weather in a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
					"unit": map[string]any{
						"type": "string",
						"enum": []string{"celsius", "fahrenheit"},
					},
				},
				"required": []string{"location"},
			},
		},
	}

	// Marshal to JSON
	data, err := json.Marshal(tool)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// Unmarshal back
	var got Tool
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	// Verify fields
	if got.Type != tool.Type {
		t.Errorf("Type = %v, want %v", got.Type, tool.Type)
	}

	if got.Function.Name != tool.Function.Name {
		t.Errorf("Function.Name = %v, want %v", got.Function.Name, tool.Function.Name)
	}

	// Verify parameters exist and have the correct type field
	if got.Function.Parameters == nil {
		t.Error("Function.Parameters is nil")
	} else if paramType, ok := got.Function.Parameters["type"].(string); !ok || paramType != "object" {
		t.Errorf("Function.Parameters[type] = %v, want 'object'", got.Function.Parameters["type"])
	}
}

func TestUsageDetails(t *testing.T) {
	usage := Usage{
		PromptTokens:     100,
		CompletionTokens: 50,
		TotalTokens:      150,
		PromptDetails: &PromptTokensDetails{
			CachedTokens: 20,
			AudioTokens:  5,
		},
		CompletionDetails: &CompletionTokensDetails{
			ReasoningTokens: 10,
			AudioTokens:     3,
		},
	}

	// Marshal to JSON
	data, err := json.Marshal(usage)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	// Unmarshal back
	var got Usage
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	// Verify fields
	if got.TotalTokens != usage.TotalTokens {
		t.Errorf("TotalTokens = %v, want %v", got.TotalTokens, usage.TotalTokens)
	}

	if got.PromptDetails == nil {
		t.Fatal("PromptDetails is nil")
	}

	if got.PromptDetails.CachedTokens != usage.PromptDetails.CachedTokens {
		t.Errorf("PromptDetails.CachedTokens = %v, want %v",
			got.PromptDetails.CachedTokens, usage.PromptDetails.CachedTokens)
	}

	if got.CompletionDetails == nil {
		t.Fatal("CompletionDetails is nil")
	}

	if got.CompletionDetails.ReasoningTokens != usage.CompletionDetails.ReasoningTokens {
		t.Errorf("CompletionDetails.ReasoningTokens = %v, want %v",
			got.CompletionDetails.ReasoningTokens, usage.CompletionDetails.ReasoningTokens)
	}
}

// Helper functions for creating pointers
func ptrFloat64(v float64) *float64 {
	return &v
}

func ptrInt(v int) *int {
	return &v
}

func ptrString(v string) *string {
	return &v
}
