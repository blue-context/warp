package vertex

import (
	"testing"

	"github.com/blue-context/warp"
)

func TestTransformRequest(t *testing.T) {
	tests := []struct {
		name    string
		req     *warp.CompletionRequest
		wantErr bool
	}{
		{
			name: "basic request",
			req: &warp.CompletionRequest{
				Model: "gemini-pro",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			wantErr: false,
		},
		{
			name: "with system message",
			req: &warp.CompletionRequest{
				Model: "gemini-pro",
				Messages: []warp.Message{
					{Role: "system", Content: "You are a helpful assistant"},
					{Role: "user", Content: "Hello"},
				},
			},
			wantErr: false,
		},
		{
			name: "with parameters",
			req: &warp.CompletionRequest{
				Model: "gemini-pro",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
				TopP:        floatPtr(0.9),
				Stop:        []string{"END"},
				N:           intPtr(1),
			},
			wantErr: false,
		},
		{
			name: "with tools",
			req: &warp.CompletionRequest{
				Model: "gemini-pro",
				Messages: []warp.Message{
					{Role: "user", Content: "What's the weather?"},
				},
				Tools: []warp.Tool{
					{
						Type: "function",
						Function: warp.Function{
							Name:        "get_weather",
							Description: "Get weather",
							Parameters: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"location": map[string]any{"type": "string"},
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "with tool calls",
			req: &warp.CompletionRequest{
				Model: "gemini-pro",
				Messages: []warp.Message{
					{Role: "user", Content: "What's the weather?"},
					{
						Role:    "assistant",
						Content: "",
						ToolCalls: []warp.ToolCall{
							{
								ID:   "call_1",
								Type: "function",
								Function: warp.FunctionCall{
									Name:      "get_weather",
									Arguments: `{"location":"SF"}`,
								},
							},
						},
					},
					{
						Role:       "tool",
						ToolCallID: "call_1",
						Name:       "get_weather",
						Content:    `{"temp":70}`,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multimodal with typed content parts",
			req: &warp.CompletionRequest{
				Model: "gemini-pro-vision",
				Messages: []warp.Message{
					{
						Role: "user",
						Content: []warp.ContentPart{
							{Type: "text", Text: "What's in this image?"},
							{
								Type: "image_url",
								ImageURL: &warp.ImageURL{
									URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
		},
		{
			name: "empty messages",
			req: &warp.CompletionRequest{
				Model:    "gemini-pro",
				Messages: []warp.Message{},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vReq, err := transformRequest(tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("transformRequest() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("transformRequest() unexpected error = %v", err)
				return
			}

			if vReq == nil {
				t.Error("transformRequest() returned nil")
				return
			}

			// Verify basic structure
			if len(vReq.Contents) == 0 && len(tt.req.Messages) > 0 {
				// System-only messages are prepended, so this might be empty
				hasNonSystem := false
				for _, msg := range tt.req.Messages {
					if msg.Role != "system" {
						hasNonSystem = true
						break
					}
				}
				if hasNonSystem {
					t.Error("transformRequest() returned empty contents")
				}
			}
		})
	}
}

func TestTransformResponse(t *testing.T) {
	tests := []struct {
		name     string
		vResp    *vertexResponse
		model    string
		wantErr  bool
		validate func(*testing.T, *warp.CompletionResponse)
	}{
		{
			name: "basic response",
			vResp: &vertexResponse{
				Candidates: []vertexCandidate{
					{
						Content: vertexContent{
							Role:  "model",
							Parts: []vertexPart{{Text: "Hello!"}},
						},
						FinishReason: "STOP",
						Index:        0,
					},
				},
				UsageMetadata: &vertexUsageMetadata{
					PromptTokenCount:     10,
					CandidatesTokenCount: 5,
					TotalTokenCount:      15,
				},
			},
			model:   "gemini-pro",
			wantErr: false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if len(resp.Choices) != 1 {
					t.Errorf("expected 1 choice, got %d", len(resp.Choices))
				}
				if resp.Choices[0].Message.Content != "Hello!" {
					t.Errorf("unexpected content: %s", resp.Choices[0].Message.Content)
				}
				if resp.Choices[0].FinishReason != "stop" {
					t.Errorf("unexpected finish reason: %s", resp.Choices[0].FinishReason)
				}
				if resp.Usage == nil {
					t.Error("usage is nil")
				} else if resp.Usage.TotalTokens != 15 {
					t.Errorf("expected 15 total tokens, got %d", resp.Usage.TotalTokens)
				}
			},
		},
		{
			name: "with function call",
			vResp: &vertexResponse{
				Candidates: []vertexCandidate{
					{
						Content: vertexContent{
							Role: "model",
							Parts: []vertexPart{
								{
									FunctionCall: &vertexFunctionCall{
										Name: "get_weather",
										Args: map[string]interface{}{
											"location": "SF",
										},
									},
								},
							},
						},
						FinishReason: "STOP",
						Index:        0,
					},
				},
			},
			model:   "gemini-pro",
			wantErr: false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if len(resp.Choices) != 1 {
					t.Errorf("expected 1 choice, got %d", len(resp.Choices))
				}
				if len(resp.Choices[0].Message.ToolCalls) != 1 {
					t.Errorf("expected 1 tool call, got %d", len(resp.Choices[0].Message.ToolCalls))
				}
				if resp.Choices[0].Message.ToolCalls[0].Function.Name != "get_weather" {
					t.Errorf("unexpected function name: %s", resp.Choices[0].Message.ToolCalls[0].Function.Name)
				}
			},
		},
		{
			name:    "nil response",
			vResp:   nil,
			model:   "gemini-pro",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := transformResponse(tt.vResp, tt.model)

			if tt.wantErr {
				if err == nil {
					t.Error("transformResponse() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("transformResponse() unexpected error = %v", err)
				return
			}

			if resp == nil {
				t.Error("transformResponse() returned nil")
				return
			}

			if tt.validate != nil {
				tt.validate(t, resp)
			}
		})
	}
}

func TestTransformRole(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"user", "user"},
		{"assistant", "model"},
		{"tool", "function"},
		{"system", "system"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := transformRole(tt.input)
			if got != tt.expected {
				t.Errorf("transformRole(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestInverseTransformRole(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"user", "user"},
		{"model", "assistant"},
		{"function", "tool"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := inverseTransformRole(tt.input)
			if got != tt.expected {
				t.Errorf("inverseTransformRole(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestTransformFinishReason(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"STOP", "stop"},
		{"MAX_TOKENS", "length"},
		{"SAFETY", "content_filter"},
		{"RECITATION", "content_filter"},
		{"UNKNOWN", "UNKNOWN"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := transformFinishReason(tt.input)
			if got != tt.expected {
				t.Errorf("transformFinishReason(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestParseDataURI(t *testing.T) {
	tests := []struct {
		name         string
		uri          string
		wantMimeType string
		wantData     string
		wantErr      bool
	}{
		{
			name:         "valid image/jpeg",
			uri:          "data:image/jpeg;base64,/9j/4AAQSkZJRg",
			wantMimeType: "image/jpeg",
			wantData:     "/9j/4AAQSkZJRg",
			wantErr:      false,
		},
		{
			name:         "valid image/png",
			uri:          "data:image/png;base64,iVBORw0KGgo",
			wantMimeType: "image/png",
			wantData:     "iVBORw0KGgo",
			wantErr:      false,
		},
		{
			name:    "missing data prefix",
			uri:     "image/jpeg;base64,/9j/4AAQ",
			wantErr: true,
		},
		{
			name:    "missing semicolon",
			uri:     "data:image/jpegbase64,/9j/4AAQ",
			wantErr: true,
		},
		{
			name:    "missing comma",
			uri:     "data:image/jpeg;base64/9j/4AAQ",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mimeType, data, err := parseDataURI(tt.uri)

			if tt.wantErr {
				if err == nil {
					t.Error("parseDataURI() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("parseDataURI() unexpected error = %v", err)
				return
			}

			if mimeType != tt.wantMimeType {
				t.Errorf("parseDataURI() mimeType = %q, want %q", mimeType, tt.wantMimeType)
			}

			if data != tt.wantData {
				t.Errorf("parseDataURI() data = %q, want %q", data, tt.wantData)
			}
		})
	}
}

func TestExtractTextContent(t *testing.T) {
	tests := []struct {
		name     string
		content  interface{}
		expected string
	}{
		{
			name:     "string content",
			content:  "Hello",
			expected: "Hello",
		},
		{
			name: "typed content parts with text",
			content: []warp.ContentPart{
				{Type: "text", Text: "Hello"},
			},
			expected: "Hello",
		},
		{
			name: "interface content parts with text",
			content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": "Hello",
				},
			},
			expected: "Hello",
		},
		{
			name:     "empty string",
			content:  "",
			expected: "",
		},
		{
			name:     "nil",
			content:  nil,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTextContent(tt.content)
			if got != tt.expected {
				t.Errorf("extractTextContent() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestJoinTextParts(t *testing.T) {
	tests := []struct {
		name     string
		parts    []string
		expected string
	}{
		{
			name:     "empty",
			parts:    []string{},
			expected: "",
		},
		{
			name:     "single part",
			parts:    []string{"Hello"},
			expected: "Hello",
		},
		{
			name:     "multiple parts",
			parts:    []string{"Hello", "World"},
			expected: "Hello\nWorld",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := joinTextParts(tt.parts)
			if got != tt.expected {
				t.Errorf("joinTextParts() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestTransformTools(t *testing.T) {
	tests := []struct {
		name     string
		tools    []warp.Tool
		wantNil  bool
		validate func(*testing.T, []vertexTool)
	}{
		{
			name:    "empty tools",
			tools:   []warp.Tool{},
			wantNil: true,
		},
		{
			name: "single function tool",
			tools: []warp.Tool{
				{
					Type: "function",
					Function: warp.Function{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters: map[string]any{
							"type": "object",
						},
					},
				},
			},
			wantNil: false,
			validate: func(t *testing.T, vTools []vertexTool) {
				if len(vTools) != 1 {
					t.Errorf("expected 1 tool, got %d", len(vTools))
				}
				if len(vTools[0].FunctionDeclarations) != 1 {
					t.Errorf("expected 1 function declaration, got %d", len(vTools[0].FunctionDeclarations))
				}
				if vTools[0].FunctionDeclarations[0].Name != "get_weather" {
					t.Errorf("unexpected function name: %s", vTools[0].FunctionDeclarations[0].Name)
				}
			},
		},
		{
			name: "non-function tool",
			tools: []warp.Tool{
				{
					Type: "other",
				},
			},
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vTools := transformTools(tt.tools)

			if tt.wantNil && vTools != nil {
				t.Error("transformTools() expected nil, got non-nil")
				return
			}

			if !tt.wantNil && vTools == nil {
				t.Error("transformTools() expected non-nil, got nil")
				return
			}

			if tt.validate != nil && vTools != nil {
				tt.validate(t, vTools)
			}
		})
	}
}

func TestTransformMessage_InvalidToolResponse(t *testing.T) {
	msg := warp.Message{
		Role:       "tool",
		ToolCallID: "call_1",
		Name:       "get_weather",
		Content:    "not valid json {",
	}

	_, err := transformMessage(msg)
	if err != nil {
		t.Errorf("transformMessage() should handle invalid JSON gracefully, got error: %v", err)
	}
}

func TestTransformContentPart_Errors(t *testing.T) {
	tests := []struct {
		name    string
		item    interface{}
		wantErr bool
	}{
		{
			name:    "not an object",
			item:    "string",
			wantErr: true,
		},
		{
			name: "unsupported type",
			item: map[string]interface{}{
				"type": "unknown",
			},
			wantErr: true,
		},
		{
			name: "image_url not object",
			item: map[string]interface{}{
				"type":      "image_url",
				"image_url": "string",
			},
			wantErr: true,
		},
		{
			name: "invalid data URI",
			item: map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": "not a data uri",
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := transformContentPart(tt.item)
			if tt.wantErr && err == nil {
				t.Error("transformContentPart() expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("transformContentPart() unexpected error = %v", err)
			}
		})
	}
}

func TestTransformTypedContentPart_Errors(t *testing.T) {
	tests := []struct {
		name    string
		item    warp.ContentPart
		wantErr bool
	}{
		{
			name: "unsupported type",
			item: warp.ContentPart{
				Type: "unknown",
			},
			wantErr: true,
		},
		{
			name: "nil image URL",
			item: warp.ContentPart{
				Type:     "image_url",
				ImageURL: nil,
			},
			wantErr: true,
		},
		{
			name: "invalid data URI",
			item: warp.ContentPart{
				Type: "image_url",
				ImageURL: &warp.ImageURL{
					URL: "not a data uri",
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := transformTypedContentPart(tt.item)
			if tt.wantErr && err == nil {
				t.Error("transformTypedContentPart() expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("transformTypedContentPart() unexpected error = %v", err)
			}
		})
	}
}

func TestTransformMessage_UnsupportedContentType(t *testing.T) {
	msg := warp.Message{
		Role:    "user",
		Content: 123, // Invalid type
	}

	_, err := transformMessage(msg)
	if err == nil {
		t.Error("transformMessage() expected error for unsupported content type, got nil")
	}
}

func TestGenerateIDs(t *testing.T) {
	// Test that IDs are generated
	id1 := generateResponseID()
	if id1 == "" {
		t.Error("generateResponseID() returned empty string")
	}

	id2 := generateToolCallID()
	if id2 == "" {
		t.Error("generateToolCallID() returned empty string")
	}

	// IDs should be different if called at different times
	if id1 == id2 {
		t.Logf("IDs happened to be the same: %s", id1)
	}
}

// Helper functions

func floatPtr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
