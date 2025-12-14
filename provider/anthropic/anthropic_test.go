package anthropic

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/blue-context/warp"
	prov "github.com/blue-context/warp/provider"
)

// mockHTTPClient is a mock HTTP client for testing
type mockHTTPClient struct {
	doFunc func(req *http.Request) (*http.Response, error)
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return m.doFunc(req)
}

// float64Ptr returns a pointer to a float64 value
func float64Ptr(v float64) *float64 {
	return &v
}

// intPtr returns a pointer to an int value
func intPtr(v int) *int {
	return &v
}

// TestNewProvider tests the NewProvider constructor
func TestNewProvider(t *testing.T) {
	tests := []struct {
		name    string
		opts    []Option
		wantErr bool
		errMsg  string
	}{
		{
			name:    "missing API key",
			opts:    []Option{},
			wantErr: true,
			errMsg:  "Anthropic API key is required",
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("sk-ant-test"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithAPIKey("sk-ant-test"),
				WithAPIBase("https://custom.example.com"),
				WithAPIVersion("2024-01-01"),
				WithHTTPClient(&mockHTTPClient{}),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewProvider(tt.opts...)

			if tt.wantErr {
				if err == nil {
					t.Error("NewProvider() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("NewProvider() error = %v, want error containing %q", err, tt.errMsg)
				}
				return
			}

			if err != nil {
				t.Errorf("NewProvider() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if provider == nil {
				t.Error("NewProvider() returned nil provider")
			}
		})
	}
}

// TestProviderName tests the Name method
func TestProviderName(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-ant-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "anthropic" {
		t.Errorf("Name() = %v, want %v", got, "anthropic")
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-ant-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	capsInterface := provider.Supports()
	caps, ok := capsInterface.(prov.Capabilities)
	if !ok {
		t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
	}

	tests := []struct {
		name     string
		value    bool
		expected bool
	}{
		{"Completion", caps.Completion, true},
		{"Streaming", caps.Streaming, true},
		{"Embedding", caps.Embedding, false},
		{"ImageGeneration", caps.ImageGeneration, false},
		{"Transcription", caps.Transcription, false},
		{"Speech", caps.Speech, false},
		{"Moderation", caps.Moderation, false},
		{"FunctionCalling", caps.FunctionCalling, true},
		{"Vision", caps.Vision, true},
		{"JSON", caps.JSON, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value != tt.expected {
				t.Errorf("Supports().%s = %v, want %v", tt.name, tt.value, tt.expected)
			}
		})
	}
}

// TestEmbedding tests that Embedding returns an error
func TestEmbedding(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-ant-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	req := &warp.EmbeddingRequest{
		Model: "text-embedding-ada-002",
		Input: "test",
	}

	resp, err := provider.Embedding(context.Background(), req)
	if err == nil {
		t.Error("Embedding() error = nil, want error")
	}
	if resp != nil {
		t.Error("Embedding() returned non-nil response")
	}
	if !strings.Contains(err.Error(), "not support embeddings") {
		t.Errorf("Embedding() error = %v, want error about embeddings not supported", err)
	}
}

// TestCompletion tests the Completion method
func TestCompletion(t *testing.T) {
	tests := []struct {
		name       string
		req        *warp.CompletionRequest
		mockResp   string
		statusCode int
		wantErr    bool
		validate   func(*testing.T, *warp.CompletionResponse)
	}{
		{
			name: "successful completion",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"id": "msg_01ABC",
				"type": "message",
				"role": "assistant",
				"model": "claude-3-opus-20240229",
				"content": [{
					"type": "text",
					"text": "Hello! How can I assist you today?"
				}],
				"stop_reason": "end_turn",
				"usage": {
					"input_tokens": 10,
					"output_tokens": 8
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.ID != "msg_01ABC" {
					t.Errorf("ID = %v, want msg_01ABC", resp.ID)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				if resp.Choices[0].Message.Content != "Hello! How can I assist you today?" {
					t.Errorf("Message.Content = %v", resp.Choices[0].Message.Content)
				}
				if resp.Choices[0].FinishReason != "stop" {
					t.Errorf("FinishReason = %v, want stop", resp.Choices[0].FinishReason)
				}
				if resp.Usage.PromptTokens != 10 {
					t.Errorf("Usage.PromptTokens = %v, want 10", resp.Usage.PromptTokens)
				}
				if resp.Usage.CompletionTokens != 8 {
					t.Errorf("Usage.CompletionTokens = %v, want 8", resp.Usage.CompletionTokens)
				}
			},
		},
		{
			name: "with system message",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "system", Content: "You are a helpful assistant"},
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"id": "msg_02DEF",
				"type": "message",
				"role": "assistant",
				"model": "claude-3-opus-20240229",
				"content": [{
					"type": "text",
					"text": "Hi there!"
				}],
				"stop_reason": "end_turn",
				"usage": {
					"input_tokens": 15,
					"output_tokens": 5
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.Choices[0].Message.Content != "Hi there!" {
					t.Errorf("Message.Content = %v", resp.Choices[0].Message.Content)
				}
			},
		},
		{
			name: "with max_tokens",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: intPtr(100),
			},
			mockResp: `{
				"id": "msg_03GHI",
				"type": "message",
				"role": "assistant",
				"model": "claude-3-opus-20240229",
				"content": [{
					"type": "text",
					"text": "Response"
				}],
				"stop_reason": "max_tokens",
				"usage": {
					"input_tokens": 10,
					"output_tokens": 100
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.Choices[0].FinishReason != "length" {
					t.Errorf("FinishReason = %v, want length", resp.Choices[0].FinishReason)
				}
			},
		},
		{
			name: "with temperature and top_p",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				Temperature: float64Ptr(0.7),
				TopP:        float64Ptr(0.9),
			},
			mockResp: `{
				"id": "msg_04JKL",
				"type": "message",
				"role": "assistant",
				"model": "claude-3-opus-20240229",
				"content": [{
					"type": "text",
					"text": "Response"
				}],
				"stop_reason": "end_turn",
				"usage": {
					"input_tokens": 10,
					"output_tokens": 5
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "authentication error",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Invalid API key",
					"type": "authentication_error"
				}
			}`,
			statusCode: http.StatusUnauthorized,
			wantErr:    true,
		},
		{
			name: "rate limit error",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Rate limit exceeded",
					"type": "rate_limit_error"
				}
			}`,
			statusCode: http.StatusTooManyRequests,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					// Verify request
					if req.Method != "POST" {
						t.Errorf("Method = %v, want POST", req.Method)
					}
					if !strings.HasSuffix(req.URL.Path, "/v1/messages") {
						t.Errorf("URL path = %v, want suffix /v1/messages", req.URL.Path)
					}
					if apiKey := req.Header.Get("x-api-key"); apiKey != "sk-ant-test" {
						t.Errorf("x-api-key = %v, want sk-ant-test", apiKey)
					}
					if version := req.Header.Get("anthropic-version"); version == "" {
						t.Error("anthropic-version header not set")
					}

					// Return mock response
					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       io.NopCloser(strings.NewReader(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("sk-ant-test"),
				WithHTTPClient(mockClient),
			)
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			resp, err := provider.Completion(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("Completion() error = nil, wantErr true")
				}
				return
			}

			if err != nil {
				t.Errorf("Completion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if resp == nil {
				t.Error("Completion() returned nil response")
				return
			}

			if tt.validate != nil {
				tt.validate(t, resp)
			}
		})
	}
}

// TestTransformRequest tests the transformRequest function
func TestTransformRequest(t *testing.T) {
	tests := []struct {
		name     string
		req      *warp.CompletionRequest
		validate func(*testing.T, *anthropicRequest)
		wantErr  bool
	}{
		{
			name: "simple message",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if result.Model != "claude-3-opus-20240229" {
					t.Errorf("Model = %v, want claude-3-opus-20240229", result.Model)
				}
				if len(result.Messages) != 1 {
					t.Errorf("len(Messages) = %v, want 1", len(result.Messages))
				}
				if result.MaxTokens != 1024 {
					t.Errorf("MaxTokens = %v, want 1024 (default)", result.MaxTokens)
				}
			},
		},
		{
			name: "with system message",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "system", Content: "You are helpful"},
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if result.System != "You are helpful" {
					t.Errorf("System = %v, want 'You are helpful'", result.System)
				}
				if len(result.Messages) != 1 {
					t.Errorf("len(Messages) = %v, want 1 (system message should be extracted)", len(result.Messages))
				}
				if result.Messages[0].Role != "user" {
					t.Errorf("Messages[0].Role = %v, want user", result.Messages[0].Role)
				}
			},
		},
		{
			name: "with max_tokens",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: intPtr(2048),
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if result.MaxTokens != 2048 {
					t.Errorf("MaxTokens = %v, want 2048", result.MaxTokens)
				}
			},
		},
		{
			name: "with stop sequences",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				Stop: []string{"\n\n", "END"},
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if len(result.StopSequences) != 2 {
					t.Errorf("len(StopSequences) = %v, want 2", len(result.StopSequences))
				}
				if result.StopSequences[0] != "\n\n" {
					t.Errorf("StopSequences[0] = %v, want \\n\\n", result.StopSequences[0])
				}
			},
		},
		{
			name: "with temperature and top_p",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				Temperature: float64Ptr(0.7),
				TopP:        float64Ptr(0.9),
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if result.Temperature == nil || *result.Temperature != 0.7 {
					t.Errorf("Temperature = %v, want 0.7", result.Temperature)
				}
				if result.TopP == nil || *result.TopP != 0.9 {
					t.Errorf("TopP = %v, want 0.9", result.TopP)
				}
			},
		},
		{
			name: "with multimodal content",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{
						Role: "user",
						Content: []warp.ContentPart{
							{Type: "text", Text: "What's in this image?"},
							{
								Type: "image_url",
								ImageURL: &warp.ImageURL{
									URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
								},
							},
						},
					},
				},
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if len(result.Messages) != 1 {
					t.Errorf("len(Messages) = %v, want 1", len(result.Messages))
					return
				}
				blocks, ok := result.Messages[0].Content.([]anthropicContentBlock)
				if !ok {
					t.Errorf("Content type = %T, want []anthropicContentBlock", result.Messages[0].Content)
					return
				}
				if len(blocks) != 2 {
					t.Errorf("len(blocks) = %v, want 2", len(blocks))
				}
			},
		},
		{
			name: "with tools",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "What's the weather?"},
				},
				Tools: []warp.Tool{
					{
						Type: "function",
						Function: warp.Function{
							Name:        "get_weather",
							Description: "Get the current weather",
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
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if len(result.Tools) != 1 {
					t.Errorf("len(Tools) = %v, want 1", len(result.Tools))
				}
				if result.Tools[0].Name != "get_weather" {
					t.Errorf("Tool name = %v, want get_weather", result.Tools[0].Name)
				}
			},
		},
		{
			name: "with tool choice",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				ToolChoice: &warp.ToolChoice{
					Type: "auto",
				},
			},
			validate: func(t *testing.T, result *anthropicRequest) {
				if result.ToolChoice == nil {
					t.Error("ToolChoice is nil")
					return
				}
				if result.ToolChoice.Type != "auto" {
					t.Errorf("ToolChoice.Type = %v, want auto", result.ToolChoice.Type)
				}
			},
		},
		// Note: Stream field is no longer tested here - it's set separately by
		// CompletionStream method, not by transformRequest
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := transformRequest(tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("transformRequest() error = nil, wantErr true")
				}
				return
			}

			if err != nil {
				t.Errorf("transformRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if result == nil {
				t.Error("transformRequest() returned nil")
				return
			}

			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

// TestMapStopReason tests the mapStopReason function
func TestMapStopReason(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"end_turn", "stop"},
		{"max_tokens", "length"},
		{"tool_use", "tool_calls"},
		{"stop_sequence", "stop"},
		{"unknown", "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := mapStopReason(tt.input)
			if got != tt.want {
				t.Errorf("mapStopReason(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

// TestCompletionStream tests the CompletionStream method
func TestCompletionStream(t *testing.T) {
	tests := []struct {
		name       string
		req        *warp.CompletionRequest
		mockEvents []string
		wantErr    bool
		validate   func(*testing.T, []*warp.CompletionChunk)
	}{
		{
			name: "successful stream",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockEvents: []string{
				`event: message_start
data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-3-opus-20240229","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}

`,
				`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

`,
				`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

`,
				`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}

`,
				`event: content_block_stop
data: {"type":"content_block_stop","index":0}

`,
				`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

`,
				`event: message_stop
data: {"type":"message_stop"}

`,
			},
			wantErr: false,
			validate: func(t *testing.T, chunks []*warp.CompletionChunk) {
				if len(chunks) < 3 {
					t.Errorf("len(chunks) = %v, want >= 3", len(chunks))
					return
				}
				// First chunk should have role
				if chunks[0].Choices[0].Delta.Role != "assistant" {
					t.Errorf("First chunk role = %v, want assistant", chunks[0].Choices[0].Delta.Role)
				}
				// Should have content chunks
				foundContent := false
				for _, chunk := range chunks {
					if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
						foundContent = true
						break
					}
				}
				if !foundContent {
					t.Error("No content chunks found")
				}
			},
		},
		{
			name: "stream with finish reason",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus-20240229",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockEvents: []string{
				`event: message_start
data: {"type":"message_start","message":{"id":"msg_456","type":"message","role":"assistant","model":"claude-3-opus-20240229","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}

`,
				`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

`,
				`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

`,
				`event: content_block_stop
data: {"type":"content_block_stop","index":0}

`,
				`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":1}}

`,
				`event: message_stop
data: {"type":"message_stop"}

`,
			},
			wantErr: false,
			validate: func(t *testing.T, chunks []*warp.CompletionChunk) {
				// Find chunk with finish reason
				foundFinishReason := false
				for _, chunk := range chunks {
					if len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != nil {
						if *chunk.Choices[0].FinishReason != "length" {
							t.Errorf("FinishReason = %v, want length", *chunk.Choices[0].FinishReason)
						}
						foundFinishReason = true
						break
					}
				}
				if !foundFinishReason {
					t.Error("No finish reason found in chunks")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock client that returns streaming events
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					// Verify headers
					if accept := req.Header.Get("Accept"); accept != "text/event-stream" {
						t.Errorf("Accept header = %v, want text/event-stream", accept)
					}

					// Build response body from mock events
					var body strings.Builder
					for _, event := range tt.mockEvents {
						body.WriteString(event)
					}

					return &http.Response{
						StatusCode: http.StatusOK,
						Body:       io.NopCloser(strings.NewReader(body.String())),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("sk-ant-test"),
				WithHTTPClient(mockClient),
			)
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			stream, err := provider.CompletionStream(context.Background(), tt.req)
			if tt.wantErr {
				if err == nil {
					t.Error("CompletionStream() error = nil, wantErr true")
				}
				return
			}

			if err != nil {
				t.Fatalf("CompletionStream() error = %v", err)
			}
			defer stream.Close()

			// Collect chunks
			var chunks []*warp.CompletionChunk
			for {
				chunk, err := stream.Recv()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("stream.Recv() error = %v", err)
				}
				chunks = append(chunks, chunk)
			}

			if len(chunks) == 0 {
				t.Error("No chunks received from stream")
			}

			if tt.validate != nil {
				tt.validate(t, chunks)
			}
		})
	}
}

// TestCompletionIntegration tests against the real Anthropic API (if key is available)
func TestCompletionIntegration(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(WithAPIKey(apiKey))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	req := &warp.CompletionRequest{
		Model: "claude-3-haiku-20240307",
		Messages: []warp.Message{
			{Role: "user", Content: "Say hello in one word"},
		},
		MaxTokens: intPtr(10),
	}

	resp, err := provider.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if resp == nil {
		t.Fatal("Completion() returned nil response")
	}

	if len(resp.Choices) == 0 {
		t.Fatal("Completion() returned no choices")
	}

	if resp.Choices[0].Message.Content == "" {
		t.Error("Completion() returned empty content")
	}

	t.Logf("Response: %s", resp.Choices[0].Message.Content)
}

// TestCompletionStreamIntegration tests streaming against the real Anthropic API
func TestCompletionStreamIntegration(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("ANTHROPIC_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(WithAPIKey(apiKey))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	req := &warp.CompletionRequest{
		Model: "claude-3-haiku-20240307",
		Messages: []warp.Message{
			{Role: "user", Content: "Count to 5"},
		},
		MaxTokens: intPtr(50),
	}

	stream, err := provider.CompletionStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CompletionStream() error = %v", err)
	}
	defer stream.Close()

	chunkCount := 0
	var content strings.Builder

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream.Recv() error = %v", err)
		}

		chunkCount++

		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	if chunkCount == 0 {
		t.Error("No chunks received from stream")
	}

	t.Logf("Received %d chunks, content: %s", chunkCount, content.String())
}
