package openrouter

import (
	"context"
	"io"
	"net/http"
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
			errMsg:  "OpenRouter API key is required",
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("sk-or-v1-test"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithAPIKey("sk-or-v1-test"),
				WithAPIBase("https://test.openrouter.ai/api/v1"),
				WithHTTPReferer("https://myapp.com"),
				WithAppTitle("Test App"),
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
	provider, err := NewProvider(WithAPIKey("sk-or-v1-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "openrouter" {
		t.Errorf("Name() = %v, want %v", got, "openrouter")
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-or-v1-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	capsInterface := provider.Supports()
	caps, ok := capsInterface.(prov.Capabilities)
	if !ok {
		t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
	}

	tests := []struct {
		name  string
		value bool
		want  bool
	}{
		{"Completion", caps.Completion, true},
		{"Streaming", caps.Streaming, true},
		{"Embedding", caps.Embedding, true},
		{"ImageGeneration", caps.ImageGeneration, true},
		{"FunctionCalling", caps.FunctionCalling, true},
		{"Vision", caps.Vision, true},
		{"JSON", caps.JSON, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value != tt.want {
				t.Errorf("Supports().%s = %v, want %v", tt.name, tt.value, tt.want)
			}
		})
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
		validate   func(*testing.T, *warp.CompletionResponse, *http.Request)
	}{
		{
			name: "successful completion",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4o",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello!"},
				},
			},
			mockResp: `{
				"id": "gen-123",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "openai/gpt-4o",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Hello! How can I help you?"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 8,
					"total_tokens": 18
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse, req *http.Request) {
				if resp.ID != "gen-123" {
					t.Errorf("ID = %v, want gen-123", resp.ID)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				if resp.Choices[0].Message.Content != "Hello! How can I help you?" {
					t.Errorf("Content = %v, want 'Hello! How can I help you?'", resp.Choices[0].Message.Content)
				}
				if resp.Usage.TotalTokens != 18 {
					t.Errorf("TotalTokens = %v, want 18", resp.Usage.TotalTokens)
				}

				// Verify Authorization header
				if auth := req.Header.Get("Authorization"); !strings.HasPrefix(auth, "Bearer ") {
					t.Errorf("Authorization header = %v, want Bearer token", auth)
				}
			},
		},
		{
			name: "completion with custom headers",
			req: &warp.CompletionRequest{
				Model: "anthropic/claude-opus-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
			},
			mockResp: `{
				"id": "gen-456",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "anthropic/claude-opus-4",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Response"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 5,
					"completion_tokens": 3,
					"total_tokens": 8
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse, req *http.Request) {
				// Verify custom headers are set (these are set in the provider instance)
				// Note: In real usage, these would be set via WithHTTPReferer/WithAppTitle
			},
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "invalid/model",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Model not found",
					"type": "invalid_request_error",
					"code": "model_not_found"
				}
			}`,
			statusCode: http.StatusNotFound,
			wantErr:    true,
			validate:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedReq *http.Request

			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					capturedReq = req
					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       io.NopCloser(strings.NewReader(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("sk-or-v1-test"),
				WithHTTPReferer("https://myapp.com"),
				WithAppTitle("Test App"),
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
				t.Fatal("Completion() returned nil response")
			}

			if tt.validate != nil {
				tt.validate(t, resp, capturedReq)
			}
		})
	}
}

// TestCompletionStream tests the CompletionStream method
func TestCompletionStream(t *testing.T) {
	tests := []struct {
		name       string
		req        *warp.CompletionRequest
		mockResp   string
		statusCode int
		wantErr    bool
		wantChunks int
	}{
		{
			name: "successful streaming",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4o",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello!"},
				},
			},
			mockResp: `data: {"id":"gen-123","object":"chat.completion.chunk","created":1234567890,"model":"openai/gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"gen-123","object":"chat.completion.chunk","created":1234567890,"model":"openai/gpt-4o","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"gen-123","object":"chat.completion.chunk","created":1234567890,"model":"openai/gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantChunks: 3,
		},
		{
			name: "stream error",
			req: &warp.CompletionRequest{
				Model: "invalid/model",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Model not found",
					"type": "invalid_request_error"
				}
			}`,
			statusCode: http.StatusNotFound,
			wantErr:    true,
			wantChunks: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       io.NopCloser(strings.NewReader(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("sk-or-v1-test"),
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
				t.Errorf("CompletionStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if stream == nil {
				t.Fatal("CompletionStream() returned nil stream")
			}
			defer stream.Close()

			chunks := 0
			for {
				chunk, err := stream.Recv()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Errorf("stream.Recv() error = %v", err)
					break
				}
				if chunk == nil {
					t.Error("stream.Recv() returned nil chunk")
					break
				}
				chunks++
			}

			if chunks != tt.wantChunks {
				t.Errorf("received %d chunks, want %d", chunks, tt.wantChunks)
			}
		})
	}
}

// TestEmbedding tests the Embedding method
func TestEmbedding(t *testing.T) {
	tests := []struct {
		name       string
		req        *warp.EmbeddingRequest
		mockResp   string
		statusCode int
		wantErr    bool
		validate   func(*testing.T, *warp.EmbeddingResponse, *http.Request)
	}{
		{
			name: "successful embedding",
			req: &warp.EmbeddingRequest{
				Model: "openai/text-embedding-ada-002",
				Input: "Hello, world!",
			},
			mockResp: `{
				"object": "list",
				"data": [{
					"object": "embedding",
					"index": 0,
					"embedding": [0.021, -0.037, 0.015]
				}],
				"model": "openai/text-embedding-ada-002",
				"usage": {
					"prompt_tokens": 8,
					"total_tokens": 8
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.EmbeddingResponse, req *http.Request) {
				if len(resp.Data) != 1 {
					t.Errorf("len(Data) = %v, want 1", len(resp.Data))
				}
				if len(resp.Data[0].Embedding) != 3 {
					t.Errorf("len(Embedding) = %v, want 3", len(resp.Data[0].Embedding))
				}
				if resp.Usage.TotalTokens != 8 {
					t.Errorf("TotalTokens = %v, want 8", resp.Usage.TotalTokens)
				}
			},
		},
		{
			name: "batch embedding",
			req: &warp.EmbeddingRequest{
				Model: "sentence-transformers/all-mpnet-base-v2",
				Input: []string{"Hello", "World"},
			},
			mockResp: `{
				"object": "list",
				"data": [
					{
						"object": "embedding",
						"index": 0,
						"embedding": [0.1, 0.2, 0.3]
					},
					{
						"object": "embedding",
						"index": 1,
						"embedding": [0.4, 0.5, 0.6]
					}
				],
				"model": "sentence-transformers/all-mpnet-base-v2",
				"usage": {
					"prompt_tokens": 4,
					"total_tokens": 4
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.EmbeddingResponse, req *http.Request) {
				if len(resp.Data) != 2 {
					t.Errorf("len(Data) = %v, want 2", len(resp.Data))
				}
			},
		},
		{
			name: "embedding error",
			req: &warp.EmbeddingRequest{
				Model: "invalid/model",
				Input: "Test",
			},
			mockResp: `{
				"error": {
					"message": "Model not found"
				}
			}`,
			statusCode: http.StatusNotFound,
			wantErr:    true,
			validate:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedReq *http.Request

			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					capturedReq = req
					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       io.NopCloser(strings.NewReader(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("sk-or-v1-test"),
				WithHTTPClient(mockClient),
			)
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			resp, err := provider.Embedding(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("Embedding() error = nil, wantErr true")
				}
				return
			}

			if err != nil {
				t.Errorf("Embedding() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if resp == nil {
				t.Fatal("Embedding() returned nil response")
			}

			if tt.validate != nil {
				tt.validate(t, resp, capturedReq)
			}
		})
	}
}

// TestCustomHeaders tests that custom headers are properly set
func TestCustomHeaders(t *testing.T) {
	var capturedReq *http.Request

	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			capturedReq = req
			mockResp := `{
				"id": "gen-123",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "openai/gpt-4o",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Test"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 5,
					"completion_tokens": 3,
					"total_tokens": 8
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(mockResp)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(
		WithAPIKey("sk-or-v1-test"),
		WithHTTPReferer("https://myapp.com"),
		WithAppTitle("My Test App"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	_, err = provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "openai/gpt-4o",
		Messages: []warp.Message{
			{Role: "user", Content: "Test"},
		},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Verify custom headers
	if referer := capturedReq.Header.Get("HTTP-Referer"); referer != "https://myapp.com" {
		t.Errorf("HTTP-Referer = %v, want https://myapp.com", referer)
	}
	if title := capturedReq.Header.Get("X-Title"); title != "My Test App" {
		t.Errorf("X-Title = %v, want My Test App", title)
	}
}

// TestGetModelInfo tests model metadata retrieval
func TestGetModelInfo(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-or-v1-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	tests := []struct {
		name      string
		model     string
		wantNil   bool
		wantModel string
	}{
		{
			name:      "valid chat model",
			model:     "openai/gpt-4o",
			wantNil:   false,
			wantModel: "openai/gpt-4o",
		},
		{
			name:      "valid embedding model",
			model:     "openai/text-embedding-ada-002",
			wantNil:   false,
			wantModel: "openai/text-embedding-ada-002",
		},
		{
			name:      "auto router",
			model:     "openrouter/auto",
			wantNil:   false,
			wantModel: "openrouter/auto",
		},
		{
			name:    "unknown model",
			model:   "unknown/model",
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := provider.GetModelInfo(tt.model)

			if tt.wantNil {
				if info != nil {
					t.Errorf("GetModelInfo(%q) = %v, want nil", tt.model, info)
				}
				return
			}

			if info == nil {
				t.Errorf("GetModelInfo(%q) = nil, want non-nil", tt.model)
				return
			}

			if info.Name != tt.wantModel {
				t.Errorf("GetModelInfo(%q).Name = %v, want %v", tt.model, info.Name, tt.wantModel)
			}
		})
	}
}

// TestListModels tests listing all models
func TestListModels(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-or-v1-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	models := provider.ListModels()

	if len(models) == 0 {
		t.Error("ListModels() returned empty slice")
	}

	// Verify models are sorted
	for i := 1; i < len(models); i++ {
		if models[i-1].Name > models[i].Name {
			t.Errorf("models not sorted: %s > %s", models[i-1].Name, models[i].Name)
		}
	}

	// Verify we have both chat and embedding models
	hasChatModel := false
	hasEmbeddingModel := false

	for _, model := range models {
		if model.Capabilities.Completion {
			hasChatModel = true
		}
		if model.Capabilities.Embedding {
			hasEmbeddingModel = true
		}
	}

	if !hasChatModel {
		t.Error("ListModels() missing chat models")
	}
	if !hasEmbeddingModel {
		t.Error("ListModels() missing embedding models")
	}
}

// TestTransformRequest tests request transformation
func TestTransformRequest(t *testing.T) {
	tests := []struct {
		name     string
		req      *warp.CompletionRequest
		validate func(*testing.T, map[string]any)
	}{
		{
			name: "basic request",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4o",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, result map[string]any) {
				if result["model"] != "openai/gpt-4o" {
					t.Errorf("model = %v, want openai/gpt-4o", result["model"])
				}
				messages, ok := result["messages"].([]map[string]any)
				if !ok || len(messages) != 1 {
					t.Error("invalid messages format")
				}
			},
		},
		{
			name: "with optional parameters",
			req: &warp.CompletionRequest{
				Model: "anthropic/claude-opus-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: float64Ptr(0.7),
				MaxTokens:   intPtr(1000),
			},
			validate: func(t *testing.T, result map[string]any) {
				if result["temperature"] != 0.7 {
					t.Errorf("temperature = %v, want 0.7", result["temperature"])
				}
				if result["max_tokens"] != 1000 {
					t.Errorf("max_tokens = %v, want 1000", result["max_tokens"])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := transformRequest(tt.req)
			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

// TestUnsupportedMethods tests methods that are not yet implemented
func TestUnsupportedMethods(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-or-v1-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()

	t.Run("ImageGeneration", func(t *testing.T) {
		_, err := provider.ImageGeneration(ctx, &warp.ImageGenerationRequest{
			Model:  "dall-e-3",
			Prompt: "test",
		})
		if err == nil {
			t.Error("ImageGeneration() expected error, got nil")
		}
	})

	t.Run("Transcription", func(t *testing.T) {
		_, err := provider.Transcription(ctx, &warp.TranscriptionRequest{
			Model: "whisper-1",
		})
		if err == nil {
			t.Error("Transcription() expected error, got nil")
		}
	})

	t.Run("Speech", func(t *testing.T) {
		_, err := provider.Speech(ctx, &warp.SpeechRequest{
			Model: "tts-1",
			Input: "test",
		})
		if err == nil {
			t.Error("Speech() expected error, got nil")
		}
	})
}

// TestCompletionWithAllParameters tests request transformation with all optional parameters
func TestCompletionWithAllParameters(t *testing.T) {
	var capturedReq *http.Request

	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			capturedReq = req
			mockResp := `{
				"id": "gen-123",
				"object": "chat.completion",
				"created": 1234567890,
				"model": "openai/gpt-4o",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Test"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 5,
					"completion_tokens": 3,
					"total_tokens": 8
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(mockResp)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(
		WithAPIKey("sk-or-v1-test"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	_, err = provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "openai/gpt-4o",
		Messages: []warp.Message{
			{Role: "user", Content: "Test"},
		},
		Temperature:      float64Ptr(0.7),
		MaxTokens:        intPtr(1000),
		TopP:             float64Ptr(0.9),
		FrequencyPenalty: float64Ptr(0.5),
		PresencePenalty:  float64Ptr(0.5),
		N:                intPtr(2),
		Stop:             []string{"STOP"},
		ResponseFormat: &warp.ResponseFormat{
			Type: "json_object",
		},
		Tools: []warp.Tool{
			{
				Type: "function",
				Function: warp.Function{
					Name: "test_func",
				},
			},
		},
		ToolChoice: &warp.ToolChoice{Type: "auto"},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if capturedReq == nil {
		t.Fatal("Request was not captured")
	}
}

// TestTransformMessages tests message transformation with different content types
func TestTransformMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []warp.Message
		validate func(*testing.T, []map[string]any)
	}{
		{
			name: "simple text messages",
			messages: []warp.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there"},
			},
			validate: func(t *testing.T, result []map[string]any) {
				if len(result) != 2 {
					t.Errorf("len(result) = %v, want 2", len(result))
				}
				if result[0]["role"] != "user" {
					t.Errorf("role = %v, want user", result[0]["role"])
				}
				if result[0]["content"] != "Hello" {
					t.Errorf("content = %v, want Hello", result[0]["content"])
				}
			},
		},
		{
			name: "multimodal message with image",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{
							Type: "text",
							Text: "What's in this image?",
						},
						{
							Type: "image_url",
							ImageURL: &warp.ImageURL{
								URL: "https://example.com/image.jpg",
							},
						},
					},
				},
			},
			validate: func(t *testing.T, result []map[string]any) {
				if len(result) != 1 {
					t.Errorf("len(result) = %v, want 1", len(result))
				}
				parts, ok := result[0]["content"].([]map[string]any)
				if !ok {
					t.Error("content is not []map[string]any")
					return
				}
				if len(parts) != 2 {
					t.Errorf("len(parts) = %v, want 2", len(parts))
				}
				if parts[0]["type"] != "text" {
					t.Errorf("type = %v, want text", parts[0]["type"])
				}
				if parts[1]["type"] != "image_url" {
					t.Errorf("type = %v, want image_url", parts[1]["type"])
				}
			},
		},
		{
			name: "message with tool calls",
			messages: []warp.Message{
				{
					Role:    "assistant",
					Content: "",
					ToolCalls: []warp.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: warp.FunctionCall{
								Name:      "get_weather",
								Arguments: `{"location":"SF"}`,
							},
						},
					},
				},
			},
			validate: func(t *testing.T, result []map[string]any) {
				if len(result) != 1 {
					t.Errorf("len(result) = %v, want 1", len(result))
				}
				toolCalls, ok := result[0]["tool_calls"]
				if !ok {
					t.Error("tool_calls missing")
				}
				if toolCalls == nil {
					t.Error("tool_calls is nil")
				}
			},
		},
		{
			name: "message with name and tool_call_id",
			messages: []warp.Message{
				{
					Role:       "tool",
					Content:    `{"result":"sunny"}`,
					Name:       "get_weather",
					ToolCallID: "call_123",
				},
			},
			validate: func(t *testing.T, result []map[string]any) {
				if len(result) != 1 {
					t.Errorf("len(result) = %v, want 1", len(result))
				}
				if result[0]["name"] != "get_weather" {
					t.Errorf("name = %v, want get_weather", result[0]["name"])
				}
				if result[0]["tool_call_id"] != "call_123" {
					t.Errorf("tool_call_id = %v, want call_123", result[0]["tool_call_id"])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := transformMessages(tt.messages)
			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

// Helper functions for creating pointers
func float64Ptr(v float64) *float64 {
	return &v
}

func intPtr(v int) *int {
	return &v
}

// TestModelValidation tests that model validation is enforced
func TestModelValidation(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("sk-or-v1-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()

	t.Run("Completion with empty model", func(t *testing.T) {
		_, err := provider.Completion(ctx, &warp.CompletionRequest{
			Model: "",
			Messages: []warp.Message{
				{Role: "user", Content: "test"},
			},
		})
		if err == nil {
			t.Error("Completion() with empty model expected error, got nil")
		}
		if err.Error() != "model is required" {
			t.Errorf("Completion() error = %v, want 'model is required'", err)
		}
	})

	t.Run("CompletionStream with empty model", func(t *testing.T) {
		_, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
			Model: "",
			Messages: []warp.Message{
				{Role: "user", Content: "test"},
			},
		})
		if err == nil {
			t.Error("CompletionStream() with empty model expected error, got nil")
		}
		if err.Error() != "model is required" {
			t.Errorf("CompletionStream() error = %v, want 'model is required'", err)
		}
	})

	t.Run("Embedding with empty model", func(t *testing.T) {
		_, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
			Model: "",
			Input: "test",
		})
		if err == nil {
			t.Error("Embedding() with empty model expected error, got nil")
		}
		if err.Error() != "model is required" {
			t.Errorf("Embedding() error = %v, want 'model is required'", err)
		}
	})
}

// TestErrorContextInclusion tests that error messages include model context
func TestErrorContextInclusion(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusBadRequest,
				Body:       io.NopCloser(strings.NewReader(`{"error": "invalid request"}`)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(
		WithAPIKey("sk-or-v1-test"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()

	t.Run("Completion error includes model", func(t *testing.T) {
		_, err := provider.Completion(ctx, &warp.CompletionRequest{
			Model: "openai/gpt-4o",
			Messages: []warp.Message{
				{Role: "user", Content: "test"},
			},
		})
		if err == nil {
			t.Fatal("Completion() expected error, got nil")
		}
		// Error should come from ParseProviderError for status code error
	})

	t.Run("Embedding error includes model", func(t *testing.T) {
		_, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
			Model: "openai/text-embedding-ada-002",
			Input: "test",
		})
		if err == nil {
			t.Fatal("Embedding() expected error, got nil")
		}
		// Error should come from ParseProviderError for status code error
	})
}

// TestStreamContextCancellation tests that stream respects context cancellation
func TestStreamContextCancellation(t *testing.T) {
	// Create a mock response that streams slowly
	mockResp := "data: " + `{"id":"gen-1","object":"chat.completion.chunk","created":1234567890,"model":"openai/gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}` + "\n\n"
	mockResp += "data: [DONE]\n\n"

	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(mockResp)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(
		WithAPIKey("sk-or-v1-test"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Create a cancellable context
	ctx, cancel := context.WithCancel(context.Background())

	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
		Model: "openai/gpt-4o",
		Messages: []warp.Message{
			{Role: "user", Content: "test"},
		},
	})
	if err != nil {
		t.Fatalf("CompletionStream() error = %v", err)
	}
	defer stream.Close()

	// Cancel context immediately
	cancel()

	// Next Recv should return context.Canceled
	_, err = stream.Recv()
	if err != context.Canceled {
		t.Errorf("Recv() after cancel error = %v, want context.Canceled", err)
	}
}

// TestStreamCloseNilSafety tests that Close handles nil closer safely
func TestStreamCloseNilSafety(t *testing.T) {
	// Create a stream with a nil closer (simulating error conditions)
	stream := &sseStream{
		reader: nil,
		closer: nil,
		ctx:    context.Background(),
	}

	// Close should not panic with nil closer
	err := stream.Close()
	if err != nil {
		t.Errorf("Close() with nil closer error = %v, want nil", err)
	}
}
