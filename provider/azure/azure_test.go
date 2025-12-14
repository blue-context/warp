package azure

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
			errMsg:  "Azure API key is required",
		},
		{
			name: "missing endpoint",
			opts: []Option{
				WithAPIKey("test-key"),
			},
			wantErr: true,
			errMsg:  "Azure endpoint is required",
		},
		{
			name: "missing deployment",
			opts: []Option{
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com"),
			},
			wantErr: true,
			errMsg:  "Azure deployment name is required",
		},
		{
			name: "with all required options",
			opts: []Option{
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com"),
				WithDeployment("gpt-4-deployment"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com"),
				WithDeployment("gpt-4-deployment"),
				WithAPIVersion("2024-02-01"),
				WithHTTPClient(&mockHTTPClient{}),
			},
			wantErr: false,
		},
		{
			name: "endpoint with trailing slash",
			opts: []Option{
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com/"),
				WithDeployment("gpt-4-deployment"),
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
				return
			}

			// Verify trailing slash is removed
			if tt.name == "endpoint with trailing slash" {
				if strings.HasSuffix(provider.apiBase, "/") {
					t.Error("NewProvider() did not remove trailing slash from apiBase")
				}
			}
		})
	}
}

// TestProviderName tests the Name method
func TestProviderName(t *testing.T) {
	provider, err := NewProvider(
		WithAPIKey("test-key"),
		WithEndpoint("https://test.openai.azure.com"),
		WithDeployment("gpt-4-deployment"),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "azure" {
		t.Errorf("Name() = %v, want %v", got, "azure")
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider(
		WithAPIKey("test-key"),
		WithEndpoint("https://test.openai.azure.com"),
		WithDeployment("gpt-4-deployment"),
	)
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
		{"Embedding", caps.Embedding, true},
		{"ImageGeneration", caps.ImageGeneration, false},
		{"Transcription", caps.Transcription, false},
		{"Speech", caps.Speech, false},
		{"Moderation", caps.Moderation, false},
		{"FunctionCalling", caps.FunctionCalling, true},
		{"Vision", caps.Vision, true},
		{"JSON", caps.JSON, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value != tt.expected {
				t.Errorf("Supports().%s = %v, want %v", tt.name, tt.value, tt.expected)
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
		validate   func(*testing.T, *http.Request, *warp.CompletionResponse)
	}{
		{
			name: "successful completion",
			req: &warp.CompletionRequest{
				Model: "gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "gpt-4",
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
			validate: func(t *testing.T, req *http.Request, resp *warp.CompletionResponse) {
				// Verify URL structure
				if !strings.Contains(req.URL.String(), "/openai/deployments/gpt-4-deployment/chat/completions") {
					t.Errorf("URL = %v, want URL containing deployments path", req.URL.String())
				}
				if !strings.Contains(req.URL.String(), "api-version=2024-02-15-preview") {
					t.Errorf("URL = %v, want URL containing api-version parameter", req.URL.String())
				}

				// Verify response
				if resp.ID != "chatcmpl-123" {
					t.Errorf("ID = %v, want chatcmpl-123", resp.ID)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				if resp.Choices[0].Message.Content != "Hello! How can I help you?" {
					t.Errorf("Message.Content = %v", resp.Choices[0].Message.Content)
				}
			},
		},
		{
			name: "with temperature",
			req: &warp.CompletionRequest{
				Model: "gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				Temperature: float64Ptr(0.7),
			},
			mockResp: `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "gpt-4",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Response"
					},
					"finish_reason": "stop"
				}]
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "authentication error",
			req: &warp.CompletionRequest{
				Model: "gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Invalid API key",
					"type": "invalid_request_error"
				}
			}`,
			statusCode: http.StatusUnauthorized,
			wantErr:    true,
		},
		{
			name: "rate limit error",
			req: &warp.CompletionRequest{
				Model: "gpt-4",
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
			var capturedReq *http.Request

			// Create mock client
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					capturedReq = req

					// Verify request
					if req.Method != "POST" {
						t.Errorf("Method = %v, want POST", req.Method)
					}

					// Verify Azure-specific header
					if apiKey := req.Header.Get("api-key"); apiKey != "test-key" {
						t.Errorf("api-key = %v, want test-key", apiKey)
					}

					// Verify Authorization header is NOT set (Azure doesn't use it)
					if auth := req.Header.Get("Authorization"); auth != "" {
						t.Errorf("Authorization = %v, want empty (Azure uses api-key)", auth)
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
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com"),
				WithDeployment("gpt-4-deployment"),
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
				tt.validate(t, capturedReq, resp)
			}
		})
	}
}

// TestTransformMessages tests the transformMessages function
func TestTransformMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []warp.Message
		validate func(*testing.T, []map[string]any)
	}{
		{
			name: "simple text message",
			messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
			validate: func(t *testing.T, result []map[string]any) {
				if len(result) != 1 {
					t.Errorf("len(result) = %v, want 1", len(result))
					return
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
			name: "multimodal message",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{Type: "text", Text: "What's in this image?"},
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
					return
				}
				content, ok := result[0]["content"].([]map[string]any)
				if !ok {
					t.Errorf("content type = %T, want []map[string]any", result[0]["content"])
					return
				}
				if len(content) != 2 {
					t.Errorf("len(content) = %v, want 2", len(content))
				}
			},
		},
		{
			name: "message with tool calls",
			messages: []warp.Message{
				{
					Role: "assistant",
					ToolCalls: []warp.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: warp.FunctionCall{
								Name:      "get_weather",
								Arguments: `{"location": "San Francisco"}`,
							},
						},
					},
				},
			},
			validate: func(t *testing.T, result []map[string]any) {
				if len(result) != 1 {
					t.Errorf("len(result) = %v, want 1", len(result))
					return
				}
				toolCalls, ok := result[0]["tool_calls"]
				if !ok {
					t.Error("tool_calls not present in result")
				}
				if toolCalls == nil {
					t.Error("tool_calls is nil")
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

// TestCompletionStream tests the CompletionStream method
func TestCompletionStream(t *testing.T) {
	tests := []struct {
		name       string
		mockResp   string
		statusCode int
		wantErr    bool
		wantChunks int
	}{
		{
			name: "successful stream",
			mockResp: `data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantChunks: 3,
		},
		{
			name: "authentication error",
			mockResp: `{
				"error": {
					"message": "Invalid API key",
					"type": "invalid_request_error"
				}
			}`,
			statusCode: http.StatusUnauthorized,
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
					if accept := req.Header.Get("Accept"); accept != "text/event-stream" {
						t.Errorf("Accept = %v, want text/event-stream", accept)
					}

					// Verify Azure-specific header
					if apiKey := req.Header.Get("api-key"); apiKey != "test-key" {
						t.Errorf("api-key = %v, want test-key", apiKey)
					}

					// Verify URL structure
					if !strings.Contains(req.URL.String(), "/openai/deployments/") {
						t.Errorf("URL = %v, want URL containing /openai/deployments/", req.URL.String())
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
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com"),
				WithDeployment("gpt-4-deployment"),
				WithHTTPClient(mockClient),
			)
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
				Model: "gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			})

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
				t.Error("CompletionStream() returned nil stream")
				return
			}
			defer stream.Close()

			// Read chunks
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
		validate   func(*testing.T, *http.Request, *warp.EmbeddingResponse)
	}{
		{
			name: "successful embedding",
			req: &warp.EmbeddingRequest{
				Model: "text-embedding-ada-002",
				Input: "Hello, world!",
			},
			mockResp: `{
				"object": "list",
				"data": [{
					"object": "embedding",
					"embedding": [0.1, 0.2, 0.3],
					"index": 0
				}],
				"model": "text-embedding-ada-002",
				"usage": {
					"prompt_tokens": 5,
					"total_tokens": 5
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, req *http.Request, resp *warp.EmbeddingResponse) {
				// Verify URL structure
				if !strings.Contains(req.URL.String(), "/openai/deployments/") {
					t.Errorf("URL = %v, want URL containing /openai/deployments/", req.URL.String())
				}
				if !strings.Contains(req.URL.String(), "/embeddings") {
					t.Errorf("URL = %v, want URL containing /embeddings", req.URL.String())
				}

				// Verify response
				if len(resp.Data) != 1 {
					t.Errorf("len(Data) = %v, want 1", len(resp.Data))
				}
				if len(resp.Data[0].Embedding) != 3 {
					t.Errorf("len(Embedding) = %v, want 3", len(resp.Data[0].Embedding))
				}
			},
		},
		{
			name: "batch embedding",
			req: &warp.EmbeddingRequest{
				Model: "text-embedding-ada-002",
				Input: []string{"Hello", "World"},
			},
			mockResp: `{
				"object": "list",
				"data": [
					{
						"object": "embedding",
						"embedding": [0.1, 0.2],
						"index": 0
					},
					{
						"object": "embedding",
						"embedding": [0.3, 0.4],
						"index": 1
					}
				],
				"model": "text-embedding-ada-002",
				"usage": {
					"prompt_tokens": 10,
					"total_tokens": 10
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, req *http.Request, resp *warp.EmbeddingResponse) {
				if len(resp.Data) != 2 {
					t.Errorf("len(Data) = %v, want 2", len(resp.Data))
				}
			},
		},
		{
			name: "with dimensions",
			req: &warp.EmbeddingRequest{
				Model:      "text-embedding-3-small",
				Input:      "Hello",
				Dimensions: intPtr(1536),
			},
			mockResp: `{
				"object": "list",
				"data": [{
					"object": "embedding",
					"embedding": [0.1, 0.2],
					"index": 0
				}],
				"model": "text-embedding-3-small"
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "authentication error",
			req: &warp.EmbeddingRequest{
				Model: "text-embedding-ada-002",
				Input: "Hello",
			},
			mockResp: `{
				"error": {
					"message": "Invalid API key",
					"type": "invalid_request_error"
				}
			}`,
			statusCode: http.StatusUnauthorized,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var capturedReq *http.Request

			// Create mock client
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					capturedReq = req

					// Verify request
					if req.Method != "POST" {
						t.Errorf("Method = %v, want POST", req.Method)
					}

					// Verify Azure-specific header
					if apiKey := req.Header.Get("api-key"); apiKey != "test-key" {
						t.Errorf("api-key = %v, want test-key", apiKey)
					}

					// Read and verify request body
					body, _ := io.ReadAll(req.Body)
					var reqBody map[string]any
					if err := json.Unmarshal(body, &reqBody); err == nil {
						// Verify dimensions if present
						if tt.req.Dimensions != nil {
							if dims, ok := reqBody["dimensions"].(float64); !ok || int(dims) != *tt.req.Dimensions {
								t.Errorf("dimensions = %v, want %v", reqBody["dimensions"], *tt.req.Dimensions)
							}
						}
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
				WithAPIKey("test-key"),
				WithEndpoint("https://test.openai.azure.com"),
				WithDeployment("embedding-deployment"),
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
				t.Error("Embedding() returned nil response")
				return
			}

			if tt.validate != nil {
				tt.validate(t, capturedReq, resp)
			}
		})
	}
}

// TestStreamContextCancellation tests stream context cancellation
func TestStreamContextCancellation(t *testing.T) {
	mockResp := `data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

`

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
		WithAPIKey("test-key"),
		WithEndpoint("https://test.openai.azure.com"),
		WithDeployment("gpt-4-deployment"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Create cancellable context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
		Model: "gpt-4",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})
	if err != nil {
		t.Fatalf("CompletionStream() error = %v", err)
	}
	defer stream.Close()

	// Try to receive - should get context canceled error
	_, err = stream.Recv()
	if err == nil {
		t.Error("stream.Recv() error = nil, want context canceled error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("stream.Recv() error = %v, want context.Canceled", err)
	}
}

// TestHTTPClientError tests HTTP client error handling
func TestHTTPClientError(t *testing.T) {
	mockErr := errors.New("network error")

	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			return nil, mockErr
		},
	}

	provider, err := NewProvider(
		WithAPIKey("test-key"),
		WithEndpoint("https://test.openai.azure.com"),
		WithDeployment("gpt-4-deployment"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Test Completion
	_, err = provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "gpt-4",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})
	if err == nil {
		t.Error("Completion() error = nil, want error")
	}

	// Test CompletionStream
	_, err = provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "gpt-4",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})
	if err == nil {
		t.Error("CompletionStream() error = nil, want error")
	}

	// Test Embedding
	_, err = provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "text-embedding-ada-002",
		Input: "Hello",
	})
	if err == nil {
		t.Error("Embedding() error = nil, want error")
	}
}

// Integration tests (requires Azure OpenAI credentials)

func TestIntegrationCompletion(t *testing.T) {
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT")

	if apiKey == "" || endpoint == "" || deployment == "" {
		t.Skip("Azure OpenAI credentials not set, skipping integration test")
	}

	provider, err := NewProvider(
		WithAPIKey(apiKey),
		WithEndpoint(endpoint),
		WithDeployment(deployment),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "gpt-4",
		Messages: []warp.Message{
			{Role: "user", Content: "Say 'test successful' and nothing else"},
		},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if resp == nil {
		t.Fatal("Completion() returned nil response")
	}

	if len(resp.Choices) == 0 {
		t.Fatal("Completion() returned no choices")
	}

	t.Logf("Completion response: %s", resp.Choices[0].Message.Content)
}

func TestIntegrationCompletionStream(t *testing.T) {
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT")

	if apiKey == "" || endpoint == "" || deployment == "" {
		t.Skip("Azure OpenAI credentials not set, skipping integration test")
	}

	provider, err := NewProvider(
		WithAPIKey(apiKey),
		WithEndpoint(endpoint),
		WithDeployment(deployment),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "gpt-4",
		Messages: []warp.Message{
			{Role: "user", Content: "Count from 1 to 5"},
		},
	})
	if err != nil {
		t.Fatalf("CompletionStream() error = %v", err)
	}
	defer stream.Close()

	var content bytes.Buffer
	chunks := 0

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("stream.Recv() error = %v", err)
		}

		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
		chunks++
	}

	if chunks == 0 {
		t.Error("stream.Recv() received no chunks")
	}

	t.Logf("Stream received %d chunks: %s", chunks, content.String())
}

func TestIntegrationEmbedding(t *testing.T) {
	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	embeddingDeployment := os.Getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

	if apiKey == "" || endpoint == "" || embeddingDeployment == "" {
		t.Skip("Azure OpenAI embedding credentials not set, skipping integration test")
	}

	provider, err := NewProvider(
		WithAPIKey(apiKey),
		WithEndpoint(endpoint),
		WithDeployment(embeddingDeployment),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "text-embedding-ada-002",
		Input: "Hello, world!",
	})
	if err != nil {
		t.Fatalf("Embedding() error = %v", err)
	}

	if resp == nil {
		t.Fatal("Embedding() returned nil response")
	}

	if len(resp.Data) == 0 {
		t.Fatal("Embedding() returned no data")
	}

	if len(resp.Data[0].Embedding) == 0 {
		t.Fatal("Embedding() returned empty embedding vector")
	}

	t.Logf("Embedding dimension: %d", len(resp.Data[0].Embedding))
}

// Helper functions

func float64Ptr(v float64) *float64 {
	return &v
}

func intPtr(v int) *int {
	return &v
}
