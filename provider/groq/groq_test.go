package groq

import (
	"bytes"
	"context"
	"errors"
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
			errMsg:  "Groq API key is required",
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("gsk-test"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithAPIKey("gsk-test"),
				WithAPIBase("https://custom.example.com/v1"),
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
	provider, err := NewProvider(WithAPIKey("gsk-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "groq" {
		t.Errorf("Name() = %v, want %v", got, "groq")
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("gsk-test"))
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
		want  bool
		value bool
	}{
		{"Completion", true, caps.Completion},
		{"Streaming", true, caps.Streaming},
		{"Embedding", false, caps.Embedding},
		{"ImageGeneration", false, caps.ImageGeneration},
		{"Transcription", false, caps.Transcription},
		{"Speech", false, caps.Speech},
		{"Moderation", false, caps.Moderation},
		{"FunctionCalling", true, caps.FunctionCalling},
		{"Vision", false, caps.Vision},
		{"JSON", true, caps.JSON},
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
		validate   func(*testing.T, *warp.CompletionResponse)
	}{
		{
			name: "successful completion",
			req: &warp.CompletionRequest{
				Model: "llama3-70b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "llama3-70b-8192",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Hello! How can I help you today?"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 20,
					"total_tokens": 30
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.ID != "chatcmpl-123" {
					t.Errorf("ID = %v, want %v", resp.ID, "chatcmpl-123")
				}
				if resp.Model != "llama3-70b-8192" {
					t.Errorf("Model = %v, want %v", resp.Model, "llama3-70b-8192")
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want %v", len(resp.Choices), 1)
				}
				content, ok := resp.Choices[0].Message.Content.(string)
				if !ok || content != "Hello! How can I help you today?" {
					t.Errorf("Content = %v, want %v", content, "Hello! How can I help you today?")
				}
			},
		},
		{
			name: "with temperature and max_tokens",
			req: &warp.CompletionRequest{
				Model: "llama3-8b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
			},
			mockResp: `{
				"id": "chatcmpl-456",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "llama3-8b-8192",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Test response"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 5,
					"completion_tokens": 10,
					"total_tokens": 15
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "llama3-70b-8192",
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
			name: "network error",
			req: &warp.CompletionRequest{
				Model: "llama3-70b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			statusCode: 0,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					// Verify headers
					if auth := req.Header.Get("Authorization"); !strings.HasPrefix(auth, "Bearer ") {
						t.Errorf("Authorization header = %v, want Bearer token", auth)
					}
					if ct := req.Header.Get("Content-Type"); ct != "application/json" {
						t.Errorf("Content-Type = %v, want application/json", ct)
					}

					// Verify URL
					expectedURL := "https://api.groq.com/openai/v1/chat/completions"
					if req.URL.String() != expectedURL {
						t.Errorf("URL = %v, want %v", req.URL.String(), expectedURL)
					}

					// Return network error if statusCode is 0
					if tt.statusCode == 0 {
						return nil, errors.New("network error")
					}

					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       io.NopCloser(bytes.NewBufferString(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("gsk-test"),
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
			name: "successful stream",
			req: &warp.CompletionRequest{
				Model: "llama3-70b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"llama3-70b-8192","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"llama3-70b-8192","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"llama3-70b-8192","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantChunks: 3,
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "llama3-70b-8192",
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: tt.statusCode,
						Body:       io.NopCloser(bytes.NewBufferString(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(
				WithAPIKey("gsk-test"),
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
				t.Error("CompletionStream() returned nil stream")
				return
			}
			defer stream.Close()

			// Read all chunks
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
	provider, err := NewProvider(WithAPIKey("gsk-test"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "text-embedding-ada-002",
		Input: "test",
	})

	if err == nil {
		t.Error("Embedding() error = nil, want error")
	}
	if resp != nil {
		t.Error("Embedding() returned non-nil response, want nil")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Errorf("Embedding() error = %v, want error containing 'not supported'", err)
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
				Model: "llama3-70b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, m map[string]any) {
				if m["model"] != "llama3-70b-8192" {
					t.Errorf("model = %v, want llama3-70b-8192", m["model"])
				}
				messages, ok := m["messages"].([]map[string]any)
				if !ok || len(messages) != 1 {
					t.Errorf("messages invalid or wrong length")
				}
			},
		},
		{
			name: "with all parameters",
			req: &warp.CompletionRequest{
				Model: "llama3-8b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
				Temperature:      floatPtr(0.7),
				MaxTokens:        intPtr(100),
				TopP:             floatPtr(0.9),
				FrequencyPenalty: floatPtr(0.5),
				PresencePenalty:  floatPtr(0.5),
				Stop:             []string{"STOP"},
				N:                intPtr(1),
			},
			validate: func(t *testing.T, m map[string]any) {
				if m["temperature"].(float64) != 0.7 {
					t.Errorf("temperature = %v, want 0.7", m["temperature"])
				}
				if m["max_tokens"].(int) != 100 {
					t.Errorf("max_tokens = %v, want 100", m["max_tokens"])
				}
				if m["top_p"].(float64) != 0.9 {
					t.Errorf("top_p = %v, want 0.9", m["top_p"])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := transformRequest(tt.req)

			if result == nil {
				t.Fatal("transformRequest returned nil")
			}

			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

// Helper functions for creating pointers
func floatPtr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

// TestConcurrentRequests tests thread safety
func TestConcurrentRequests(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			resp := `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "llama3-70b-8192",
				"choices": [{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Hello"
					},
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 10,
					"total_tokens": 20
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(resp)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(
		WithAPIKey("gsk-test"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	// Run 10 concurrent requests
	const numRequests = 10
	errChan := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			_, err := provider.Completion(context.Background(), &warp.CompletionRequest{
				Model: "llama3-70b-8192",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			})
			errChan <- err
		}()
	}

	// Collect results
	for i := 0; i < numRequests; i++ {
		if err := <-errChan; err != nil {
			t.Errorf("Concurrent request %d failed: %v", i, err)
		}
	}
}
