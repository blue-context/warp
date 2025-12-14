package vllmsemanticrouter

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

// Helper functions for creating pointers
func floatPtr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

// TestNewProvider tests the NewProvider constructor
func TestNewProvider(t *testing.T) {
	tests := []struct {
		name     string
		opts     []Option
		validate func(*testing.T, *Provider)
	}{
		{
			name: "default configuration",
			opts: []Option{},
			validate: func(t *testing.T, p *Provider) {
				if p.baseURL != "http://localhost:8801" {
					t.Errorf("baseURL = %v, want http://localhost:8801", p.baseURL)
				}
				if p.classificationURL != "http://localhost:8080" {
					t.Errorf("classificationURL = %v, want http://localhost:8080", p.classificationURL)
				}
				if p.apiKey != "" {
					t.Errorf("apiKey = %v, want empty string", p.apiKey)
				}
			},
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("test-key"),
			},
			validate: func(t *testing.T, p *Provider) {
				if p.apiKey != "test-key" {
					t.Errorf("apiKey = %v, want test-key", p.apiKey)
				}
			},
		},
		{
			name: "with custom URLs",
			opts: []Option{
				WithBaseURL("http://router.example.com:8801"),
				WithClassificationURL("http://classifier.example.com:8080"),
			},
			validate: func(t *testing.T, p *Provider) {
				if p.baseURL != "http://router.example.com:8801" {
					t.Errorf("baseURL = %v, want http://router.example.com:8801", p.baseURL)
				}
				if p.classificationURL != "http://classifier.example.com:8080" {
					t.Errorf("classificationURL = %v, want http://classifier.example.com:8080", p.classificationURL)
				}
			},
		},
		{
			name: "with all options",
			opts: []Option{
				WithAPIKey("test-key"),
				WithBaseURL("http://custom.example.com:8801"),
				WithClassificationURL("http://custom.example.com:8080"),
				WithHTTPClient(&mockHTTPClient{}),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewProvider(tt.opts...)

			if err != nil {
				t.Errorf("NewProvider() error = %v, wantErr false", err)
				return
			}

			if provider == nil {
				t.Error("NewProvider() returned nil provider")
				return
			}

			if tt.validate != nil {
				tt.validate(t, provider)
			}
		})
	}
}

// TestProviderName tests the Name method
func TestProviderName(t *testing.T) {
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "vllm-semantic-router" {
		t.Errorf("Name() = %v, want vllm-semantic-router", got)
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider()
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
			name: "successful completion with auto model",
			req: &warp.CompletionRequest{
				Model: "auto",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "auto",
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
					t.Errorf("ID = %v, want chatcmpl-123", resp.ID)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				if resp.Model != "auto" {
					t.Errorf("Model = %v, want auto", resp.Model)
				}
			},
		},
		{
			name: "with parameters",
			req: &warp.CompletionRequest{
				Model: "auto",
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
				"model": "auto",
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
			name: "nil context",
			req: &warp.CompletionRequest{
				Model:    "auto",
				Messages: []warp.Message{{Role: "user", Content: "Hello"}},
			},
			statusCode: 0,
			wantErr:    true,
		},
		{
			name:       "nil request",
			req:        nil,
			statusCode: 0,
			wantErr:    true,
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "auto",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Invalid request",
					"type": "invalid_request_error"
				}
			}`,
			statusCode: http.StatusBadRequest,
			wantErr:    true,
		},
		{
			name: "network error",
			req: &warp.CompletionRequest{
				Model: "auto",
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
			// Handle nil context and nil request tests
			if tt.name == "nil context" {
				provider, _ := NewProvider()
				_, err := provider.Completion(nil, tt.req)
				if err == nil {
					t.Error("Completion() with nil context error = nil, wantErr true")
				}
				return
			}

			if tt.name == "nil request" {
				provider, _ := NewProvider()
				_, err := provider.Completion(context.Background(), nil)
				if err == nil {
					t.Error("Completion() with nil request error = nil, wantErr true")
				}
				return
			}

			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					// Verify headers
					if ct := req.Header.Get("Content-Type"); ct != "application/json" {
						t.Errorf("Content-Type = %v, want application/json", ct)
					}

					// Verify URL
					expectedURL := "http://localhost:8801/v1/chat/completions"
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

// TestCompletionWithAPIKey tests that API key is included when set
func TestCompletionWithAPIKey(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			// Verify authorization header
			auth := req.Header.Get("Authorization")
			if !strings.HasPrefix(auth, "Bearer ") {
				t.Errorf("Authorization header = %v, want Bearer token", auth)
			}
			if auth != "Bearer test-key" {
				t.Errorf("Authorization = %v, want Bearer test-key", auth)
			}

			resp := `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "auto",
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
		WithAPIKey("test-key"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	_, err = provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "auto",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err != nil {
		t.Errorf("Completion() error = %v", err)
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
			name: "successful stream with auto model",
			req: &warp.CompletionRequest{
				Model: "auto",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"auto","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"auto","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"auto","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantChunks: 3,
		},
		{
			name: "nil context",
			req: &warp.CompletionRequest{
				Model:    "auto",
				Messages: []warp.Message{{Role: "user", Content: "Hello"}},
			},
			statusCode: 0,
			wantErr:    true,
		},
		{
			name:       "nil request",
			req:        nil,
			statusCode: 0,
			wantErr:    true,
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "auto",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": {
					"message": "Invalid request",
					"type": "invalid_request_error"
				}
			}`,
			statusCode: http.StatusBadRequest,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Handle nil context and nil request tests
			if tt.name == "nil context" {
				provider, _ := NewProvider()
				_, err := provider.CompletionStream(nil, tt.req)
				if err == nil {
					t.Error("CompletionStream() with nil context error = nil, wantErr true")
				}
				return
			}

			if tt.name == "nil request" {
				provider, _ := NewProvider()
				_, err := provider.CompletionStream(context.Background(), nil)
				if err == nil {
					t.Error("CompletionStream() with nil request error = nil, wantErr true")
				}
				return
			}

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

// TestUnsupportedMethods tests unsupported provider methods
func TestUnsupportedMethods(t *testing.T) {
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()

	t.Run("Embedding", func(t *testing.T) {
		resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
			Model: "text-embedding-ada-002",
			Input: "test",
		})
		if err == nil {
			t.Error("Embedding() error = nil, want error")
		}
		if resp != nil {
			t.Error("Embedding() returned non-nil response")
		}
		if !strings.Contains(err.Error(), "not supported") {
			t.Errorf("Embedding() error = %v, want error containing 'not supported'", err)
		}
	})

	t.Run("ImageGeneration", func(t *testing.T) {
		resp, err := provider.ImageGeneration(ctx, &warp.ImageGenerationRequest{
			Prompt: "test",
		})
		if err == nil {
			t.Error("ImageGeneration() error = nil, want error")
		}
		if resp != nil {
			t.Error("ImageGeneration() returned non-nil response")
		}
	})

	t.Run("Rerank", func(t *testing.T) {
		resp, err := provider.Rerank(ctx, &warp.RerankRequest{
			Query:     "test",
			Documents: []string{"doc1", "doc2"},
		})
		if err == nil {
			t.Error("Rerank() error = nil, want error")
		}
		if resp != nil {
			t.Error("Rerank() returned non-nil response")
		}
	})
}

// TestGetModelInfo tests the GetModelInfo method
func TestGetModelInfo(t *testing.T) {
	provider, err := NewProvider()
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
			name:      "auto model",
			model:     "auto",
			wantNil:   false,
			wantModel: "auto",
		},
		{
			name:    "unknown model",
			model:   "unknown-model",
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := provider.GetModelInfo(tt.model)

			if tt.wantNil {
				if info != nil {
					t.Errorf("GetModelInfo(%s) = %v, want nil", tt.model, info)
				}
				return
			}

			if info == nil {
				t.Errorf("GetModelInfo(%s) = nil, want non-nil", tt.model)
				return
			}

			if info.Name != tt.wantModel {
				t.Errorf("GetModelInfo(%s).Name = %v, want %v", tt.model, info.Name, tt.wantModel)
			}

			if info.Provider != "vllm-semantic-router" {
				t.Errorf("GetModelInfo(%s).Provider = %v, want vllm-semantic-router", tt.model, info.Provider)
			}
		})
	}
}

// TestListModels tests the ListModels method
func TestListModels(t *testing.T) {
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	models := provider.ListModels()

	if len(models) == 0 {
		t.Error("ListModels() returned empty list")
	}

	// Should include "auto" model
	foundAuto := false
	for _, model := range models {
		if model.Name == "auto" {
			foundAuto = true
			if model.Provider != "vllm-semantic-router" {
				t.Errorf("auto model Provider = %v, want vllm-semantic-router", model.Provider)
			}
		}
	}

	if !foundAuto {
		t.Error("ListModels() did not include 'auto' model")
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
			name: "basic request with auto model",
			req: &warp.CompletionRequest{
				Model: "auto",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, m map[string]any) {
				if m["model"] != "auto" {
					t.Errorf("model = %v, want auto", m["model"])
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
				Model: "auto",
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

// TestConcurrentRequests tests thread safety
func TestConcurrentRequests(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			resp := `{
				"id": "chatcmpl-123",
				"object": "chat.completion",
				"created": 1677652288,
				"model": "auto",
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
				Model: "auto",
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
