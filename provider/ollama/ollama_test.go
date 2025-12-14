package ollama

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
		name    string
		opts    []Option
		wantErr bool
	}{
		{
			name:    "default options",
			opts:    []Option{},
			wantErr: false,
		},
		{
			name: "with base URL",
			opts: []Option{
				WithBaseURL("http://192.168.1.100:11434"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithBaseURL("http://localhost:11434"),
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
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "ollama" {
		t.Errorf("Name() = %v, want %v", got, "ollama")
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
		{"FunctionCalling", false, caps.FunctionCalling},
		{"Vision", false, caps.Vision},
		{"JSON", false, caps.JSON},
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
				Model: "llama3",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"model": "llama3",
				"created_at": "2024-01-01T00:00:00Z",
				"message": {
					"role": "assistant",
					"content": "Hello! How can I help you today?"
				},
				"done": true
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.Model != "llama3" {
					t.Errorf("Model = %v, want llama3", resp.Model)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				content, ok := resp.Choices[0].Message.Content.(string)
				if !ok || content != "Hello! How can I help you today?" {
					t.Errorf("Content = %v, want 'Hello! How can I help you today?'", content)
				}
			},
		},
		{
			name: "with parameters",
			req: &warp.CompletionRequest{
				Model: "mistral",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
			},
			mockResp: `{
				"model": "mistral",
				"created_at": "2024-01-01T00:00:00Z",
				"message": {
					"role": "assistant",
					"content": "Test response"
				},
				"done": true
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "llama3",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": "model not found"
			}`,
			statusCode: http.StatusNotFound,
			wantErr:    true,
		},
		{
			name: "network error",
			req: &warp.CompletionRequest{
				Model: "llama3",
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
					// Verify headers (no auth for Ollama)
					if auth := req.Header.Get("Authorization"); auth != "" {
						t.Errorf("Authorization header = %v, want empty", auth)
					}
					if ct := req.Header.Get("Content-Type"); ct != "application/json" {
						t.Errorf("Content-Type = %v, want application/json", ct)
					}

					// Verify URL
					expectedURL := "http://localhost:11434/api/chat"
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
				Model: "llama3",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{"model":"llama3","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}
{"model":"llama3","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"!"},"done":false}
{"model":"llama3","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true}
`,
			statusCode: http.StatusOK,
			wantErr:    false,
			wantChunks: 3,
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "llama3",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"error": "model not found"
			}`,
			statusCode: http.StatusNotFound,
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
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "llama3",
		Input: "test",
	})

	if err == nil {
		t.Error("Embedding() error = nil, want error")
	}
	if resp != nil {
		t.Error("Embedding() returned non-nil response, want nil")
	}
	if !strings.Contains(err.Error(), "not") {
		t.Errorf("Embedding() error = %v, want error containing 'not'", err)
	}
}

// TestTransformToOllamaRequest tests request transformation
func TestTransformToOllamaRequest(t *testing.T) {
	tests := []struct {
		name     string
		req      *warp.CompletionRequest
		validate func(*testing.T, *ollamaRequest)
	}{
		{
			name: "basic request",
			req: &warp.CompletionRequest{
				Model: "llama3",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, req *ollamaRequest) {
				if req.Model != "llama3" {
					t.Errorf("model = %v, want llama3", req.Model)
				}
				if len(req.Messages) != 1 {
					t.Errorf("len(messages) = %v, want 1", len(req.Messages))
				}
				if req.Messages[0].Content != "Hello" {
					t.Errorf("message content = %v, want Hello", req.Messages[0].Content)
				}
			},
		},
		{
			name: "with options",
			req: &warp.CompletionRequest{
				Model: "mistral",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
				TopP:        floatPtr(0.9),
			},
			validate: func(t *testing.T, req *ollamaRequest) {
				if req.Options == nil {
					t.Fatal("options is nil")
				}
				if req.Options.Temperature == nil || *req.Options.Temperature != 0.7 {
					t.Errorf("temperature = %v, want 0.7", req.Options.Temperature)
				}
				if req.Options.NumPredict == nil || *req.Options.NumPredict != 100 {
					t.Errorf("num_predict = %v, want 100", req.Options.NumPredict)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := transformToOllamaRequest(tt.req, false) // Non-streaming for test

			if result == nil {
				t.Fatal("transformToOllamaRequest returned nil")
			}

			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

// TestExtractTextContent tests text content extraction
func TestExtractTextContent(t *testing.T) {
	tests := []struct {
		name    string
		content any
		want    string
	}{
		{
			name:    "string content",
			content: "Hello, world!",
			want:    "Hello, world!",
		},
		{
			name: "multimodal content with text",
			content: []warp.ContentPart{
				{Type: "text", Text: "Hello"},
				{Type: "text", Text: " world"},
			},
			want: "Hello world",
		},
		{
			name:    "nil content",
			content: nil,
			want:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTextContent(tt.content)
			if got != tt.want {
				t.Errorf("extractTextContent() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestConcurrentRequests tests thread safety
func TestConcurrentRequests(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			resp := `{
				"model": "llama3",
				"created_at": "2024-01-01T00:00:00Z",
				"message": {
					"role": "assistant",
					"content": "Hello"
				},
				"done": true
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
				Model: "llama3",
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
