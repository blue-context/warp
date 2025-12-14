package cohere

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
		errMsg  string
	}{
		{
			name:    "missing API key",
			opts:    []Option{},
			wantErr: true,
			errMsg:  "Cohere API key is required",
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("test-key"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithAPIKey("test-key"),
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
	provider, err := NewProvider(WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	if got := provider.Name(); got != "cohere" {
		t.Errorf("Name() = %v, want %v", got, "cohere")
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("test-key"))
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
		{"Streaming", false, caps.Streaming},
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
				Model: "command-r",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"response_id": "resp-123",
				"text": "Hello! How can I help you today?",
				"generation_id": "gen-456",
				"chat_history": [],
				"finish_reason": "COMPLETE",
				"meta": {
					"api_version": {},
					"billed_units": {
						"input_tokens": 10,
						"output_tokens": 20
					},
					"tokens": {
						"input_tokens": 10,
						"output_tokens": 20
					}
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.ID != "gen-456" {
					t.Errorf("ID = %v, want gen-456", resp.ID)
				}
				if resp.Model != "command-r" {
					t.Errorf("Model = %v, want command-r", resp.Model)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				content, ok := resp.Choices[0].Message.Content.(string)
				if !ok || content != "Hello! How can I help you today?" {
					t.Errorf("Content = %v, want 'Hello! How can I help you today?'", content)
				}
				if resp.Usage == nil {
					t.Error("Usage is nil")
				} else {
					if resp.Usage.PromptTokens != 10 {
						t.Errorf("PromptTokens = %v, want 10", resp.Usage.PromptTokens)
					}
					if resp.Usage.CompletionTokens != 20 {
						t.Errorf("CompletionTokens = %v, want 20", resp.Usage.CompletionTokens)
					}
				}
			},
		},
		{
			name: "with parameters",
			req: &warp.CompletionRequest{
				Model: "command-r-plus",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
			},
			mockResp: `{
				"response_id": "resp-789",
				"text": "Test response",
				"generation_id": "gen-012",
				"chat_history": [],
				"finish_reason": "COMPLETE",
				"meta": {
					"api_version": {},
					"billed_units": {
						"input_tokens": 5,
						"output_tokens": 10
					},
					"tokens": {
						"input_tokens": 5,
						"output_tokens": 10
					}
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "with chat history",
			req: &warp.CompletionRequest{
				Model: "command-r",
				Messages: []warp.Message{
					{Role: "user", Content: "What's 2+2?"},
					{Role: "assistant", Content: "4"},
					{Role: "user", Content: "And 3+3?"},
				},
			},
			mockResp: `{
				"response_id": "resp-345",
				"text": "6",
				"generation_id": "gen-678",
				"chat_history": [],
				"finish_reason": "COMPLETE",
				"meta": {
					"api_version": {},
					"billed_units": {
						"input_tokens": 15,
						"output_tokens": 2
					},
					"tokens": {
						"input_tokens": 15,
						"output_tokens": 2
					}
				}
			}`,
			statusCode: http.StatusOK,
			wantErr:    false,
			validate: func(t *testing.T, resp *warp.CompletionResponse) {
				content, ok := resp.Choices[0].Message.Content.(string)
				if !ok || content != "6" {
					t.Errorf("Content = %v, want '6'", content)
				}
			},
		},
		{
			name: "API error",
			req: &warp.CompletionRequest{
				Model: "command-r",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: `{
				"message": "Invalid API key"
			}`,
			statusCode: http.StatusUnauthorized,
			wantErr:    true,
		},
		{
			name: "network error",
			req: &warp.CompletionRequest{
				Model: "command-r",
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
					expectedURL := "https://api.cohere.ai/v1/chat"
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
				WithAPIKey("test-key"),
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
	provider, err := NewProvider(WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "command-r",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err == nil {
		t.Error("CompletionStream() error = nil, want error")
	}
	if stream != nil {
		t.Error("CompletionStream() returned non-nil stream, want nil")
	}
	if !strings.Contains(err.Error(), "not supported") {
		t.Errorf("CompletionStream() error = %v, want error containing 'not supported'", err)
	}
}

// TestEmbedding tests the Embedding method
func TestEmbedding(t *testing.T) {
	provider, err := NewProvider(WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "embed-english-v3.0",
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

// TestTransformToCohereRequest tests request transformation
func TestTransformToCohereRequest(t *testing.T) {
	tests := []struct {
		name     string
		req      *warp.CompletionRequest
		validate func(*testing.T, *cohereRequest)
	}{
		{
			name: "single message",
			req: &warp.CompletionRequest{
				Model: "command-r",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			validate: func(t *testing.T, req *cohereRequest) {
				if req.Model != "command-r" {
					t.Errorf("model = %v, want command-r", req.Model)
				}
				if req.Message != "Hello" {
					t.Errorf("message = %v, want Hello", req.Message)
				}
				if len(req.ChatHistory) != 0 {
					t.Errorf("len(chat_history) = %v, want 0", len(req.ChatHistory))
				}
			},
		},
		{
			name: "with chat history",
			req: &warp.CompletionRequest{
				Model: "command-r",
				Messages: []warp.Message{
					{Role: "user", Content: "What's 2+2?"},
					{Role: "assistant", Content: "4"},
					{Role: "user", Content: "And 3+3?"},
				},
			},
			validate: func(t *testing.T, req *cohereRequest) {
				if req.Message != "And 3+3?" {
					t.Errorf("message = %v, want 'And 3+3?'", req.Message)
				}
				if len(req.ChatHistory) != 2 {
					t.Errorf("len(chat_history) = %v, want 2", len(req.ChatHistory))
				}
				if req.ChatHistory[0].Role != "USER" {
					t.Errorf("chat_history[0].role = %v, want USER", req.ChatHistory[0].Role)
				}
				if req.ChatHistory[1].Role != "CHATBOT" {
					t.Errorf("chat_history[1].role = %v, want CHATBOT", req.ChatHistory[1].Role)
				}
			},
		},
		{
			name: "with parameters",
			req: &warp.CompletionRequest{
				Model: "command-r-plus",
				Messages: []warp.Message{
					{Role: "user", Content: "Test"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
				TopP:        floatPtr(0.9),
			},
			validate: func(t *testing.T, req *cohereRequest) {
				if req.Temperature == nil || *req.Temperature != 0.7 {
					t.Errorf("temperature = %v, want 0.7", req.Temperature)
				}
				if req.MaxTokens == nil || *req.MaxTokens != 100 {
					t.Errorf("max_tokens = %v, want 100", req.MaxTokens)
				}
				if req.P == nil || *req.P != 0.9 {
					t.Errorf("p = %v, want 0.9", req.P)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := transformToCohereRequest(tt.req)

			if result == nil {
				t.Fatal("transformToCohereRequest returned nil")
			}

			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

// TestConvertRoleToCohere tests role conversion
func TestConvertRoleToCohere(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"user", "USER"},
		{"assistant", "CHATBOT"},
		{"system", "SYSTEM"},
		{"unknown", "USER"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := convertRoleToCohere(tt.input)
			if got != tt.want {
				t.Errorf("convertRoleToCohere(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

// TestMapCohereFinishReason tests finish reason mapping
func TestMapCohereFinishReason(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"COMPLETE", "stop"},
		{"MAX_TOKENS", "length"},
		{"ERROR", "error"},
		{"UNKNOWN", "stop"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := mapCohereFinishReason(tt.input)
			if got != tt.want {
				t.Errorf("mapCohereFinishReason(%v) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

// TestConcurrentRequests tests thread safety
func TestConcurrentRequests(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			resp := `{
				"response_id": "resp-123",
				"text": "Hello",
				"generation_id": "gen-456",
				"chat_history": [],
				"finish_reason": "COMPLETE",
				"meta": {
					"api_version": {},
					"billed_units": {
						"input_tokens": 10,
						"output_tokens": 10
					},
					"tokens": {
						"input_tokens": 10,
						"output_tokens": 10
					}
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

	// Run 10 concurrent requests
	const numRequests = 10
	errChan := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			_, err := provider.Completion(context.Background(), &warp.CompletionRequest{
				Model: "command-r",
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
