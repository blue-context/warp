package vllm

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

// Helper functions for creating pointers
func floatPtr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

func boolPtr(b bool) *bool {
	return &b
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
				WithBaseURL("http://192.168.1.100:8000"),
			},
			wantErr: false,
		},
		{
			name: "with API key",
			opts: []Option{
				WithAPIKey("secret-token-abc123"),
			},
			wantErr: false,
		},
		{
			name: "with all options",
			opts: []Option{
				WithBaseURL("http://localhost:8000"),
				WithAPIKey("test-key"),
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

	if got := provider.Name(); got != "vllm" {
		t.Errorf("Name() = %v, want %v", got, "vllm")
	}
}

// TestProviderSupports tests the Supports method
func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	caps := provider.Supports().(prov.Capabilities)

	// Test expected capabilities
	if !caps.Completion {
		t.Error("Supports().Completion = false, want true")
	}
	if !caps.Streaming {
		t.Error("Supports().Streaming = false, want true")
	}
	if !caps.Embedding {
		t.Error("Supports().Embedding = false, want true")
	}
	if !caps.JSON {
		t.Error("Supports().JSON = false, want true")
	}
	if !caps.Rerank {
		t.Error("Supports().Rerank = false, want true")
	}

	// Test unsupported capabilities
	if caps.ImageGeneration {
		t.Error("Supports().ImageGeneration = true, want false")
	}
	if caps.Transcription {
		t.Error("Supports().Transcription = true, want false")
	}
	if caps.Speech {
		t.Error("Supports().Speech = true, want false")
	}
	if caps.Moderation {
		t.Error("Supports().Moderation = true, want false")
	}
}

// TestCompletion tests the Completion method
func TestCompletion(t *testing.T) {
	tests := []struct {
		name       string
		request    *warp.CompletionRequest
		mockResp   string
		mockStatus int
		wantErr    bool
		checkResp  func(*testing.T, *warp.CompletionResponse)
	}{
		{
			name: "successful completion",
			request: &warp.CompletionRequest{
				Model: "meta-llama/Llama-2-7b-hf",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello!"},
				},
			},
			mockResp: `{
				"id": "cmpl-test123",
				"object": "text_completion",
				"created": 1234567890,
				"model": "meta-llama/Llama-2-7b-hf",
				"choices": [{
					"index": 0,
					"text": " Hello! How can I help you today?",
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 10,
					"completion_tokens": 20,
					"total_tokens": 30
				}
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
			checkResp: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.ID != "cmpl-test123" {
					t.Errorf("ID = %v, want cmpl-test123", resp.ID)
				}
				if len(resp.Choices) != 1 {
					t.Fatalf("len(Choices) = %v, want 1", len(resp.Choices))
				}
				if resp.Choices[0].Message.Content != " Hello! How can I help you today?" {
					t.Errorf("Content = %v, want ' Hello! How can I help you today?'", resp.Choices[0].Message.Content)
				}
				if resp.Usage == nil {
					t.Fatal("Usage = nil, want non-nil")
				}
				if resp.Usage.PromptTokens != 10 {
					t.Errorf("PromptTokens = %v, want 10", resp.Usage.PromptTokens)
				}
			},
		},
		{
			name: "with temperature and max_tokens",
			request: &warp.CompletionRequest{
				Model: "meta-llama/Llama-2-7b-hf",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello!"},
				},
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(100),
			},
			mockResp: `{
				"id": "cmpl-test456",
				"object": "text_completion",
				"created": 1234567890,
				"model": "meta-llama/Llama-2-7b-hf",
				"choices": [{
					"index": 0,
					"text": " Hi there!",
					"finish_reason": "stop"
				}],
				"usage": {
					"prompt_tokens": 5,
					"completion_tokens": 3,
					"total_tokens": 8
				}
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
			checkResp: func(t *testing.T, resp *warp.CompletionResponse) {
				if resp.Choices[0].Message.Content != " Hi there!" {
					t.Errorf("Content = %v, want ' Hi there!'", resp.Choices[0].Message.Content)
				}
			},
		},
		{
			name: "server error",
			request: &warp.CompletionRequest{
				Model: "meta-llama/Llama-2-7b-hf",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello!"},
				},
			},
			mockResp:   `{"error": "Internal server error"}`,
			mockStatus: http.StatusInternalServerError,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					// Verify request URL
					if !strings.HasSuffix(req.URL.Path, "/inference/v1/generate") {
						t.Errorf("Request URL = %v, want path ending with /inference/v1/generate", req.URL.Path)
					}

					// Verify content type
					if req.Header.Get("Content-Type") != "application/json" {
						t.Errorf("Content-Type = %v, want application/json", req.Header.Get("Content-Type"))
					}

					return &http.Response{
						StatusCode: tt.mockStatus,
						Body:       io.NopCloser(strings.NewReader(tt.mockResp)),
						Header:     make(http.Header),
					}, nil
				},
			}

			provider, err := NewProvider(WithHTTPClient(mockClient))
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			resp, err := provider.Completion(context.Background(), tt.request)

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

			if tt.checkResp != nil {
				tt.checkResp(t, resp)
			}
		})
	}
}

// TestCompletionStream tests the CompletionStream method
func TestCompletionStream(t *testing.T) {
	mockStreamData := `data: {"id":"cmpl-1","object":"text_completion.chunk","created":1234567890,"model":"meta-llama/Llama-2-7b-hf","choices":[{"index":0,"text":" Hello","finish_reason":null}]}

data: {"id":"cmpl-1","object":"text_completion.chunk","created":1234567890,"model":"meta-llama/Llama-2-7b-hf","choices":[{"index":0,"text":" there","finish_reason":null}]}

data: {"id":"cmpl-1","object":"text_completion.chunk","created":1234567890,"model":"meta-llama/Llama-2-7b-hf","choices":[{"index":0,"text":"!","finish_reason":"stop"}]}

data: [DONE]
`

	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(mockStreamData)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(WithHTTPClient(mockClient))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "meta-llama/Llama-2-7b-hf",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello!"},
		},
	})
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
			t.Fatalf("Recv() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}

	// Verify we got 3 chunks
	if len(chunks) != 3 {
		t.Fatalf("len(chunks) = %v, want 3", len(chunks))
	}

	// Verify first chunk
	if chunks[0].Choices[0].Delta.Content != " Hello" {
		t.Errorf("chunk[0].Content = %v, want ' Hello'", chunks[0].Choices[0].Delta.Content)
	}

	// Verify last chunk has finish reason
	if chunks[2].Choices[0].FinishReason == nil {
		t.Error("chunk[2].FinishReason = nil, want non-nil")
	} else if *chunks[2].Choices[0].FinishReason != "stop" {
		t.Errorf("chunk[2].FinishReason = %v, want 'stop'", *chunks[2].Choices[0].FinishReason)
	}
}

// TestEmbedding tests the Embedding method
func TestEmbedding(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			// Verify endpoint
			if !strings.HasSuffix(req.URL.Path, "/pooling") {
				t.Errorf("Request URL = %v, want path ending with /pooling", req.URL.Path)
			}

			mockResp := `{
				"data": [
					{"data": [0.1, 0.2, 0.3]},
					{"data": [0.4, 0.5, 0.6]}
				]
			}`

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(mockResp)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(WithHTTPClient(mockClient))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "intfloat/e5-small",
		Input: []string{"Hello", "World"},
	})
	if err != nil {
		t.Fatalf("Embedding() error = %v", err)
	}

	if len(resp.Data) != 2 {
		t.Fatalf("len(Data) = %v, want 2", len(resp.Data))
	}

	if len(resp.Data[0].Embedding) != 3 {
		t.Errorf("len(Embedding[0]) = %v, want 3", len(resp.Data[0].Embedding))
	}

	if resp.Data[0].Embedding[0] != 0.1 {
		t.Errorf("Embedding[0][0] = %v, want 0.1", resp.Data[0].Embedding[0])
	}
}

// TestRerank tests the Rerank method
func TestRerank(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			// Verify endpoint
			if !strings.HasSuffix(req.URL.Path, "/rerank") {
				t.Errorf("Request URL = %v, want path ending with /rerank", req.URL.Path)
			}

			mockResp := `{
				"results": [
					{"index": 0, "relevance_score": 0.95, "document": "Paris is the capital of France"},
					{"index": 1, "relevance_score": 0.15, "document": "London is the capital of England"}
				]
			}`

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(strings.NewReader(mockResp)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider, err := NewProvider(WithHTTPClient(mockClient))
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	resp, err := provider.Rerank(context.Background(), &warp.RerankRequest{
		Model: "BAAI/bge-reranker-large",
		Query: "What is the capital of France?",
		Documents: []string{
			"Paris is the capital of France",
			"London is the capital of England",
		},
		TopN:            intPtr(2),
		ReturnDocuments: boolPtr(true),
	})
	if err != nil {
		t.Fatalf("Rerank() error = %v", err)
	}

	if len(resp.Results) != 2 {
		t.Fatalf("len(Results) = %v, want 2", len(resp.Results))
	}

	if resp.Results[0].RelevanceScore != 0.95 {
		t.Errorf("Results[0].RelevanceScore = %v, want 0.95", resp.Results[0].RelevanceScore)
	}

	if resp.Results[0].Document != "Paris is the capital of France" {
		t.Errorf("Results[0].Document = %v, want 'Paris is the capital of France'", resp.Results[0].Document)
	}
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
		wantName  string
		wantFound bool
	}{
		{
			name:      "known model",
			model:     "meta-llama/Llama-2-7b-hf",
			wantName:  "meta-llama/Llama-2-7b-hf",
			wantFound: true,
		},
		{
			name:      "unknown model returns default",
			model:     "unknown/model",
			wantName:  "unknown/model",
			wantFound: true, // Returns default info
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := provider.GetModelInfo(tt.model)

			if tt.wantFound {
				if info == nil {
					t.Fatal("GetModelInfo() = nil, want non-nil")
				}
				if info.Name != tt.wantName {
					t.Errorf("Name = %v, want %v", info.Name, tt.wantName)
				}
				// All vLLM models should have $0 cost
				if info.InputCostPer1M != 0.00 {
					t.Errorf("InputCostPer1M = %v, want 0.00", info.InputCostPer1M)
				}
			} else {
				if info != nil {
					t.Errorf("GetModelInfo() = %v, want nil", info)
				}
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

	// Verify models are sorted
	for i := 1; i < len(models); i++ {
		if models[i-1].Name > models[i].Name {
			t.Errorf("Models not sorted: %v > %v", models[i-1].Name, models[i].Name)
		}
	}

	// Verify all models have vllm provider
	for _, model := range models {
		if model.Provider != "vllm" {
			t.Errorf("Model %v has provider %v, want vllm", model.Name, model.Provider)
		}
	}
}

// TestUnsupportedMethods tests that unsupported methods return errors
func TestUnsupportedMethods(t *testing.T) {
	provider, err := NewProvider()
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()

	// Test ImageGeneration
	t.Run("ImageGeneration", func(t *testing.T) {
		_, err := provider.ImageGeneration(ctx, &warp.ImageGenerationRequest{})
		if err == nil {
			t.Error("ImageGeneration() error = nil, want error")
		}
	})

	// Test ImageEdit
	t.Run("ImageEdit", func(t *testing.T) {
		_, err := provider.ImageEdit(ctx, &warp.ImageEditRequest{})
		if err == nil {
			t.Error("ImageEdit() error = nil, want error")
		}
	})

	// Test ImageVariation
	t.Run("ImageVariation", func(t *testing.T) {
		_, err := provider.ImageVariation(ctx, &warp.ImageVariationRequest{})
		if err == nil {
			t.Error("ImageVariation() error = nil, want error")
		}
	})

	// Test Transcription
	t.Run("Transcription", func(t *testing.T) {
		_, err := provider.Transcription(ctx, &warp.TranscriptionRequest{})
		if err == nil {
			t.Error("Transcription() error = nil, want error")
		}
	})

	// Test Speech
	t.Run("Speech", func(t *testing.T) {
		_, err := provider.Speech(ctx, &warp.SpeechRequest{})
		if err == nil {
			t.Error("Speech() error = nil, want error")
		}
	})

	// Test Moderation
	t.Run("Moderation", func(t *testing.T) {
		_, err := provider.Moderation(ctx, &warp.ModerationRequest{})
		if err == nil {
			t.Error("Moderation() error = nil, want error")
		}
	})
}

// TestMessagesToPrompt tests the message to prompt conversion
func TestMessagesToPrompt(t *testing.T) {
	tests := []struct {
		name     string
		messages []warp.Message
		want     string
	}{
		{
			name: "single user message",
			messages: []warp.Message{
				{Role: "user", Content: "Hello!"},
			},
			want: "User: Hello!\n\nAssistant:",
		},
		{
			name: "system and user messages",
			messages: []warp.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello!"},
			},
			want: "System: You are a helpful assistant.\n\nUser: Hello!\n\nAssistant:",
		},
		{
			name: "conversation with assistant response",
			messages: []warp.Message{
				{Role: "user", Content: "Hello!"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "How are you?"},
			},
			want: "User: Hello!\n\nAssistant: Hi there!\n\nUser: How are you?\n\nAssistant:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := messagesToPrompt(tt.messages)
			if got != tt.want {
				t.Errorf("messagesToPrompt() = %q, want %q", got, tt.want)
			}
		})
	}
}
