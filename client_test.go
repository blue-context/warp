package warp

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/blue-context/warp/cache"
)

// Capabilities defines what operations a provider supports (for testing).
type Capabilities struct {
	Completion      bool
	Streaming       bool
	Embedding       bool
	ImageGeneration bool
	ImageEdit       bool
	ImageVariation  bool
	Transcription   bool
	Speech          bool
	Moderation      bool
	FunctionCalling bool
	Vision          bool
	JSON            bool
	Rerank          bool
}

// mockProvider is a mock provider for testing
type mockProvider struct {
	name                 string
	completionFunc       func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
	completionStreamFunc func(ctx context.Context, req *CompletionRequest) (Stream, error)
	embeddingFunc        func(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
	transcriptionFunc    func(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)
	supports             Capabilities
	rerankResp           *RerankResponse
	rerankErr            error
	rerankReq            *RerankRequest
	rerankCtx            context.Context
	capabilities         Capabilities
}

func (m *mockProvider) Name() string {
	return m.name
}

func (m *mockProvider) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	if m.completionFunc != nil {
		return m.completionFunc(ctx, req)
	}
	return &CompletionResponse{
		ID:      "test-123",
		Model:   req.Model,
		Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test response"}, FinishReason: "stop"}},
		Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}, nil
}

func (m *mockProvider) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	if m.completionStreamFunc != nil {
		return m.completionStreamFunc(ctx, req)
	}
	return &mockStream{}, nil
}

func (m *mockProvider) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	if m.embeddingFunc != nil {
		return m.embeddingFunc(ctx, req)
	}
	return &EmbeddingResponse{
		Object: "list",
		Data:   []Embedding{{Object: "embedding", Embedding: []float64{0.1, 0.2, 0.3}, Index: 0}},
		Model:  req.Model,
	}, nil
}

func (m *mockProvider) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	if m.transcriptionFunc != nil {
		return m.transcriptionFunc(ctx, req)
	}
	return &TranscriptionResponse{
		Text:     "Mock transcription",
		Language: "en",
	}, nil
}

func (m *mockProvider) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	return nil, errors.New("not implemented")
}

func (m *mockProvider) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockProvider) Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error) {
	m.rerankCtx = ctx
	m.rerankReq = req
	if m.rerankErr != nil {
		return nil, m.rerankErr
	}
	return m.rerankResp, nil
}

func (m *mockProvider) Supports() interface{} {
	// Return capabilities if set
	if m.capabilities.Completion || m.capabilities.Streaming || m.capabilities.Embedding || m.capabilities.Rerank {
		return m.capabilities
	}
	// Return supports if set
	if m.supports.Completion || m.supports.Streaming || m.supports.Embedding {
		return m.supports
	}
	// Default capabilities
	return struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		ImageEdit       bool
		ImageVariation  bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
		Rerank          bool
	}{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: false,
		ImageEdit:       false,
		ImageVariation:  false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true,
		Vision:          false,
		JSON:            true,
		Rerank:          false,
	}
}

// mockStream is a mock stream for testing
type mockStream struct {
	chunks   []*CompletionChunk
	index    int
	closeErr error
}

func (m *mockStream) Recv() (*CompletionChunk, error) {
	if m.index >= len(m.chunks) {
		return nil, io.EOF
	}
	chunk := m.chunks[m.index]
	m.index++
	return chunk, nil
}

func (m *mockStream) Close() error {
	return m.closeErr
}

// TestNewClient tests the NewClient constructor
func TestNewClient(t *testing.T) {
	tests := []struct {
		name    string
		opts    []ClientOption
		wantErr bool
		errMsg  string
	}{
		{
			name:    "empty client",
			opts:    []ClientOption{},
			wantErr: false,
		},
		{
			name: "with OpenAI API key",
			opts: []ClientOption{
				WithAPIKey("openai", "sk-test"),
			},
			wantErr: false,
		},
		{
			name: "with multiple providers",
			opts: []ClientOption{
				WithAPIKey("openai", "sk-test"),
				WithAPIKey("anthropic", "sk-ant-test"),
			},
			wantErr: false,
		},
		{
			name: "with timeout",
			opts: []ClientOption{
				WithTimeout(30 * time.Second),
			},
			wantErr: false,
		},
		{
			name: "with retries",
			opts: []ClientOption{
				WithRetries(5, 2*time.Second, 2.5),
			},
			wantErr: false,
		},
		{
			name: "with invalid timeout",
			opts: []ClientOption{
				WithTimeout(0),
			},
			wantErr: true,
			errMsg:  "timeout must be positive",
		},
		{
			name: "with invalid retries",
			opts: []ClientOption{
				WithRetries(-1, time.Second, 2.0),
			},
			wantErr: true,
			errMsg:  "maxRetries must be non-negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.opts...)

			if tt.wantErr {
				if err == nil {
					t.Error("NewClient() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("NewClient() error = %v, want error containing %q", err, tt.errMsg)
				}
				return
			}

			if err != nil {
				t.Errorf("NewClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if client == nil {
				t.Error("NewClient() returned nil client")
			}

			// Clean up
			if client != nil {
				client.Close()
			}
		})
	}
}

// TestParseModel tests the parseModel function
func TestParseModel(t *testing.T) {
	tests := []struct {
		name         string
		model        string
		wantProvider string
		wantModel    string
		wantErr      bool
		errMsg       string
	}{
		{
			name:         "valid model",
			model:        "openai/gpt-4",
			wantProvider: "openai",
			wantModel:    "gpt-4",
			wantErr:      false,
		},
		{
			name:         "valid model with hyphen",
			model:        "anthropic/claude-3-opus-20240229",
			wantProvider: "anthropic",
			wantModel:    "claude-3-opus-20240229",
			wantErr:      false,
		},
		{
			name:         "valid model with slash in name",
			model:        "azure/gpt-4/deployment",
			wantProvider: "azure",
			wantModel:    "gpt-4/deployment",
			wantErr:      false,
		},
		{
			name:    "missing provider",
			model:   "gpt-4",
			wantErr: true,
			errMsg:  "invalid model format",
		},
		{
			name:    "empty provider",
			model:   "/gpt-4",
			wantErr: true,
			errMsg:  "provider name is empty",
		},
		{
			name:    "empty model",
			model:   "openai/",
			wantErr: true,
			errMsg:  "model name is empty",
		},
		{
			name:    "empty string",
			model:   "",
			wantErr: true,
			errMsg:  "invalid model format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, model, err := parseModel(tt.model)

			if tt.wantErr {
				if err == nil {
					t.Error("parseModel() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("parseModel() error = %v, want error containing %q", err, tt.errMsg)
				}
				return
			}

			if err != nil {
				t.Errorf("parseModel() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if provider != tt.wantProvider {
				t.Errorf("parseModel() provider = %v, want %v", provider, tt.wantProvider)
			}

			if model != tt.wantModel {
				t.Errorf("parseModel() model = %v, want %v", model, tt.wantModel)
			}
		})
	}
}

// TestCompletion tests the Completion method
func TestCompletion(t *testing.T) {
	tests := []struct {
		name       string
		req        *CompletionRequest
		setupMock  func(*mockProvider)
		wantErr    bool
		errMsg     string
		validateFn func(*testing.T, *CompletionResponse)
	}{
		{
			name: "successful completion",
			req: &CompletionRequest{
				Model:    "test/gpt-4",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			},
			setupMock: func(m *mockProvider) {
				m.completionFunc = func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
					return &CompletionResponse{
						ID:      "test-123",
						Model:   req.Model,
						Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Hello!"}, FinishReason: "stop"}},
						Usage:   &Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
					}, nil
				}
			},
			wantErr: false,
			validateFn: func(t *testing.T, resp *CompletionResponse) {
				if resp.ID != "test-123" {
					t.Errorf("ID = %v, want test-123", resp.ID)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("len(Choices) = %v, want 1", len(resp.Choices))
				}
			},
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
			errMsg:  "request cannot be nil",
		},
		{
			name: "invalid model format",
			req: &CompletionRequest{
				Model:    "gpt-4",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			},
			wantErr: true,
			errMsg:  "invalid model format",
		},
		{
			name: "provider not found",
			req: &CompletionRequest{
				Model:    "nonexistent/model",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			},
			wantErr: true,
			errMsg:  "provider \"nonexistent\" not found",
		},
		{
			name: "with timeout",
			req: &CompletionRequest{
				Model:    "test/gpt-4",
				Messages: []Message{{Role: "user", Content: "Hello"}},
				Timeout:  5 * time.Second,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client with mock provider
			client, err := NewClient()
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			defer client.Close()

			// Register mock provider if needed
			if !tt.wantErr || strings.Contains(tt.errMsg, "provider") {
				mock := &mockProvider{
					name: "test",
				}
				if tt.setupMock != nil {
					tt.setupMock(mock)
				}

				if err := client.RegisterProvider(mock); err != nil {
					t.Fatalf("failed to register mock provider: %v", err)
				}
			}

			// Call Completion
			resp, err := client.Completion(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("Completion() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("Completion() error = %v, want error containing %q", err, tt.errMsg)
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

			if tt.validateFn != nil {
				tt.validateFn(t, resp)
			}
		})
	}
}

// TestCompletionStream tests the CompletionStream method
func TestCompletionStream(t *testing.T) {
	tests := []struct {
		name      string
		req       *CompletionRequest
		setupMock func(*mockProvider)
		wantErr   bool
		errMsg    string
	}{
		{
			name: "successful stream",
			req: &CompletionRequest{
				Model:    "test/gpt-4",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			},
			setupMock: func(m *mockProvider) {
				m.completionStreamFunc = func(ctx context.Context, req *CompletionRequest) (Stream, error) {
					return &mockStream{
						chunks: []*CompletionChunk{
							{ID: "1", Choices: []ChunkChoice{{Index: 0, Delta: MessageDelta{Content: "Hello"}}}},
							{ID: "1", Choices: []ChunkChoice{{Index: 0, Delta: MessageDelta{Content: " world"}}}},
						},
					}, nil
				}
			},
			wantErr: false,
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
			errMsg:  "request cannot be nil",
		},
		{
			name: "invalid model format",
			req: &CompletionRequest{
				Model:    "gpt-4",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			},
			wantErr: true,
			errMsg:  "invalid model format",
		},
		{
			name: "provider not found",
			req: &CompletionRequest{
				Model:    "nonexistent/model",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			},
			wantErr: true,
			errMsg:  "provider \"nonexistent\" not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client with mock provider
			client, err := NewClient()
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			defer client.Close()

			// Register mock provider if needed
			if !tt.wantErr || strings.Contains(tt.errMsg, "provider") {
				mock := &mockProvider{
					name: "test",
				}
				if tt.setupMock != nil {
					tt.setupMock(mock)
				}

				if err := client.RegisterProvider(mock); err != nil {
					t.Fatalf("failed to register mock provider: %v", err)
				}
			}

			// Call CompletionStream
			stream, err := client.CompletionStream(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("CompletionStream() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("CompletionStream() error = %v, want error containing %q", err, tt.errMsg)
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

			// Try to read from stream
			chunk, err := stream.Recv()
			if err != nil && err != io.EOF {
				t.Errorf("stream.Recv() error = %v", err)
			}

			if err == nil && chunk == nil {
				t.Error("stream.Recv() returned nil chunk")
			}

			stream.Close()
		})
	}
}

// TestEmbedding tests the Embedding method
func TestEmbedding(t *testing.T) {
	tests := []struct {
		name       string
		req        *EmbeddingRequest
		setupMock  func(*mockProvider)
		wantErr    bool
		errMsg     string
		validateFn func(*testing.T, *EmbeddingResponse)
	}{
		{
			name: "successful embedding",
			req: &EmbeddingRequest{
				Model: "test/text-embedding-ada-002",
				Input: "Hello, world!",
			},
			setupMock: func(m *mockProvider) {
				m.embeddingFunc = func(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
					return &EmbeddingResponse{
						Object: "list",
						Data:   []Embedding{{Object: "embedding", Embedding: []float64{0.1, 0.2, 0.3}, Index: 0}},
						Model:  req.Model,
					}, nil
				}
			},
			wantErr: false,
			validateFn: func(t *testing.T, resp *EmbeddingResponse) {
				if len(resp.Data) != 1 {
					t.Errorf("len(Data) = %v, want 1", len(resp.Data))
				}
				if len(resp.Data[0].Embedding) != 3 {
					t.Errorf("len(Embedding) = %v, want 3", len(resp.Data[0].Embedding))
				}
			},
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
			errMsg:  "request cannot be nil",
		},
		{
			name: "invalid model format",
			req: &EmbeddingRequest{
				Model: "text-embedding-ada-002",
				Input: "Hello",
			},
			wantErr: true,
			errMsg:  "invalid model format",
		},
		{
			name: "provider not found",
			req: &EmbeddingRequest{
				Model: "nonexistent/model",
				Input: "Hello",
			},
			wantErr: true,
			errMsg:  "provider \"nonexistent\" not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client with mock provider
			client, err := NewClient()
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			defer client.Close()

			// Register mock provider if needed
			if !tt.wantErr || strings.Contains(tt.errMsg, "provider") {
				mock := &mockProvider{
					name: "test",
				}
				if tt.setupMock != nil {
					tt.setupMock(mock)
				}

				if err := client.RegisterProvider(mock); err != nil {
					t.Fatalf("failed to register mock provider: %v", err)
				}
			}

			// Call Embedding
			resp, err := client.Embedding(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Error("Embedding() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("Embedding() error = %v, want error containing %q", err, tt.errMsg)
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

			if tt.validateFn != nil {
				tt.validateFn(t, resp)
			}
		})
	}
}

// TestRetryLogic tests the retry logic
func TestRetryLogic(t *testing.T) {
	tests := []struct {
		name        string
		maxRetries  int
		attempts    int
		errorType   error
		wantErr     bool
		wantAttempt int
	}{
		{
			name:        "success on first attempt",
			maxRetries:  3,
			attempts:    0,
			wantErr:     false,
			wantAttempt: 1,
		},
		{
			name:        "success on second attempt",
			maxRetries:  3,
			attempts:    1,
			errorType:   NewRateLimitError("rate limit", "test", 0, nil),
			wantErr:     false,
			wantAttempt: 2,
		},
		{
			name:        "exhausted retries",
			maxRetries:  2,
			attempts:    3,
			errorType:   NewTimeoutError("timeout", "test", nil),
			wantErr:     true,
			wantAttempt: 3,
		},
		{
			name:        "non-retryable error",
			maxRetries:  3,
			attempts:    1,
			errorType:   NewAuthenticationError("auth failed", "test", nil),
			wantErr:     true,
			wantAttempt: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(WithMaxRetries(tt.maxRetries))
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			defer client.Close()

			attemptCount := 0
			mock := &mockProvider{
				name: "test",
				completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
					attemptCount++
					if attemptCount <= tt.attempts {
						return nil, tt.errorType
					}
					return &CompletionResponse{
						ID:      "test-123",
						Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Success"}, FinishReason: "stop"}},
					}, nil
				},
			}

			if err := client.RegisterProvider(mock); err != nil {
				t.Fatalf("failed to register mock provider: %v", err)
			}

			_, err = client.Completion(context.Background(), &CompletionRequest{
				Model:    "test/gpt-4",
				Messages: []Message{{Role: "user", Content: "Hello"}},
			})

			if tt.wantErr {
				if err == nil {
					t.Error("Completion() error = nil, wantErr true")
				}
			} else {
				if err != nil {
					t.Errorf("Completion() error = %v, wantErr %v", err, tt.wantErr)
				}
			}

			if attemptCount != tt.wantAttempt {
				t.Errorf("attempt count = %v, want %v", attemptCount, tt.wantAttempt)
			}
		})
	}
}

// TestContextPropagation tests context propagation
func TestContextPropagation(t *testing.T) {
	client, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			// Verify context has required values
			if RequestIDFromContext(ctx) == "" {
				t.Error("context missing request ID")
			}
			if ProviderFromContext(ctx) != "test" {
				t.Errorf("context provider = %v, want test", ProviderFromContext(ctx))
			}
			if ModelFromContext(ctx) != "gpt-4" {
				t.Errorf("context model = %v, want gpt-4", ModelFromContext(ctx))
			}
			if StartTimeFromContext(ctx).IsZero() {
				t.Error("context missing start time")
			}

			return &CompletionResponse{
				ID:      "test-123",
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
			}, nil
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})

	if err != nil {
		t.Errorf("Completion() error = %v", err)
	}
}

// TestContextCancellation tests context cancellation
func TestContextCancellation(t *testing.T) {
	client, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			// Simulate slow operation that respects context cancellation
			select {
			case <-time.After(100 * time.Millisecond):
				return &CompletionResponse{
					ID:      "test-123",
					Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				}, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err = client.Completion(ctx, &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})

	if err == nil {
		t.Error("Completion() error = nil, want context deadline exceeded")
	}

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Completion() error = %v, want context.DeadlineExceeded", err)
	}
}

// TestCompletionCost tests the CompletionCost method
func TestCompletionCost(t *testing.T) {
	client, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	tests := []struct {
		name    string
		resp    *CompletionResponse
		wantErr bool
		errMsg  string
	}{
		{
			name: "with usage",
			resp: &CompletionResponse{
				ID:      "test-123",
				Model:   "openai/gpt-4",
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			},
			wantErr: false,
		},
		{
			name:    "nil response",
			resp:    nil,
			wantErr: true,
			errMsg:  "response cannot be nil",
		},
		{
			name: "no usage",
			resp: &CompletionResponse{
				ID:      "test-123",
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test"}, FinishReason: "stop"}},
			},
			wantErr: true,
			errMsg:  "usage information not available",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cost, err := client.CompletionCost(tt.resp)

			if tt.wantErr {
				if err == nil {
					t.Error("CompletionCost() error = nil, wantErr true")
					return
				}
				if tt.errMsg != "" && !strings.Contains(err.Error(), tt.errMsg) {
					t.Errorf("CompletionCost() error = %v, want error containing %q", err, tt.errMsg)
				}
				return
			}

			if err != nil {
				t.Errorf("CompletionCost() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Cost should now be calculated based on actual pricing
			if cost <= 0 {
				t.Errorf("CompletionCost() = %v, want > 0", cost)
			}
		})
	}
}

// TestClose tests the Close method
func TestClose(t *testing.T) {
	client, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	err = client.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// Calling Close multiple times should be safe
	err = client.Close()
	if err != nil {
		t.Errorf("Close() error on second call = %v", err)
	}
}

// TestCostTracking tests cost tracking integration
func TestCostTracking(t *testing.T) {
	tests := []struct {
		name      string
		trackCost bool
		model     string
		usage     *Usage
		wantCost  bool
	}{
		{
			name:      "cost tracking enabled",
			trackCost: true,
			model:     "openai/gpt-4",
			usage:     &Usage{PromptTokens: 1000, CompletionTokens: 500},
			wantCost:  true,
		},
		{
			name:      "cost tracking disabled",
			trackCost: false,
			model:     "openai/gpt-4",
			usage:     &Usage{PromptTokens: 1000, CompletionTokens: 500},
			wantCost:  true, // Cost calculator is lazily initialized
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client with cost tracking setting
			opts := []ClientOption{}
			if tt.trackCost {
				opts = append(opts, WithCostTracking(true))
			}

			client, err := NewClient(opts...)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			defer client.Close()

			// Create response
			resp := &CompletionResponse{
				Model: tt.model,
				Usage: tt.usage,
			}

			// Calculate cost
			cost, err := client.CompletionCost(resp)
			if err != nil {
				t.Errorf("CompletionCost() error = %v", err)
				return
			}

			if tt.wantCost && cost <= 0 {
				t.Errorf("CompletionCost() = %v, want > 0", cost)
			}
		})
	}
}

// TestCostTrackingWithRealPricing tests cost calculation with actual pricing
func TestCostTrackingWithRealPricing(t *testing.T) {
	client, err := NewClient(WithCostTracking(true))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	tests := []struct {
		name         string
		model        string
		promptTokens int
		compTokens   int
		minCost      float64
		maxCost      float64
	}{
		{
			name:         "gpt-4 small request",
			model:        "openai/gpt-4",
			promptTokens: 1000,
			compTokens:   500,
			minCost:      0.05, // Reasonable lower bound
			maxCost:      0.07, // Reasonable upper bound
		},
		{
			name:         "gpt-3.5-turbo request",
			model:        "openai/gpt-3.5-turbo",
			promptTokens: 2000,
			compTokens:   1000,
			minCost:      0.002,
			maxCost:      0.003,
		},
		{
			name:         "claude-3-opus request",
			model:        "anthropic/claude-3-opus-20240229",
			promptTokens: 1000,
			compTokens:   500,
			minCost:      0.05,
			maxCost:      0.06,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &CompletionResponse{
				Model: tt.model,
				Usage: &Usage{
					PromptTokens:     tt.promptTokens,
					CompletionTokens: tt.compTokens,
				},
			}

			cost, err := client.CompletionCost(resp)
			if err != nil {
				t.Errorf("CompletionCost() error = %v", err)
				return
			}

			if cost < tt.minCost || cost > tt.maxCost {
				t.Errorf("CompletionCost() = %v, want between %v and %v", cost, tt.minCost, tt.maxCost)
			}
		})
	}
}

// TestCostTrackingErrors tests error cases in cost tracking
func TestCostTrackingErrors(t *testing.T) {
	client, err := NewClient(WithCostTracking(true))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	tests := []struct {
		name    string
		resp    *CompletionResponse
		wantErr bool
	}{
		{
			name:    "nil response",
			resp:    nil,
			wantErr: true,
		},
		{
			name: "nil usage",
			resp: &CompletionResponse{
				Model: "openai/gpt-4",
				Usage: nil,
			},
			wantErr: true,
		},
		{
			name: "unknown model",
			resp: &CompletionResponse{
				Model: "unknown/unknown-model",
				Usage: &Usage{PromptTokens: 100, CompletionTokens: 50},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := client.CompletionCost(tt.resp)
			if (err != nil) != tt.wantErr {
				t.Errorf("CompletionCost() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestClientWithCostTrackingInitialization tests that cost calculator is initialized correctly by testing its behavior
func TestClientWithCostTrackingInitialization(t *testing.T) {
	tests := []struct {
		name    string
		opts    []ClientOption
		wantErr bool // Whether cost calculation should work
	}{
		{
			name:    "cost tracking enabled",
			opts:    []ClientOption{WithCostTracking(true)},
			wantErr: false,
		},
		{
			name:    "cost tracking with budget",
			opts:    []ClientOption{WithCostTracking(true), WithMaxBudget(10.0)},
			wantErr: false,
		},
		{
			name:    "no cost tracking still allows cost calculation (lazy init)",
			opts:    []ClientOption{},
			wantErr: false, // Cost calculator is lazily initialized
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.opts...)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			defer client.Close()

			// Test that cost calculation works (or doesn't)
			resp := &CompletionResponse{
				Model: "openai/gpt-4",
				Usage: &Usage{
					PromptTokens:     1000,
					CompletionTokens: 500,
				},
			}

			_, err = client.CompletionCost(resp)
			if (err != nil) != tt.wantErr {
				t.Errorf("CompletionCost() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestClientWithCache tests cache integration
func TestClientWithCache(t *testing.T) {
	memCache := cache.NewMemoryCache(1024 * 1024) // 1MB
	defer memCache.Close()

	client, err := NewClient(WithCache(memCache))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	// Mock provider that tracks call count
	callCount := 0
	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			callCount++
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test response"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	req := &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	// First call - should hit provider
	resp1, err := client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1", callCount)
	}

	// Second call - should hit cache
	resp2, err := client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}
	if callCount != 1 {
		t.Errorf("callCount = %d, want 1 (cached)", callCount)
	}

	// Responses should be identical
	if resp1.ID != resp2.ID {
		t.Errorf("cached response ID = %s, want %s", resp2.ID, resp1.ID)
	}
}

// TestClientCacheHitMiss tests cache hit and miss behavior
func TestClientCacheHitMiss(t *testing.T) {
	memCache := cache.NewMemoryCache(0)
	defer memCache.Close()

	client, err := NewClient(WithCache(memCache))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	callCount := 0
	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			callCount++
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test response"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	// First request
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Different request - should miss cache
	_, err = client.Completion(context.Background(), &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Goodbye"}},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (different messages)", callCount)
	}
}

// TestClientCacheDifferentParameters tests that different parameters generate different cache keys
func TestClientCacheDifferentParameters(t *testing.T) {
	memCache := cache.NewMemoryCache(0)
	defer memCache.Close()

	client, err := NewClient(WithCache(memCache))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	callCount := 0
	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			callCount++
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test response"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	baseReq := &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	// First call
	_, err = client.Completion(context.Background(), baseReq)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Same messages, different temperature
	temp := float64(0.7)
	tempReq := &CompletionRequest{
		Model:       "test/gpt-4",
		Messages:    []Message{{Role: "user", Content: "Hello"}},
		Temperature: &temp,
	}
	_, err = client.Completion(context.Background(), tempReq)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Same messages, different maxTokens
	maxTokens := 100
	tokensReq := &CompletionRequest{
		Model:     "test/gpt-4",
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		MaxTokens: &maxTokens,
	}
	_, err = client.Completion(context.Background(), tokensReq)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if callCount != 3 {
		t.Errorf("callCount = %d, want 3 (different parameters)", callCount)
	}
}

// TestClientNoCache tests behavior without cache
func TestClientNoCache(t *testing.T) {
	client, err := NewClient() // No cache
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	callCount := 0
	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			callCount++
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test response"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	req := &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	// First call
	_, err = client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Second call - should NOT be cached
	_, err = client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (no cache)", callCount)
	}
}

// TestClientWithNoopCache tests behavior with no-op cache
func TestClientWithNoopCache(t *testing.T) {
	noopCache := cache.NewNoopCache()

	client, err := NewClient(WithCache(noopCache))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	defer client.Close()

	callCount := 0
	mock := &mockProvider{
		name: "test",
		completionFunc: func(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
			callCount++
			return &CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: "Test response"}, FinishReason: "stop"}},
				Usage:   &Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			}, nil
		},
	}

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("failed to register mock provider: %v", err)
	}

	req := &CompletionRequest{
		Model:    "test/gpt-4",
		Messages: []Message{{Role: "user", Content: "Hello"}},
	}

	// First call
	_, err = client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	// Second call - should NOT be cached (noop cache)
	_, err = client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}

	if callCount != 2 {
		t.Errorf("callCount = %d, want 2 (noop cache)", callCount)
	}
}

// TestClientCacheClose tests that closing client closes cache
func TestClientCacheClose(t *testing.T) {
	memCache := cache.NewMemoryCache(0)

	client, err := NewClient(WithCache(memCache))
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	err = client.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}
}
