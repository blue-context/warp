package bedrock

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/blue-context/warp"
	prov "github.com/blue-context/warp/provider"
)

// mockHTTPClient is a mock HTTP client for testing.
type mockHTTPClient struct {
	doFunc func(req *http.Request) (*http.Response, error)
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return m.doFunc(req)
}

func TestNewProvider(t *testing.T) {
	tests := []struct {
		name    string
		opts    []Option
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid provider",
			opts: []Option{
				WithCredentials("test-key", "test-secret"),
				WithRegion("us-east-1"),
			},
			wantErr: false,
		},
		{
			name: "missing access key",
			opts: []Option{
				WithCredentials("", "test-secret"),
			},
			wantErr: true,
			errMsg:  "AWS access key ID is required",
		},
		{
			name: "missing secret key",
			opts: []Option{
				WithCredentials("test-key", ""),
			},
			wantErr: true,
			errMsg:  "AWS secret access key is required",
		},
		{
			name: "with session token",
			opts: []Option{
				WithCredentials("test-key", "test-secret"),
				WithSessionToken("test-token"),
				WithRegion("us-west-2"),
			},
			wantErr: false,
		},
		{
			name: "default region",
			opts: []Option{
				WithCredentials("test-key", "test-secret"),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewProvider(tt.opts...)

			if (err != nil) != tt.wantErr {
				t.Errorf("NewProvider() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				if tt.errMsg != "" {
					liteLLMErr, ok := err.(*warp.WarpError)
					if !ok {
						t.Errorf("expected WarpError, got %T", err)
						return
					}
					if liteLLMErr.Message != tt.errMsg {
						t.Errorf("error message = %q, want %q", liteLLMErr.Message, tt.errMsg)
					}
				}
				return
			}

			if provider == nil {
				t.Error("provider is nil")
			}
		})
	}
}

func TestProviderName(t *testing.T) {
	provider, err := NewProvider(
		WithCredentials("test-key", "test-secret"),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	name := provider.Name()
	if name != "bedrock" {
		t.Errorf("Name() = %q, want %q", name, "bedrock")
	}
}

func TestProviderSupports(t *testing.T) {
	provider, err := NewProvider(
		WithCredentials("test-key", "test-secret"),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	capsInterface := provider.Supports()
	caps, ok := capsInterface.(prov.Capabilities)
	if !ok {
		t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
	}

	if !caps.Completion {
		t.Error("should support completion")
	}

	if !caps.Streaming {
		t.Error("should support streaming")
	}

	if caps.Embedding {
		t.Error("should not support embedding via completion endpoint")
	}

	if !caps.FunctionCalling {
		t.Error("should support function calling")
	}

	if !caps.Vision {
		t.Error("should support vision")
	}
}

func TestProviderEmbedding(t *testing.T) {
	provider, err := NewProvider(
		WithCredentials("test-key", "test-secret"),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()
	req := &warp.EmbeddingRequest{
		Model: "titan-embed-text",
		Input: "test",
	}

	_, err = provider.Embedding(ctx, req)
	if err == nil {
		t.Error("Embedding() should return error")
	}
}

func TestBuildEndpoint(t *testing.T) {
	provider, err := NewProvider(
		WithCredentials("test-key", "test-secret"),
		WithRegion("us-west-2"),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	tests := []struct {
		name    string
		modelID string
		stream  bool
		want    string
	}{
		{
			name:    "invoke endpoint",
			modelID: "anthropic.claude-3-opus-20240229-v1:0",
			stream:  false,
			want:    "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-opus-20240229-v1:0/invoke",
		},
		{
			name:    "streaming endpoint",
			modelID: "anthropic.claude-3-opus-20240229-v1:0",
			stream:  true,
			want:    "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-opus-20240229-v1:0/invoke-with-response-stream",
		},
		{
			name:    "llama model",
			modelID: "meta.llama3-70b-instruct-v1:0",
			stream:  false,
			want:    "https://bedrock-runtime.us-west-2.amazonaws.com/model/meta.llama3-70b-instruct-v1:0/invoke",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := provider.buildEndpoint(tt.modelID, tt.stream)
			if got != tt.want {
				t.Errorf("buildEndpoint() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestGetModelID(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		want    string
		wantErr bool
	}{
		{
			name:    "claude-3-opus",
			model:   "claude-3-opus",
			want:    "anthropic.claude-3-opus-20240229-v1:0",
			wantErr: false,
		},
		{
			name:    "claude-3-sonnet",
			model:   "claude-3-sonnet",
			want:    "anthropic.claude-3-sonnet-20240229-v1:0",
			wantErr: false,
		},
		{
			name:    "llama3-70b",
			model:   "llama3-70b",
			want:    "meta.llama3-70b-instruct-v1:0",
			wantErr: false,
		},
		{
			name:    "titan-text-express",
			model:   "titan-text-express",
			want:    "amazon.titan-text-express-v1",
			wantErr: false,
		},
		{
			name:    "full bedrock ID",
			model:   "anthropic.claude-3-opus-20240229-v1:0",
			want:    "anthropic.claude-3-opus-20240229-v1:0",
			wantErr: false,
		},
		{
			name:    "unknown model without dot",
			model:   "unknown-model",
			want:    "",
			wantErr: true,
		},
		{
			name:    "empty model",
			model:   "",
			want:    "",
			wantErr: true,
		},
		{
			name:    "custom model with vendor prefix",
			model:   "custom.model-v1",
			want:    "custom.model-v1",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getModelID(tt.model)

			if (err != nil) != tt.wantErr {
				t.Errorf("getModelID() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if got != tt.want {
				t.Errorf("getModelID() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDetectModelFamily(t *testing.T) {
	tests := []struct {
		name    string
		modelID string
		want    modelFamily
	}{
		{
			name:    "claude model",
			modelID: "anthropic.claude-3-opus-20240229-v1:0",
			want:    familyClaude,
		},
		{
			name:    "llama model",
			modelID: "meta.llama3-70b-instruct-v1:0",
			want:    familyLlama,
		},
		{
			name:    "titan text model",
			modelID: "amazon.titan-text-express-v1",
			want:    familyTitan,
		},
		{
			name:    "titan embed model",
			modelID: "amazon.titan-embed-text-v1",
			want:    familyTitan,
		},
		{
			name:    "cohere model",
			modelID: "cohere.command-r-v1:0",
			want:    familyCohere,
		},
		{
			name:    "ai21 model",
			modelID: "ai21.j2-ultra-v1",
			want:    familyAI21,
		},
		{
			name:    "stability model",
			modelID: "stability.stable-diffusion-xl-v1",
			want:    familyStability,
		},
		{
			name:    "unknown model",
			modelID: "unknown.model-v1",
			want:    familyUnknown,
		},
		{
			name:    "invalid format",
			modelID: "invalid-format",
			want:    familyUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := detectModelFamily(tt.modelID)
			if got != tt.want {
				t.Errorf("detectModelFamily() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSupportsStreaming(t *testing.T) {
	tests := []struct {
		name   string
		family modelFamily
		want   bool
	}{
		{
			name:   "claude supports streaming",
			family: familyClaude,
			want:   true,
		},
		{
			name:   "llama doesn't support streaming (not yet implemented)",
			family: familyLlama,
			want:   false,
		},
		{
			name:   "titan doesn't support streaming (not yet implemented)",
			family: familyTitan,
			want:   false,
		},
		{
			name:   "cohere doesn't support streaming (not yet implemented)",
			family: familyCohere,
			want:   false,
		},
		{
			name:   "unknown doesn't support streaming",
			family: familyUnknown,
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := supportsStreaming(tt.family)
			if got != tt.want {
				t.Errorf("supportsStreaming() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSupportsTools(t *testing.T) {
	tests := []struct {
		name   string
		family modelFamily
		want   bool
	}{
		{
			name:   "claude supports tools",
			family: familyClaude,
			want:   true,
		},
		{
			name:   "llama doesn't support tools",
			family: familyLlama,
			want:   false,
		},
		{
			name:   "titan doesn't support tools",
			family: familyTitan,
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := supportsTools(tt.family)
			if got != tt.want {
				t.Errorf("supportsTools() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSupportsVision(t *testing.T) {
	tests := []struct {
		name   string
		family modelFamily
		want   bool
	}{
		{
			name:   "claude supports vision",
			family: familyClaude,
			want:   true,
		},
		{
			name:   "llama supports vision",
			family: familyLlama,
			want:   true,
		},
		{
			name:   "titan doesn't support vision",
			family: familyTitan,
			want:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := supportsVision(tt.family)
			if got != tt.want {
				t.Errorf("supportsVision() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCompletion(t *testing.T) {
	tests := []struct {
		name       string
		req        *warp.CompletionRequest
		mockResp   *http.Response
		mockErr    error
		wantErr    bool
		errMsg     string
		checkModel bool
	}{
		{
			name: "successful claude completion",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: &http.Response{
				StatusCode: 200,
				Body: io.NopCloser(bytes.NewBufferString(`{
					"id": "msg_123",
					"type": "message",
					"role": "assistant",
					"content": [{"type": "text", "text": "Hello!"}],
					"stop_reason": "end_turn",
					"usage": {"input_tokens": 10, "output_tokens": 5}
				}`)),
			},
			mockErr:    nil,
			wantErr:    false,
			checkModel: true,
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
			errMsg:  "request cannot be nil",
		},
		{
			name: "empty model",
			req: &warp.CompletionRequest{
				Model: "",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			wantErr: true,
			errMsg:  "model is required",
		},
		{
			name: "invalid model",
			req: &warp.CompletionRequest{
				Model: "invalid-model",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			wantErr: true,
		},
		{
			name: "http error",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: &http.Response{
				StatusCode: 401,
				Body:       io.NopCloser(bytes.NewBufferString(`{"error":{"message":"Invalid credentials"}}`)),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock HTTP client
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					if tt.mockErr != nil {
						return nil, tt.mockErr
					}
					return tt.mockResp, nil
				},
			}

			provider, err := NewProvider(
				WithCredentials("test-key", "test-secret"),
				WithRegion("us-east-1"),
				WithHTTPClient(mockClient),
			)
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			ctx := context.Background()
			resp, err := provider.Completion(ctx, tt.req)

			if (err != nil) != tt.wantErr {
				t.Errorf("Completion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				if tt.errMsg != "" {
					liteLLMErr, ok := err.(*warp.WarpError)
					if ok && liteLLMErr.Message != tt.errMsg {
						t.Errorf("error message = %q, want %q", liteLLMErr.Message, tt.errMsg)
					}
				}
				return
			}

			if resp == nil {
				t.Error("response is nil")
				return
			}

			if tt.checkModel && resp.Model != tt.req.Model {
				t.Errorf("response model = %q, want %q", resp.Model, tt.req.Model)
			}
		})
	}
}

func TestTransformClaudeRequest(t *testing.T) {
	maxTokens := 1024
	temperature := 0.7

	req := &warp.CompletionRequest{
		Model: "claude-3-opus",
		Messages: []warp.Message{
			{Role: "system", Content: "You are helpful"},
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   &maxTokens,
		Temperature: &temperature,
	}

	bedrockReq := transformClaudeRequest(req)

	// Check required fields
	if bedrockReq["anthropic_version"] != "bedrock-2023-05-31" {
		t.Error("missing anthropic_version")
	}

	if bedrockReq["system"] != "You are helpful" {
		t.Error("system message not extracted")
	}

	messages, ok := bedrockReq["messages"].([]map[string]interface{})
	if !ok {
		t.Fatal("messages field is not a slice")
	}

	if len(messages) != 1 {
		t.Errorf("messages length = %d, want 1", len(messages))
	}

	if bedrockReq["max_tokens"] != 1024 {
		t.Error("max_tokens not set")
	}

	if bedrockReq["temperature"] != 0.7 {
		t.Error("temperature not set")
	}
}

func TestConvertMessagesToLlamaPrompt(t *testing.T) {
	messages := []warp.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
		{Role: "user", Content: "How are you?"},
	}

	prompt := convertMessagesToLlamaPrompt(messages)

	// Check prompt contains expected tokens
	if !bytes.Contains([]byte(prompt), []byte("[INST]")) {
		t.Error("prompt missing [INST] token")
	}

	if !bytes.Contains([]byte(prompt), []byte("[/INST]")) {
		t.Error("prompt missing [/INST] token")
	}

	if !bytes.Contains([]byte(prompt), []byte("<s>")) {
		t.Error("prompt missing <s> token")
	}
}

func TestConvertMessagesToText(t *testing.T) {
	messages := []warp.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi!"},
	}

	text := convertMessagesToText(messages)

	// Check text contains role labels
	if !bytes.Contains([]byte(text), []byte("System:")) {
		t.Error("text missing System label")
	}

	if !bytes.Contains([]byte(text), []byte("User:")) {
		t.Error("text missing User label")
	}

	if !bytes.Contains([]byte(text), []byte("Assistant:")) {
		t.Error("text missing Assistant label")
	}
}
