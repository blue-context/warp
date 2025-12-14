package provider

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/blue-context/warp"
)

func TestAllSupported(t *testing.T) {
	caps := AllSupported()

	tests := []struct {
		name  string
		value bool
	}{
		{"Completion", caps.Completion},
		{"Streaming", caps.Streaming},
		{"Embedding", caps.Embedding},
		{"ImageGeneration", caps.ImageGeneration},
		{"Transcription", caps.Transcription},
		{"Speech", caps.Speech},
		{"Moderation", caps.Moderation},
		{"FunctionCalling", caps.FunctionCalling},
		{"Vision", caps.Vision},
		{"JSON", caps.JSON},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.value {
				t.Errorf("AllSupported().%s = false, want true", tt.name)
			}
		})
	}
}

func TestNoneSupported(t *testing.T) {
	caps := NoneSupported()

	tests := []struct {
		name  string
		value bool
	}{
		{"Completion", caps.Completion},
		{"Streaming", caps.Streaming},
		{"Embedding", caps.Embedding},
		{"ImageGeneration", caps.ImageGeneration},
		{"Transcription", caps.Transcription},
		{"Speech", caps.Speech},
		{"Moderation", caps.Moderation},
		{"FunctionCalling", caps.FunctionCalling},
		{"Vision", caps.Vision},
		{"JSON", caps.JSON},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value {
				t.Errorf("NoneSupported().%s = true, want false", tt.name)
			}
		})
	}
}

func TestCapabilities(t *testing.T) {
	tests := []struct {
		name string
		caps Capabilities
		want map[string]bool
	}{
		{
			name: "all enabled",
			caps: AllSupported(),
			want: map[string]bool{
				"Completion":      true,
				"Streaming":       true,
				"Embedding":       true,
				"ImageGeneration": true,
				"Transcription":   true,
				"Speech":          true,
				"Moderation":      true,
				"FunctionCalling": true,
				"Vision":          true,
				"JSON":            true,
			},
		},
		{
			name: "all disabled",
			caps: NoneSupported(),
			want: map[string]bool{
				"Completion":      false,
				"Streaming":       false,
				"Embedding":       false,
				"ImageGeneration": false,
				"Transcription":   false,
				"Speech":          false,
				"Moderation":      false,
				"FunctionCalling": false,
				"Vision":          false,
				"JSON":            false,
			},
		},
		{
			name: "partial support",
			caps: Capabilities{
				Completion:      true,
				Streaming:       true,
				FunctionCalling: true,
			},
			want: map[string]bool{
				"Completion":      true,
				"Streaming":       true,
				"Embedding":       false,
				"ImageGeneration": false,
				"Transcription":   false,
				"Speech":          false,
				"Moderation":      false,
				"FunctionCalling": true,
				"Vision":          false,
				"JSON":            false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.caps.Completion != tt.want["Completion"] {
				t.Errorf("Completion = %v, want %v", tt.caps.Completion, tt.want["Completion"])
			}
			if tt.caps.Streaming != tt.want["Streaming"] {
				t.Errorf("Streaming = %v, want %v", tt.caps.Streaming, tt.want["Streaming"])
			}
			if tt.caps.Embedding != tt.want["Embedding"] {
				t.Errorf("Embedding = %v, want %v", tt.caps.Embedding, tt.want["Embedding"])
			}
			if tt.caps.ImageGeneration != tt.want["ImageGeneration"] {
				t.Errorf("ImageGeneration = %v, want %v", tt.caps.ImageGeneration, tt.want["ImageGeneration"])
			}
			if tt.caps.Transcription != tt.want["Transcription"] {
				t.Errorf("Transcription = %v, want %v", tt.caps.Transcription, tt.want["Transcription"])
			}
			if tt.caps.Speech != tt.want["Speech"] {
				t.Errorf("Speech = %v, want %v", tt.caps.Speech, tt.want["Speech"])
			}
			if tt.caps.Moderation != tt.want["Moderation"] {
				t.Errorf("Moderation = %v, want %v", tt.caps.Moderation, tt.want["Moderation"])
			}
			if tt.caps.FunctionCalling != tt.want["FunctionCalling"] {
				t.Errorf("FunctionCalling = %v, want %v", tt.caps.FunctionCalling, tt.want["FunctionCalling"])
			}
			if tt.caps.Vision != tt.want["Vision"] {
				t.Errorf("Vision = %v, want %v", tt.caps.Vision, tt.want["Vision"])
			}
			if tt.caps.JSON != tt.want["JSON"] {
				t.Errorf("JSON = %v, want %v", tt.caps.JSON, tt.want["JSON"])
			}
		})
	}
}

// MockProvider is a test implementation of the Provider interface
type MockProvider struct {
	name         string
	capabilities Capabilities
	completionFn func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error)
	streamFn     func(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error)
	embeddingFn  func(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error)
	imageFn      func(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error)
}

func (m *MockProvider) Name() string {
	return m.name
}

func (m *MockProvider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	if m.completionFn != nil {
		return m.completionFn(ctx, req)
	}
	return &warp.CompletionResponse{
		ID:    "mock-completion",
		Model: req.Model,
		Choices: []warp.Choice{
			{
				Index:   0,
				Message: warp.Message{Role: "assistant", Content: "mock response"},
			},
		},
	}, nil
}

func (m *MockProvider) CompletionStream(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error) {
	if m.streamFn != nil {
		return m.streamFn(ctx, req)
	}
	return &MockStream{}, nil
}

func (m *MockProvider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	if m.embeddingFn != nil {
		return m.embeddingFn(ctx, req)
	}
	return &warp.EmbeddingResponse{
		Object: "list",
		Model:  req.Model,
		Data: []warp.Embedding{
			{
				Object:    "embedding",
				Embedding: []float64{0.1, 0.2, 0.3},
				Index:     0,
			},
		},
	}, nil
}

func (m *MockProvider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	if m.imageFn != nil {
		return m.imageFn(ctx, req)
	}
	return &warp.ImageGenerationResponse{
		Created: 1234567890,
		Data: []warp.ImageData{
			{
				URL: "https://example.com/mock-image.png",
			},
		},
	}, nil
}

func (m *MockProvider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return &warp.TranscriptionResponse{
		Text: "Mock transcription text",
	}, nil
}

func (m *MockProvider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, nil
}

func (m *MockProvider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return &warp.ImageGenerationResponse{
		Created: 1234567890,
		Data: []warp.ImageData{
			{
				URL: "https://example.com/mock-edited-image.png",
			},
		},
	}, nil
}

func (m *MockProvider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return &warp.ImageGenerationResponse{
		Created: 1234567890,
		Data: []warp.ImageData{
			{
				URL: "https://example.com/mock-variation-image.png",
			},
		},
	}, nil
}

func (m *MockProvider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return &warp.ModerationResponse{
		ID:    "mock-moderation",
		Model: "mock-moderation-model",
		Results: []warp.ModerationResult{
			{
				Flagged: false,
				Categories: warp.ModerationCategories{
					Hate:            false,
					HateThreatening: false,
					SelfHarm:        false,
					Sexual:          false,
					SexualMinors:    false,
					Violence:        false,
					ViolenceGraphic: false,
				},
			},
		},
	}, nil
}

func (m *MockProvider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	results := make([]warp.RerankResult, len(req.Documents))
	for i := range req.Documents {
		results[i] = warp.RerankResult{
			Index:          i,
			RelevanceScore: 0.5,
		}
	}
	return &warp.RerankResponse{
		ID:      "mock-rerank",
		Results: results,
	}, nil
}

func (m *MockProvider) GetModelInfo(model string) *ModelInfo {
	return nil
}

func (m *MockProvider) ListModels() []*ModelInfo {
	return []*ModelInfo{}
}

func (m *MockProvider) Supports() interface{} {
	return m.capabilities
}

// MockStream is a test implementation of the Stream interface
type MockStream struct {
	chunks   []*warp.CompletionChunk
	index    int
	closeErr error
	closed   bool
}

func (s *MockStream) Recv() (*warp.CompletionChunk, error) {
	if s.closed {
		return nil, errors.New("stream closed")
	}

	if s.index >= len(s.chunks) {
		return nil, io.EOF
	}

	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}

func (s *MockStream) Close() error {
	s.closed = true
	return s.closeErr
}

func TestMockProvider(t *testing.T) {
	provider := &MockProvider{
		name: "mock",
		capabilities: Capabilities{
			Completion: true,
			Streaming:  true,
			Embedding:  true,
		},
	}

	t.Run("Name", func(t *testing.T) {
		if got := provider.Name(); got != "mock" {
			t.Errorf("Name() = %q, want %q", got, "mock")
		}
	})

	t.Run("Supports", func(t *testing.T) {
		capsInterface := provider.Supports()
		caps, ok := capsInterface.(Capabilities)
		if !ok {
			t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
		}
		if !caps.Completion {
			t.Error("Supports().Completion = false, want true")
		}
		if !caps.Streaming {
			t.Error("Supports().Streaming = false, want true")
		}
		if !caps.Embedding {
			t.Error("Supports().Embedding = false, want true")
		}
		if caps.Vision {
			t.Error("Supports().Vision = true, want false")
		}
	})

	t.Run("Completion", func(t *testing.T) {
		req := &warp.CompletionRequest{
			Model: "test-model",
			Messages: []warp.Message{
				{Role: "user", Content: "test"},
			},
		}

		resp, err := provider.Completion(context.Background(), req)
		if err != nil {
			t.Fatalf("Completion() error = %v", err)
		}

		if resp.ID != "mock-completion" {
			t.Errorf("Completion() ID = %q, want %q", resp.ID, "mock-completion")
		}
		if resp.Model != "test-model" {
			t.Errorf("Completion() Model = %q, want %q", resp.Model, "test-model")
		}
		if len(resp.Choices) != 1 {
			t.Errorf("Completion() len(Choices) = %d, want 1", len(resp.Choices))
		}
	})

	t.Run("Embedding", func(t *testing.T) {
		req := &warp.EmbeddingRequest{
			Model: "test-model",
			Input: "test input",
		}

		resp, err := provider.Embedding(context.Background(), req)
		if err != nil {
			t.Fatalf("Embedding() error = %v", err)
		}

		if resp.Model != "test-model" {
			t.Errorf("Embedding() Model = %q, want %q", resp.Model, "test-model")
		}
		if len(resp.Data) != 1 {
			t.Errorf("Embedding() len(Data) = %d, want 1", len(resp.Data))
		}
		if len(resp.Data[0].Embedding) != 3 {
			t.Errorf("Embedding() len(Data[0].Embedding) = %d, want 3", len(resp.Data[0].Embedding))
		}
	})

	t.Run("CompletionStream", func(t *testing.T) {
		req := &warp.CompletionRequest{
			Model: "test-model",
			Messages: []warp.Message{
				{Role: "user", Content: "test"},
			},
		}

		stream, err := provider.CompletionStream(context.Background(), req)
		if err != nil {
			t.Fatalf("CompletionStream() error = %v", err)
		}
		defer stream.Close()

		// Read from empty stream should return EOF
		_, err = stream.Recv()
		if err != io.EOF {
			t.Errorf("Recv() error = %v, want io.EOF", err)
		}
	})
}

func TestMockProviderCustomFunctions(t *testing.T) {
	t.Run("custom completion function", func(t *testing.T) {
		provider := &MockProvider{
			name: "custom",
			completionFn: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
				return nil, errors.New("custom error")
			},
		}

		_, err := provider.Completion(context.Background(), &warp.CompletionRequest{})
		if err == nil {
			t.Error("Completion() error = nil, want error")
		}
		if err.Error() != "custom error" {
			t.Errorf("Completion() error = %q, want %q", err.Error(), "custom error")
		}
	})

	t.Run("custom stream function", func(t *testing.T) {
		provider := &MockProvider{
			name: "custom",
			streamFn: func(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error) {
				return nil, errors.New("stream error")
			},
		}

		_, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{})
		if err == nil {
			t.Error("CompletionStream() error = nil, want error")
		}
		if err.Error() != "stream error" {
			t.Errorf("CompletionStream() error = %q, want %q", err.Error(), "stream error")
		}
	})

	t.Run("custom embedding function", func(t *testing.T) {
		provider := &MockProvider{
			name: "custom",
			embeddingFn: func(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
				return nil, errors.New("embedding error")
			},
		}

		_, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{})
		if err == nil {
			t.Error("Embedding() error = nil, want error")
		}
		if err.Error() != "embedding error" {
			t.Errorf("Embedding() error = %q, want %q", err.Error(), "embedding error")
		}
	})
}

func TestMockStream(t *testing.T) {
	t.Run("empty stream", func(t *testing.T) {
		stream := &MockStream{}

		chunk, err := stream.Recv()
		if err != io.EOF {
			t.Errorf("Recv() error = %v, want io.EOF", err)
		}
		if chunk != nil {
			t.Errorf("Recv() chunk = %v, want nil", chunk)
		}

		err = stream.Close()
		if err != nil {
			t.Errorf("Close() error = %v, want nil", err)
		}
	})

	t.Run("stream with chunks", func(t *testing.T) {
		stream := &MockStream{
			chunks: []*warp.CompletionChunk{
				{ID: "chunk-1", Model: "test"},
				{ID: "chunk-2", Model: "test"},
			},
		}

		// Read first chunk
		chunk, err := stream.Recv()
		if err != nil {
			t.Fatalf("Recv() error = %v", err)
		}
		if chunk.ID != "chunk-1" {
			t.Errorf("Recv() chunk.ID = %q, want %q", chunk.ID, "chunk-1")
		}

		// Read second chunk
		chunk, err = stream.Recv()
		if err != nil {
			t.Fatalf("Recv() error = %v", err)
		}
		if chunk.ID != "chunk-2" {
			t.Errorf("Recv() chunk.ID = %q, want %q", chunk.ID, "chunk-2")
		}

		// Read EOF
		chunk, err = stream.Recv()
		if err != io.EOF {
			t.Errorf("Recv() error = %v, want io.EOF", err)
		}
		if chunk != nil {
			t.Errorf("Recv() chunk = %v, want nil", chunk)
		}

		err = stream.Close()
		if err != nil {
			t.Errorf("Close() error = %v, want nil", err)
		}
	})

	t.Run("close error", func(t *testing.T) {
		closeErr := errors.New("close error")
		stream := &MockStream{closeErr: closeErr}

		err := stream.Close()
		if err != closeErr {
			t.Errorf("Close() error = %v, want %v", err, closeErr)
		}
	})

	t.Run("recv after close", func(t *testing.T) {
		stream := &MockStream{
			chunks: []*warp.CompletionChunk{
				{ID: "chunk-1", Model: "test"},
			},
		}

		err := stream.Close()
		if err != nil {
			t.Fatalf("Close() error = %v", err)
		}

		_, err = stream.Recv()
		if err == nil {
			t.Error("Recv() after Close() error = nil, want error")
		}
	})
}
