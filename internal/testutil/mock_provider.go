package testutil

import (
	"context"
	"io"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// MockProvider is a mock provider for testing.
//
// It implements the provider.Provider interface and allows you to customize
// behavior via function fields. It also tracks how many times each method
// was called, which is useful for verification in tests.
//
// Example:
//
//	mock := &testutil.MockProvider{
//	    NameFunc: func() string { return "mock" },
//	    CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
//	        return &warp.CompletionResponse{
//	            ID:      "test-123",
//	            Model:   req.Model,
//	            Choices: []warp.Choice{
//	                {Message: warp.Message{Content: "test response"}},
//	            },
//	        }, nil
//	    },
//	}
//
//	// Use the mock
//	resp, err := mock.Completion(ctx, req)
//
//	// Verify calls
//	if mock.CompletionCalls != 1 {
//	    t.Error("expected 1 completion call")
//	}
type MockProvider struct {
	// Function fields for customizing behavior
	NameFunc             func() string
	CompletionFunc       func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error)
	CompletionStreamFunc func(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error)
	EmbeddingFunc        func(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error)
	TranscriptionFunc    func(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error)
	SupportsFunc         func() provider.Capabilities

	// Call tracking
	CompletionCalls       int
	CompletionStreamCalls int
	EmbeddingCalls        int
	TranscriptionCalls    int
}

// Name returns the provider name.
//
// If NameFunc is set, it calls that function.
// Otherwise, returns "mock" as the default name.
func (m *MockProvider) Name() string {
	if m.NameFunc != nil {
		return m.NameFunc()
	}
	return "mock"
}

// Completion calls the mock function and increments the call counter.
//
// If CompletionFunc is set, it calls that function.
// Otherwise, returns a default mock response with the request model
// and a simple "mock response" message.
func (m *MockProvider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	m.CompletionCalls++
	if m.CompletionFunc != nil {
		return m.CompletionFunc(ctx, req)
	}
	return &warp.CompletionResponse{
		ID:    "mock-123",
		Model: req.Model,
		Choices: []warp.Choice{
			{Message: warp.Message{Content: "mock response"}},
		},
	}, nil
}

// CompletionStream calls the mock function and increments the call counter.
//
// If CompletionStreamFunc is set, it calls that function.
// Otherwise, returns an empty mock stream with no chunks.
func (m *MockProvider) CompletionStream(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error) {
	m.CompletionStreamCalls++
	if m.CompletionStreamFunc != nil {
		return m.CompletionStreamFunc(ctx, req)
	}
	return NewMockStream(), nil
}

// Embedding calls the mock function and increments the call counter.
//
// If EmbeddingFunc is set, it calls that function.
// Otherwise, returns a default mock embedding response with a simple vector.
func (m *MockProvider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	m.EmbeddingCalls++
	if m.EmbeddingFunc != nil {
		return m.EmbeddingFunc(ctx, req)
	}
	return &warp.EmbeddingResponse{
		Object: "list",
		Data: []warp.Embedding{
			{Embedding: []float64{0.1, 0.2, 0.3}},
		},
	}, nil
}

// Transcription provides a mock implementation of audio transcription.
func (m *MockProvider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	if m.TranscriptionFunc != nil {
		return m.TranscriptionFunc(ctx, req)
	}
	return &warp.TranscriptionResponse{
		Text: "Mock transcription text",
	}, nil
}

// Speech provides a mock implementation of text-to-speech.
func (m *MockProvider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, nil
}

// Supports returns capabilities.
//
// If SupportsFunc is set, it calls that function.
// Otherwise, returns all capabilities enabled via provider.AllSupported().
func (m *MockProvider) Supports() interface{} {
	if m.SupportsFunc != nil {
		return m.SupportsFunc()
	}
	return provider.AllSupported()
}
