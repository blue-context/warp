package testutil

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// TestAssert tests all assertion helpers
func TestAssert(t *testing.T) {
	t.Run("NoError", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.NoError(nil)
		// Should not fail
	})

	t.Run("Error", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.Error(errors.New("test error"))
		// Should not fail
	})

	t.Run("Equal", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.Equal(42, 42)
		assert.Equal("hello", "hello")
		assert.Equal([]int{1, 2, 3}, []int{1, 2, 3})
	})

	t.Run("NotEqual", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.NotEqual(42, 43)
		assert.NotEqual("hello", "world")
	})

	t.Run("Nil", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		var ptr *string
		assert.Nil(nil)
		assert.Nil(ptr)
	})

	t.Run("NotNil", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		str := "test"
		assert.NotNil(&str)
		assert.NotNil("test")
	})

	t.Run("True", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.True(true)
		assert.True(1 == 1)
	})

	t.Run("False", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.False(false)
		assert.False(1 == 2)
	})

	t.Run("Contains", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.Contains("hello world", "world")
		assert.Contains("test string", "test")
	})

	t.Run("NotContains", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.NotContains("hello world", "goodbye")
		assert.NotContains("test string", "missing")
	})

	t.Run("Len", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.Len([]int{1, 2, 3}, 3)
		assert.Len("hello", 5)
		assert.Len([]string{}, 0)
	})

	t.Run("Empty", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.Empty([]int{})
		assert.Empty("")
		assert.Empty([]string{})
	})

	t.Run("NotEmpty", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.NotEmpty([]int{1})
		assert.NotEmpty("test")
		assert.NotEmpty([]string{"a"})
	})

	t.Run("Panics", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.Panics(func() {
			panic("test panic")
		})
	})

	t.Run("NotPanics", func(t *testing.T) {
		mockT := &testing.T{}
		assert := New(mockT)
		assert.NotPanics(func() {
			// Normal function
		})
	})
}

// TestMockHTTPClient tests the mock HTTP client
func TestMockHTTPClient(t *testing.T) {
	t.Run("default response", func(t *testing.T) {
		mock := &MockHTTPClient{}
		req, _ := http.NewRequest("GET", "http://example.com", nil)
		resp, err := mock.Do(req)

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if resp.StatusCode != 200 {
			t.Errorf("expected status 200, got: %d", resp.StatusCode)
		}
		if len(mock.RequestsMade) != 1 {
			t.Errorf("expected 1 request tracked, got: %d", len(mock.RequestsMade))
		}
	})

	t.Run("custom DoFunc", func(t *testing.T) {
		mock := &MockHTTPClient{
			DoFunc: func(req *http.Request) (*http.Response, error) {
				return MockResponse(201, `{"status":"created"}`), nil
			},
		}
		req, _ := http.NewRequest("POST", "http://example.com", nil)
		resp, err := mock.Do(req)

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if resp.StatusCode != 201 {
			t.Errorf("expected status 201, got: %d", resp.StatusCode)
		}

		body, _ := io.ReadAll(resp.Body)
		if !strings.Contains(string(body), "created") {
			t.Errorf("expected body to contain 'created', got: %s", body)
		}
	})

	t.Run("tracks multiple requests", func(t *testing.T) {
		mock := &MockHTTPClient{}
		req1, _ := http.NewRequest("GET", "http://example.com/1", nil)
		req2, _ := http.NewRequest("GET", "http://example.com/2", nil)

		mock.Do(req1)
		mock.Do(req2)

		if len(mock.RequestsMade) != 2 {
			t.Errorf("expected 2 requests tracked, got: %d", len(mock.RequestsMade))
		}
	})

	t.Run("MockResponse helper", func(t *testing.T) {
		resp := MockResponse(200, `{"key":"value"}`)

		if resp.StatusCode != 200 {
			t.Errorf("expected status 200, got: %d", resp.StatusCode)
		}

		body, _ := io.ReadAll(resp.Body)
		if !strings.Contains(string(body), "key") {
			t.Errorf("expected body to contain 'key', got: %s", body)
		}
	})

	t.Run("MockErrorResponse helper", func(t *testing.T) {
		resp := MockErrorResponse(401, `{"error":{"message":"Unauthorized"}}`)

		if resp.StatusCode != 401 {
			t.Errorf("expected status 401, got: %d", resp.StatusCode)
		}

		body, _ := io.ReadAll(resp.Body)
		if !strings.Contains(string(body), "Unauthorized") {
			t.Errorf("expected body to contain 'Unauthorized', got: %s", body)
		}
	})

	t.Run("RoundTrip implementation", func(t *testing.T) {
		mock := &MockHTTPClient{
			DoFunc: func(req *http.Request) (*http.Response, error) {
				return MockResponse(200, "ok"), nil
			},
		}

		// Test that it can be used as http.RoundTripper
		client := &http.Client{Transport: mock}
		resp, err := client.Get("http://example.com")

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if resp.StatusCode != 200 {
			t.Errorf("expected status 200, got: %d", resp.StatusCode)
		}
	})
}

// TestMockStream tests the mock stream
func TestMockStream(t *testing.T) {
	t.Run("empty stream", func(t *testing.T) {
		stream := NewMockStream()
		chunk, err := stream.Recv()

		if err != io.EOF {
			t.Errorf("expected io.EOF, got: %v", err)
		}
		if chunk != nil {
			t.Errorf("expected nil chunk, got: %v", chunk)
		}
	})

	t.Run("stream with chunks", func(t *testing.T) {
		chunk1 := &warp.CompletionChunk{
			ID: "chunk-1",
			Choices: []warp.ChunkChoice{
				{Delta: warp.MessageDelta{Content: "Hello"}},
			},
		}
		chunk2 := &warp.CompletionChunk{
			ID: "chunk-2",
			Choices: []warp.ChunkChoice{
				{Delta: warp.MessageDelta{Content: " world"}},
			},
		}

		stream := NewMockStream(chunk1, chunk2)

		// First chunk
		got1, err := stream.Recv()
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if got1.ID != "chunk-1" {
			t.Errorf("expected chunk-1, got: %s", got1.ID)
		}

		// Second chunk
		got2, err := stream.Recv()
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if got2.ID != "chunk-2" {
			t.Errorf("expected chunk-2, got: %s", got2.ID)
		}

		// EOF
		_, err = stream.Recv()
		if err != io.EOF {
			t.Errorf("expected io.EOF, got: %v", err)
		}
	})

	t.Run("close before exhaustion", func(t *testing.T) {
		chunk1 := &warp.CompletionChunk{ID: "chunk-1"}
		stream := NewMockStream(chunk1)

		err := stream.Close()
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}

		// After close, should get EOF
		_, err = stream.Recv()
		if err != io.EOF {
			t.Errorf("expected io.EOF after close, got: %v", err)
		}
	})

	t.Run("multiple closes", func(t *testing.T) {
		stream := NewMockStream()
		err1 := stream.Close()
		err2 := stream.Close()

		if err1 != nil || err2 != nil {
			t.Errorf("expected no errors from multiple closes, got: %v, %v", err1, err2)
		}
	})
}

// TestMockProvider tests the mock provider
func TestMockProvider(t *testing.T) {
	t.Run("default name", func(t *testing.T) {
		mock := &MockProvider{}
		if mock.Name() != "mock" {
			t.Errorf("expected name 'mock', got: %s", mock.Name())
		}
	})

	t.Run("custom name", func(t *testing.T) {
		mock := &MockProvider{
			NameFunc: func() string { return "custom" },
		}
		if mock.Name() != "custom" {
			t.Errorf("expected name 'custom', got: %s", mock.Name())
		}
	})

	t.Run("default completion", func(t *testing.T) {
		mock := &MockProvider{}
		req := &warp.CompletionRequest{Model: "test-model"}
		resp, err := mock.Completion(context.Background(), req)

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if resp.Model != "test-model" {
			t.Errorf("expected model 'test-model', got: %s", resp.Model)
		}
		if len(resp.Choices) != 1 {
			t.Errorf("expected 1 choice, got: %d", len(resp.Choices))
		}
		if mock.CompletionCalls != 1 {
			t.Errorf("expected 1 completion call, got: %d", mock.CompletionCalls)
		}
	})

	t.Run("custom completion", func(t *testing.T) {
		mock := &MockProvider{
			CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
				return &warp.CompletionResponse{
					ID:    "custom-123",
					Model: req.Model,
					Choices: []warp.Choice{
						{Message: warp.Message{Content: "custom response"}},
					},
				}, nil
			},
		}

		req := &warp.CompletionRequest{Model: "test-model"}
		resp, err := mock.Completion(context.Background(), req)

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if resp.ID != "custom-123" {
			t.Errorf("expected ID 'custom-123', got: %s", resp.ID)
		}
	})

	t.Run("default completion stream", func(t *testing.T) {
		mock := &MockProvider{}
		req := &warp.CompletionRequest{Model: "test-model"}
		stream, err := mock.CompletionStream(context.Background(), req)

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if stream == nil {
			t.Fatal("expected non-nil stream")
		}
		if mock.CompletionStreamCalls != 1 {
			t.Errorf("expected 1 completion stream call, got: %d", mock.CompletionStreamCalls)
		}
	})

	t.Run("default embedding", func(t *testing.T) {
		mock := &MockProvider{}
		req := &warp.EmbeddingRequest{Model: "test-embedding"}
		resp, err := mock.Embedding(context.Background(), req)

		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if len(resp.Data) != 1 {
			t.Errorf("expected 1 embedding, got: %d", len(resp.Data))
		}
		if mock.EmbeddingCalls != 1 {
			t.Errorf("expected 1 embedding call, got: %d", mock.EmbeddingCalls)
		}
	})

	t.Run("default supports", func(t *testing.T) {
		mock := &MockProvider{}
		capsInterface := mock.Supports()
		caps, ok := capsInterface.(provider.Capabilities)
		if !ok {
			t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
		}

		if !caps.Completion {
			t.Error("expected completion support")
		}
		if !caps.Streaming {
			t.Error("expected streaming support")
		}
		if !caps.Embedding {
			t.Error("expected embedding support")
		}
	})

	t.Run("custom supports", func(t *testing.T) {
		mock := &MockProvider{
			SupportsFunc: func() provider.Capabilities {
				caps := provider.NoneSupported()
				caps.Completion = true
				return caps
			},
		}
		capsInterface := mock.Supports()
		caps, ok := capsInterface.(provider.Capabilities)
		if !ok {
			t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
		}

		if !caps.Completion {
			t.Error("expected completion support")
		}
		if caps.Streaming {
			t.Error("expected no streaming support")
		}
	})

	t.Run("call tracking", func(t *testing.T) {
		mock := &MockProvider{}
		ctx := context.Background()

		// Make multiple calls
		mock.Completion(ctx, &warp.CompletionRequest{Model: "test"})
		mock.Completion(ctx, &warp.CompletionRequest{Model: "test"})
		mock.CompletionStream(ctx, &warp.CompletionRequest{Model: "test"})
		mock.Embedding(ctx, &warp.EmbeddingRequest{Model: "test"})

		if mock.CompletionCalls != 2 {
			t.Errorf("expected 2 completion calls, got: %d", mock.CompletionCalls)
		}
		if mock.CompletionStreamCalls != 1 {
			t.Errorf("expected 1 completion stream call, got: %d", mock.CompletionStreamCalls)
		}
		if mock.EmbeddingCalls != 1 {
			t.Errorf("expected 1 embedding call, got: %d", mock.EmbeddingCalls)
		}
	})
}

// TestFixtures tests all fixture functions
func TestFixtures(t *testing.T) {
	t.Run("CompletionRequestFixture", func(t *testing.T) {
		req := CompletionRequestFixture()

		if req.Model == "" {
			t.Error("expected non-empty model")
		}
		if len(req.Messages) == 0 {
			t.Error("expected at least one message")
		}
		if req.Messages[0].Role != "user" {
			t.Errorf("expected role 'user', got: %s", req.Messages[0].Role)
		}
	})

	t.Run("CompletionResponseFixture", func(t *testing.T) {
		resp := CompletionResponseFixture()

		if resp.ID == "" {
			t.Error("expected non-empty ID")
		}
		if resp.Object != "chat.completion" {
			t.Errorf("expected object 'chat.completion', got: %s", resp.Object)
		}
		if len(resp.Choices) == 0 {
			t.Error("expected at least one choice")
		}
		if resp.Choices[0].Message.Role != "assistant" {
			t.Errorf("expected role 'assistant', got: %s", resp.Choices[0].Message.Role)
		}
		if resp.Usage == nil {
			t.Error("expected non-nil usage")
		}
	})

	t.Run("EmbeddingRequestFixture", func(t *testing.T) {
		req := EmbeddingRequestFixture()

		if req.Model == "" {
			t.Error("expected non-empty model")
		}
		if req.Input == nil {
			t.Error("expected non-nil input")
		}
	})

	t.Run("EmbeddingResponseFixture", func(t *testing.T) {
		resp := EmbeddingResponseFixture()

		if resp.Object != "list" {
			t.Errorf("expected object 'list', got: %s", resp.Object)
		}
		if len(resp.Data) == 0 {
			t.Error("expected at least one embedding")
		}
		if len(resp.Data[0].Embedding) == 0 {
			t.Error("expected non-empty embedding vector")
		}
		if resp.Usage == nil {
			t.Error("expected non-nil usage")
		}
	})
}
