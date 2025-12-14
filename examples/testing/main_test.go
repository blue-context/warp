// Package testing demonstrates how to test code that uses the Warp Go SDK.
//
// This example shows how to:
//   - Use MockProvider for testing
//   - Test error conditions
//   - Verify request handling
//   - Test streaming responses
//
// To run:
//
//	go test -v examples/testing/main_test.go
package testing

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/internal/testutil"
)

// TestBasicCompletion demonstrates testing a basic completion request.
func TestBasicCompletion(t *testing.T) {
	// Create a mock provider
	mock := &testutil.MockProvider{
		CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
			return &warp.CompletionResponse{
				ID:    "test-123",
				Model: req.Model,
				Choices: []warp.Choice{
					{
						Message: warp.Message{
							Role:    "assistant",
							Content: "Hello! This is a test response.",
						},
						FinishReason: "stop",
					},
				},
				Usage: &warp.Usage{
					PromptTokens:     10,
					CompletionTokens: 8,
					TotalTokens:      18,
				},
			}, nil
		},
	}

	// Create client and register mock provider
	client, err := warp.NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Make request
	req := &warp.CompletionRequest{
		Model: "mock/test-model",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	// Verify response
	if len(resp.Choices) == 0 {
		t.Fatal("Expected at least one choice in response")
	}

	if resp.Choices[0].Message.Content != "Hello! This is a test response." {
		t.Errorf("Unexpected response content: %s", resp.Choices[0].Message.Content)
	}

	// Verify call count
	if mock.CompletionCalls != 1 {
		t.Errorf("Expected 1 completion call, got %d", mock.CompletionCalls)
	}
}

// TestErrorHandling demonstrates testing error conditions.
func TestErrorHandling(t *testing.T) {
	testErr := errors.New("test error")

	// Create a mock that returns an error
	mock := &testutil.MockProvider{
		CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
			return nil, testErr
		},
	}

	client, err := warp.NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Make request
	req := &warp.CompletionRequest{
		Model: "mock/test-model",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	_, err = client.Completion(context.Background(), req)
	if err == nil {
		t.Fatal("Expected error, got nil")
	}

	// Verify the error is what we expect
	if !errors.Is(err, testErr) {
		t.Errorf("Expected test error, got: %v", err)
	}
}

// TestStreamingCompletion demonstrates testing streaming responses.
func TestStreamingCompletion(t *testing.T) {
	// Create chunks to stream
	finishReason := "stop"
	chunks := []*warp.CompletionChunk{
		{
			ID:    "chunk-1",
			Model: "mock/test-model",
			Choices: []warp.ChunkChoice{
				{
					Delta: warp.MessageDelta{
						Role:    "assistant",
						Content: "Hello",
					},
				},
			},
		},
		{
			ID:    "chunk-2",
			Model: "mock/test-model",
			Choices: []warp.ChunkChoice{
				{
					Delta: warp.MessageDelta{
						Content: " world",
					},
				},
			},
		},
		{
			ID:    "chunk-3",
			Model: "mock/test-model",
			Choices: []warp.ChunkChoice{
				{
					Delta: warp.MessageDelta{
						Content: "!",
					},
					FinishReason: &finishReason,
				},
			},
		},
	}

	// Create mock stream
	mockStream := testutil.NewMockStreamWithChunks(chunks)

	// Create mock provider
	mock := &testutil.MockProvider{
		CompletionStreamFunc: func(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error) {
			return mockStream, nil
		},
	}

	client, err := warp.NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Make streaming request
	req := &warp.CompletionRequest{
		Model:  "mock/test-model",
		Messages: []warp.Message{
			{Role: "user", Content: "Say hello"},
		},
	}

	stream, err := client.CompletionStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CompletionStream failed: %v", err)
	}
	defer stream.Close()

	// Collect chunks
	var receivedChunks []*warp.CompletionChunk
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Stream recv error: %v", err)
		}
		receivedChunks = append(receivedChunks, chunk)
	}

	// Verify we got all chunks
	if len(receivedChunks) != len(chunks) {
		t.Errorf("Expected %d chunks, got %d", len(chunks), len(receivedChunks))
	}

	// Verify call count
	if mock.CompletionStreamCalls != 1 {
		t.Errorf("Expected 1 stream call, got %d", mock.CompletionStreamCalls)
	}
}

// TestEmptyResponse demonstrates testing edge cases like empty responses.
func TestEmptyResponse(t *testing.T) {
	// Create a mock that returns empty choices
	mock := &testutil.MockProvider{
		CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
			return &warp.CompletionResponse{
				ID:      "test-123",
				Model:   req.Model,
				Choices: []warp.Choice{}, // Empty choices
			}, nil
		},
	}

	client, err := warp.NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	if err := client.RegisterProvider(mock); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Make request
	req := &warp.CompletionRequest{
		Model: "mock/test-model",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	resp, err := client.Completion(context.Background(), req)
	if err != nil {
		t.Fatalf("Completion failed: %v", err)
	}

	// Verify we handle empty choices properly
	if len(resp.Choices) != 0 {
		t.Errorf("Expected empty choices, got %d", len(resp.Choices))
	}
}

// TestTableDrivenTests demonstrates table-driven testing pattern.
func TestTableDrivenTests(t *testing.T) {
	tests := []struct {
		name          string
		model         string
		messages      []warp.Message
		mockResponse  *warp.CompletionResponse
		mockError     error
		wantErr       bool
		wantContent   string
		wantCallCount int
	}{
		{
			name:  "successful completion",
			model: "mock/gpt-4",
			messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
			mockResponse: &warp.CompletionResponse{
				ID:    "test-1",
				Model: "mock/gpt-4",
				Choices: []warp.Choice{
					{Message: warp.Message{Content: "Hi there!"}},
				},
			},
			wantErr:       false,
			wantContent:   "Hi there!",
			wantCallCount: 1,
		},
		{
			name:  "provider error",
			model: "mock/gpt-4",
			messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
			mockError:     errors.New("provider unavailable"),
			wantErr:       true,
			wantCallCount: 1,
		},
		{
			name:  "empty response",
			model: "mock/gpt-4",
			messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
			mockResponse: &warp.CompletionResponse{
				ID:      "test-2",
				Model:   "mock/gpt-4",
				Choices: []warp.Choice{},
			},
			wantErr:       false,
			wantContent:   "",
			wantCallCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock with test-specific behavior
			mock := &testutil.MockProvider{
				CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
					if tt.mockError != nil {
						return nil, tt.mockError
					}
					return tt.mockResponse, nil
				},
			}

			client, err := warp.NewClient()
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer client.Close()

			if err := client.RegisterProvider(mock); err != nil {
				t.Fatalf("Failed to register provider: %v", err)
			}

			// Make request
			req := &warp.CompletionRequest{
				Model:    tt.model,
				Messages: tt.messages,
			}

			resp, err := client.Completion(context.Background(), req)

			// Check error expectation
			if (err != nil) != tt.wantErr {
				t.Errorf("wantErr %v, got error: %v", tt.wantErr, err)
				return
			}

			// Check content if no error expected
			if !tt.wantErr && tt.wantContent != "" {
				if len(resp.Choices) == 0 {
					t.Error("Expected choices in response")
					return
				}
				if resp.Choices[0].Message.Content != tt.wantContent {
					t.Errorf("wantContent %q, got %q", tt.wantContent, resp.Choices[0].Message.Content)
				}
			}

			// Verify call count
			if mock.CompletionCalls != tt.wantCallCount {
				t.Errorf("wantCallCount %d, got %d", tt.wantCallCount, mock.CompletionCalls)
			}
		})
	}
}
