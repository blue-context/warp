package testutil

import (
	"github.com/blue-context/warp"
)

// CompletionRequestFixture returns a sample completion request for testing.
//
// The returned request has sensible defaults and can be modified for specific tests.
//
// Example:
//
//	req := testutil.CompletionRequestFixture()
//	req.Model = "openai/gpt-4"
//	resp, err := client.Completion(ctx, req)
func CompletionRequestFixture() *warp.CompletionRequest {
	return &warp.CompletionRequest{
		Model: "test-model",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	}
}

// CompletionResponseFixture returns a sample completion response for testing.
//
// The returned response has all fields populated with realistic test data.
//
// Example:
//
//	mock := &testutil.MockProvider{
//	    CompletionFunc: func(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
//	        return testutil.CompletionResponseFixture(), nil
//	    },
//	}
func CompletionResponseFixture() *warp.CompletionResponse {
	return &warp.CompletionResponse{
		ID:      "test-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "test-model",
		Choices: []warp.Choice{
			{
				Index: 0,
				Message: warp.Message{
					Role:    "assistant",
					Content: "Hello! How can I help you?",
				},
				FinishReason: "stop",
			},
		},
		Usage: &warp.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
}

// EmbeddingRequestFixture returns a sample embedding request for testing.
//
// The returned request has sensible defaults and can be modified for specific tests.
//
// Example:
//
//	req := testutil.EmbeddingRequestFixture()
//	req.Model = "openai/text-embedding-3-small"
//	resp, err := client.Embedding(ctx, req)
func EmbeddingRequestFixture() *warp.EmbeddingRequest {
	return &warp.EmbeddingRequest{
		Model: "test-embedding-model",
		Input: "Hello, world!",
	}
}

// EmbeddingResponseFixture returns a sample embedding response for testing.
//
// The returned response has all fields populated with realistic test data,
// including a simple 5-dimensional embedding vector.
//
// Example:
//
//	mock := &testutil.MockProvider{
//	    EmbeddingFunc: func(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
//	        return testutil.EmbeddingResponseFixture(), nil
//	    },
//	}
func EmbeddingResponseFixture() *warp.EmbeddingResponse {
	return &warp.EmbeddingResponse{
		Object: "list",
		Data: []warp.Embedding{
			{
				Object:    "embedding",
				Embedding: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
				Index:     0,
			},
		},
		Model: "test-embedding-model",
		Usage: &warp.EmbeddingUsage{
			PromptTokens: 5,
			TotalTokens:  5,
		},
	}
}
