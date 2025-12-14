// Package main demonstrates how to use the vLLM provider.
//
// vLLM is a self-hosted inference engine for large language models.
// This example shows how to:
// - Create a vLLM provider
// - Send completion requests
// - Stream completion responses
// - Generate embeddings
// - Rerank documents
package main

import (
	"context"
	"fmt"
	"io"
	"log"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider/vllm"
)

func main() {
	// Create vLLM provider
	// By default, connects to http://localhost:8000
	provider, err := vllm.NewProvider(
		vllm.WithBaseURL("http://localhost:8000"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Example 1: Basic completion
	fmt.Println("=== Example 1: Basic Completion ===")
	if err := basicCompletion(provider); err != nil {
		log.Printf("Basic completion failed: %v", err)
	}

	// Example 2: Streaming completion
	fmt.Println("\n=== Example 2: Streaming Completion ===")
	if err := streamingCompletion(provider); err != nil {
		log.Printf("Streaming completion failed: %v", err)
	}

	// Example 3: Completion with parameters
	fmt.Println("\n=== Example 3: Completion with Parameters ===")
	if err := completionWithParameters(provider); err != nil {
		log.Printf("Completion with parameters failed: %v", err)
	}

	// Example 4: Embeddings (requires pooling-compatible model)
	fmt.Println("\n=== Example 4: Embeddings ===")
	if err := embeddingsExample(provider); err != nil {
		log.Printf("Embeddings failed (may require different model): %v", err)
	}

	// Example 5: Reranking (requires reranking-compatible model)
	fmt.Println("\n=== Example 5: Document Reranking ===")
	if err := rerankingExample(provider); err != nil {
		log.Printf("Reranking failed (may require different model): %v", err)
	}
}

// basicCompletion demonstrates a simple completion request.
func basicCompletion(provider *vllm.Provider) error {
	resp, err := provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "meta-llama/Llama-2-7b-hf",
		Messages: []warp.Message{
			{Role: "user", Content: "What is the capital of France?"},
		},
	})
	if err != nil {
		return fmt.Errorf("completion failed: %w", err)
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Printf("Tokens used: %d (prompt: %d, completion: %d)\n",
		resp.Usage.TotalTokens,
		resp.Usage.PromptTokens,
		resp.Usage.CompletionTokens,
	)

	return nil
}

// streamingCompletion demonstrates streaming responses.
func streamingCompletion(provider *vllm.Provider) error {
	stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "meta-llama/Llama-2-7b-hf",
		Messages: []warp.Message{
			{Role: "user", Content: "Tell me a short story about a robot."},
		},
	})
	if err != nil {
		return fmt.Errorf("stream creation failed: %w", err)
	}
	defer stream.Close()

	fmt.Print("Response: ")
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("stream recv failed: %w", err)
		}

		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	fmt.Println()

	return nil
}

// completionWithParameters demonstrates using various parameters.
func completionWithParameters(provider *vllm.Provider) error {
	resp, err := provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "meta-llama/Llama-2-7b-hf",
		Messages: []warp.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "Explain quantum computing in one sentence."},
		},
		Temperature: warp.Float64Ptr(0.7),
		MaxTokens:   warp.IntPtr(50),
		TopP:        warp.Float64Ptr(0.9),
	})
	if err != nil {
		return fmt.Errorf("completion failed: %w", err)
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)

	return nil
}

// embeddingsExample demonstrates generating embeddings.
//
// Note: This requires vLLM to be running with a pooling-compatible model
// like intfloat/e5-small or sentence-transformers models.
func embeddingsExample(provider *vllm.Provider) error {
	resp, err := provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "intfloat/e5-small",
		Input: []string{
			"The quick brown fox jumps over the lazy dog",
			"Machine learning is a subset of artificial intelligence",
		},
	})
	if err != nil {
		return fmt.Errorf("embedding failed: %w", err)
	}

	fmt.Printf("Generated %d embeddings\n", len(resp.Data))
	for i, emb := range resp.Data {
		fmt.Printf("Embedding %d: dimension=%d, first 3 values=[%.4f, %.4f, %.4f]\n",
			i,
			len(emb.Embedding),
			emb.Embedding[0],
			emb.Embedding[1],
			emb.Embedding[2],
		)
	}

	return nil
}

// rerankingExample demonstrates document reranking.
//
// Note: This requires vLLM to be running with a reranking-compatible model
// like BAAI/bge-reranker-large.
func rerankingExample(provider *vllm.Provider) error {
	resp, err := provider.Rerank(context.Background(), &warp.RerankRequest{
		Model: "BAAI/bge-reranker-large",
		Query: "What is the capital of France?",
		Documents: []string{
			"Paris is the capital of France and its largest city.",
			"London is the capital of England and the United Kingdom.",
			"France is a country in Western Europe.",
			"The Eiffel Tower is located in Paris.",
		},
		TopN:            warp.IntPtr(3),
		ReturnDocuments: warp.BoolPtr(true),
	})
	if err != nil {
		return fmt.Errorf("reranking failed: %w", err)
	}

	fmt.Printf("Reranked %d documents:\n", len(resp.Results))
	for i, result := range resp.Results {
		fmt.Printf("%d. Score: %.4f - %s\n",
			i+1,
			result.RelevanceScore,
			result.Document,
		)
	}

	return nil
}
