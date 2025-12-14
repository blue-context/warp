// Package main demonstrates embeddings with the Warp Go SDK.
//
// This example shows how to:
//   - Generate embeddings for text
//   - Process embedding vectors
//   - Handle single and batch inputs
//   - Display embedding dimensions and usage
//
// To run:
//
//	export OPENAI_API_KEY=sk-...
//	go run examples/embedding/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider/openai"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create OpenAI provider
	provider, err := openai.NewProvider(openai.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	// Create client
	client, err := warp.NewClient()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Register provider with client
	if err := client.RegisterProvider(provider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	// Example 1: Single text embedding
	fmt.Println("Example 1: Single text embedding")
	fmt.Println("---")

	singleReq := &warp.EmbeddingRequest{
		Model: "openai/text-embedding-ada-002",
		Input: "The quick brown fox jumps over the lazy dog.",
	}

	singleResp, err := client.Embedding(context.Background(), singleReq)
	if err != nil {
		log.Fatalf("Embedding failed: %v", err)
	}

	if len(singleResp.Data) > 0 {
		embedding := singleResp.Data[0].Embedding
		fmt.Printf("Embedding dimensions: %d\n", len(embedding))
		fmt.Printf("First 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
			embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
	}

	if singleResp.Usage != nil {
		fmt.Printf("Tokens used: %d\n", singleResp.Usage.TotalTokens)
	}

	// Example 2: Batch embeddings
	fmt.Println()
	fmt.Println("Example 2: Batch embeddings")
	fmt.Println("---")

	batchReq := &warp.EmbeddingRequest{
		Model: "openai/text-embedding-ada-002",
		Input: []string{
			"Hello, world!",
			"How are you today?",
			"Goodbye, see you later!",
		},
	}

	batchResp, err := client.Embedding(context.Background(), batchReq)
	if err != nil {
		log.Fatalf("Batch embedding failed: %v", err)
	}

	fmt.Printf("Generated %d embeddings\n", len(batchResp.Data))

	for i, data := range batchResp.Data {
		fmt.Printf("Embedding %d: %d dimensions\n", i+1, len(data.Embedding))
	}

	if batchResp.Usage != nil {
		fmt.Printf("Total tokens used: %d\n", batchResp.Usage.TotalTokens)
	}

	fmt.Println()
	fmt.Println("Embeddings completed successfully")
}
