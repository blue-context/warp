// Package main demonstrates streaming with the Warp Go SDK.
//
// This example shows how to:
//   - Use streaming completions
//   - Process chunks as they arrive
//   - Handle stream completion (io.EOF)
//   - Show real-time output
//
// To run:
//
//	export OPENAI_API_KEY=sk-...
//	go run examples/streaming/main.go
package main

import (
	"context"
	"fmt"
	"io"
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

	// Create streaming request
	req := &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{
				Role:    "user",
				Content: "Write a haiku about Go programming.",
			},
		},
	}

	// Send streaming request
	stream, err := client.CompletionStream(context.Background(), req)
	if err != nil {
		log.Fatalf("Streaming failed: %v", err)
	}
	defer stream.Close()

	// Process stream
	fmt.Println("Streaming response:")
	fmt.Println("---")

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			// Stream completed successfully
			break
		}
		if err != nil {
			log.Fatalf("Stream error: %v", err)
		}

		// Print content as it arrives
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}

	fmt.Println()
	fmt.Println("---")
	fmt.Println("Stream completed")
}
