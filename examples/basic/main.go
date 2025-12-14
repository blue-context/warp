// Package main demonstrates basic usage of the Warp Go SDK.
//
// This example shows how to:
//   - Create a provider and client
//   - Register the provider with the client
//   - Send a simple completion request
//   - Handle errors properly
//   - Print response and token usage
//
// To run:
//
//	export OPENAI_API_KEY=sk-...
//	go run examples/basic/main.go
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
	provider, err := openai.NewProvider(
		openai.WithAPIKey(apiKey),
	)
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

	// Create completion request
	req := &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant.",
			},
			{
				Role:    "user",
				Content: "What is the capital of France?",
			},
		},
	}

	// Send request
	resp, err := client.Completion(context.Background(), req)
	if err != nil {
		log.Fatalf("Completion failed: %v", err)
	}

	// Print response
	if len(resp.Choices) == 0 {
		log.Fatal("No choices returned in response")
	}
	fmt.Println("Response:", resp.Choices[0].Message.Content)

	if resp.Usage != nil {
		fmt.Printf("Tokens used: %d (prompt: %d, completion: %d)\n",
			resp.Usage.TotalTokens,
			resp.Usage.PromptTokens,
			resp.Usage.CompletionTokens)
	}
}
