// Package main demonstrates basic callback usage with Warp.
//
// This example shows how to register callbacks for different stages
// of the request lifecycle: before request, success, and failure.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/callback"
)

func main() {
	// Create client with callbacks
	client, err := warp.NewClient(
		warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
		// Before-request callback: log request details
		warp.WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			log.Printf("[BEFORE] Starting request to %s/%s (ID: %s)",
				event.Provider, event.Model, event.RequestID)
			return nil
		}),
		// Success callback: log response metrics
		warp.WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			log.Printf("[SUCCESS] Request completed in %v",
				event.Duration)
			log.Printf("[SUCCESS] Tokens: %d, Cost: $%.4f",
				event.Tokens, event.Cost)
		}),
		// Failure callback: log errors
		warp.WithFailureCallback(func(ctx context.Context, event *callback.FailureEvent) {
			log.Printf("[FAILURE] Request failed after %v",
				event.Duration)
			log.Printf("[FAILURE] Error: %v", event.Error)
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Make a completion request
	resp, err := client.Completion(context.Background(), &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{Role: "user", Content: "What is the capital of France?"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Print response
	fmt.Println("\nResponse:", resp.Choices[0].Message.Content)
}
