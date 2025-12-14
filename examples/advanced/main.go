// Package main demonstrates advanced features of the Warp Go SDK.
//
// This example shows how to:
//   - Use client configuration options
//   - Handle timeouts with context
//   - Use request-level timeout
//   - Handle different error types
//   - Use multimodal messages (demonstration)
//
// To run:
//
//	export OPENAI_API_KEY=sk-...
//	go run examples/advanced/main.go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

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

	// Create client with advanced configuration options
	client, err := warp.NewClient(
		warp.WithTimeout(30*time.Second),      // Default timeout for all requests
		warp.WithRetries(3, time.Second, 2.0), // Max 3 retries with exponential backoff
		warp.WithDebug(true),                  // Enable debug logging
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Register provider with client
	if err := client.RegisterProvider(provider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	// Example 1: Using context with timeout
	example1ContextTimeout(client)

	// Example 2: Using request-level timeout
	example2RequestTimeout(client)

	// Example 3: Error type handling
	example3ErrorHandling(client)

	// Example 4: Multimodal message structure (commented for reference)
	// Uncomment and set OPENAI_API_KEY to test multimodal requests
	// example4MultimodalRequest(client)
}

// example1ContextTimeout demonstrates using context with timeout.
func example1ContextTimeout(client warp.Client) {
	fmt.Println("Example 1: Context with timeout")
	fmt.Println("---")

	// Create context with 10-second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{
				Role:    "user",
				Content: "Say hello in a friendly way!",
			},
		},
	}

	resp, err := client.Completion(ctx, req)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	if len(resp.Choices) == 0 {
		log.Println("No choices returned in response")
		return
	}
	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Println()
}

// example2RequestTimeout demonstrates using request-level timeout.
func example2RequestTimeout(client warp.Client) {
	fmt.Println("Example 2: Request-level timeout")
	fmt.Println("---")

	req := &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{
				Role:    "user",
				Content: "What is the meaning of life?",
			},
		},
		Timeout: 5 * time.Second, // Request-specific timeout
	}

	resp, err := client.Completion(context.Background(), req)
	if err != nil {
		log.Printf("Error: %v", err)
		return
	}

	if len(resp.Choices) == 0 {
		log.Println("No choices returned in response")
		return
	}
	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Println()
}

// example3ErrorHandling demonstrates handling different error types.
func example3ErrorHandling(client warp.Client) {
	fmt.Println("Example 3: Error type handling")
	fmt.Println("---")

	// Create an invalid request to trigger an error
	req := &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{
				Role:    "user",
				Content: "Tell me a joke.",
			},
		},
	}

	_, err := client.Completion(context.Background(), req)

	// Check for specific error types
	if err != nil {
		var rateLimitErr *warp.RateLimitError
		var authErr *warp.AuthenticationError
		var timeoutErr *warp.TimeoutError
		var contextWindowErr *warp.ContextWindowExceededError
		var apiErr *warp.APIError

		switch {
		case errors.As(err, &rateLimitErr):
			fmt.Printf("Rate limited! Retry after: %v\n", rateLimitErr.RetryAfter)
			fmt.Printf("Message: %s\n", rateLimitErr.Message)

		case errors.As(err, &authErr):
			fmt.Printf("Authentication failed: %v\n", authErr)
			fmt.Println("Please check your API key")

		case errors.As(err, &timeoutErr):
			fmt.Printf("Request timed out: %v\n", timeoutErr)
			fmt.Println("Consider increasing timeout or retrying")

		case errors.As(err, &contextWindowErr):
			fmt.Printf("Context window exceeded: %d tokens (max: %d)\n",
				contextWindowErr.Tokens, contextWindowErr.MaxTokens)
			fmt.Println("Consider reducing input size or using a different model")

		case errors.As(err, &apiErr):
			fmt.Printf("API error (status %d): %s\n", apiErr.StatusCode, apiErr.Message)

		default:
			fmt.Printf("Unknown error: %v\n", err)
		}
	} else {
		fmt.Println("Request succeeded (no error to demonstrate)")
	}

	fmt.Println()
}

// example4MultimodalRequest demonstrates multimodal message with vision.
// This example shows how to send images along with text to vision-capable models.
//
// Commented out by default - uncomment and run to test with a real API key.
/*
func example4MultimodalRequest(client warp.Client) {
	fmt.Println("Example 4: Multimodal message (vision)")
	fmt.Println("---")

	// Helper functions to create pointers
	ptr := func(v float64) *float64 { return &v }
	ptrInt := func(v int) *int { return &v }

	// Create a multimodal request with text and image
	req := &warp.CompletionRequest{
		Model: "openai/gpt-4-vision-preview",
		Messages: []warp.Message{
			{
				Role: "user",
				Content: []warp.ContentPart{
					{
						Type: "text",
						Text: "What's in this image?",
					},
					{
						Type: "image_url",
						ImageURL: &warp.ImageURL{
							URL:    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
							Detail: "high", // "auto", "low", or "high"
						},
					},
				},
			},
		},
		Temperature: ptr(0.7),
		MaxTokens:   ptrInt(300),
	}

	resp, err := client.Completion(context.Background(), req)
	if err != nil {
		log.Printf("Multimodal request error: %v", err)
		return
	}

	if len(resp.Choices) == 0 {
		log.Println("No choices returned in response")
		return
	}
	fmt.Printf("Vision response: %s\n", resp.Choices[0].Message.Content)
	fmt.Println()
}
*/
