package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider/vllmsemanticrouter"
)

func main() {
	// Create vLLM Semantic Router provider
	// Defaults to localhost:8801 for main API, localhost:8080 for classification
	provider, err := vllmsemanticrouter.NewProvider(
		// Optional: Custom base URL
		// vllmsemanticrouter.WithBaseURL("http://router.example.com:8801"),

		// Optional: Custom classification URL
		// vllmsemanticrouter.WithClassificationURL("http://classifier.example.com:8080"),

		// Optional: API key (if your deployment requires authentication)
		// vllmsemanticrouter.WithAPIKey(os.Getenv("VLLM_ROUTER_API_KEY")),
	)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	ctx := context.Background()

	// Example 1: Semantic Routing with "auto" model
	fmt.Println("=== Example 1: Semantic Routing ===")
	semanticRoutingExample(ctx, provider)

	// Example 2: Streaming with semantic routing
	fmt.Println("\n=== Example 2: Streaming with Semantic Routing ===")
	streamingExample(ctx, provider)

	// Example 3: Function calling with semantic routing
	fmt.Println("\n=== Example 3: Function Calling ===")
	functionCallingExample(ctx, provider)
}

// semanticRoutingExample demonstrates the "auto" model for intelligent routing
func semanticRoutingExample(ctx context.Context, provider *vllmsemanticrouter.Provider) {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "auto", // Semantic router selects optimal backend model
		Messages: []warp.Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant that explains complex topics simply.",
			},
			{
				Role:    "user",
				Content: "Explain quantum computing in simple terms.",
			},
		},
		Temperature: warp.Float64Ptr(0.7),
		MaxTokens:   warp.IntPtr(500),
	})
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Model routed to: %s\n", resp.Model)
	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	fmt.Printf("Usage: %d prompt tokens, %d completion tokens\n",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
}

// streamingExample demonstrates streaming with semantic routing
func streamingExample(ctx context.Context, provider *vllmsemanticrouter.Provider) {
	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
		Model: "auto", // Semantic routing works with streaming too
		Messages: []warp.Message{
			{
				Role:    "user",
				Content: "Write a short poem about artificial intelligence.",
			},
		},
		Temperature: warp.Float64Ptr(0.8),
		MaxTokens:   warp.IntPtr(200),
	})
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}
	defer stream.Close()

	fmt.Print("Streaming response: ")
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("\nError receiving chunk: %v\n", err)
			return
		}

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	fmt.Println()
}

// functionCallingExample demonstrates function calling with semantic routing
func functionCallingExample(ctx context.Context, provider *vllmsemanticrouter.Provider) {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "auto",
		Messages: []warp.Message{
			{
				Role:    "user",
				Content: "What's the weather like in San Francisco?",
			},
		},
		Tools: []warp.Tool{
			{
				Type: "function",
				Function: warp.FunctionDefinition{
					Name:        "get_weather",
					Description: "Get the current weather for a location",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
							"unit": map[string]interface{}{
								"type": "string",
								"enum": []string{"celsius", "fahrenheit"},
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
		ToolChoice: "auto",
	})
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	if len(resp.Choices) > 0 && len(resp.Choices[0].Message.ToolCalls) > 0 {
		toolCall := resp.Choices[0].Message.ToolCalls[0]
		fmt.Printf("Function called: %s\n", toolCall.Function.Name)
		fmt.Printf("Arguments: %s\n", toolCall.Function.Arguments)
	} else {
		fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	}
}

// Additional example: JSON mode with semantic routing
func jsonModeExample(ctx context.Context, provider *vllmsemanticrouter.Provider) {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "auto",
		Messages: []warp.Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant that responds in JSON format.",
			},
			{
				Role:    "user",
				Content: "List three benefits of cloud computing in JSON format with 'benefit' and 'description' fields.",
			},
		},
		ResponseFormat: &warp.ResponseFormat{
			Type: "json_object",
		},
		Temperature: warp.Float64Ptr(0.5),
	})
	if err != nil {
		log.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("JSON Response:\n%s\n", resp.Choices[0].Message.Content)
}

// Example output:
//
// === Example 1: Semantic Routing ===
// Model routed to: meta-llama/Llama-3-70b-chat-hf
// Response: Quantum computing is a revolutionary technology that uses the principles of quantum mechanics...
// Usage: 45 prompt tokens, 156 completion tokens
//
// === Example 2: Streaming with Semantic Routing ===
// Streaming response: In circuits deep, where logic flows,
// A mind emerges, silicon grows.
// Intelligence born from human thought,
// In data's realm, wisdom is sought...
//
// === Example 3: Function Calling ===
// Function called: get_weather
// Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}

func init() {
	// Check if running against actual vLLM Semantic Router deployment
	if os.Getenv("VLLM_ROUTER_URL") == "" {
		fmt.Println("Note: This example expects vLLM Semantic Router running on localhost:8801")
		fmt.Println("Set VLLM_ROUTER_URL environment variable to use a different endpoint")
		fmt.Println()
	}
}
