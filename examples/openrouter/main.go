// Package main demonstrates how to use the OpenRouter provider.
//
// OpenRouter provides unified access to 300+ models from multiple providers
// including OpenAI, Anthropic, Google, Meta, Mistral, and more through a
// single OpenAI-compatible API.
//
// Usage:
//
//	export OPENROUTER_API_KEY="sk-or-v1-..."
//	go run examples/openrouter/main.go
package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider/openrouter"
)

func main() {
	// Get API key from environment
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY environment variable is required")
	}

	// Create OpenRouter provider with custom headers
	provider, err := openrouter.NewProvider(
		openrouter.WithAPIKey(apiKey),
		openrouter.WithHTTPReferer("https://github.com/blue-context/warp"), // Optional but recommended
		openrouter.WithAppTitle("Warp Go SDK Example"),                        // Optional but recommended
	)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	ctx := context.Background()

	// Demonstrate all features
	fmt.Println("=== OpenRouter Provider Examples ===")

	// 1. Basic Chat Completion
	fmt.Println("1. Basic Chat Completion (OpenAI GPT-4o)")
	if err := basicCompletion(ctx, provider); err != nil {
		log.Printf("Basic completion error: %v", err)
	}
	fmt.Println()

	// 2. Streaming Chat
	fmt.Println("2. Streaming Chat (Anthropic Claude Opus 4)")
	if err := streamingCompletion(ctx, provider); err != nil {
		log.Printf("Streaming error: %v", err)
	}
	fmt.Println()

	// 3. Embeddings
	fmt.Println("3. Text Embeddings (OpenAI Ada-002)")
	if err := embeddings(ctx, provider); err != nil {
		log.Printf("Embeddings error: %v", err)
	}
	fmt.Println()

	// 4. Function Calling
	fmt.Println("4. Function Calling (OpenAI GPT-4o)")
	if err := functionCalling(ctx, provider); err != nil {
		log.Printf("Function calling error: %v", err)
	}
	fmt.Println()

	// 5. Vision
	fmt.Println("5. Vision (OpenAI GPT-4o)")
	if err := vision(ctx, provider); err != nil {
		log.Printf("Vision error: %v", err)
	}
	fmt.Println()

	// 6. Auto Router
	fmt.Println("6. Auto Router (Intelligent Model Selection)")
	if err := autoRouter(ctx, provider); err != nil {
		log.Printf("Auto router error: %v", err)
	}
	fmt.Println()

	// 7. List Models
	fmt.Println("7. List Available Models")
	listModels(provider)
	fmt.Println()

	fmt.Println("=== All examples completed ===")
}

// basicCompletion demonstrates basic chat completion
func basicCompletion(ctx context.Context, provider *openrouter.Provider) error {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "openai/gpt-4o",
		Messages: []warp.Message{
			{Role: "user", Content: "What is OpenRouter and why would I use it?"},
		},
		Temperature: warp.Float64Ptr(0.7),
		MaxTokens:   warp.IntPtr(200),
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

// streamingCompletion demonstrates streaming chat
func streamingCompletion(ctx context.Context, provider *openrouter.Provider) error {
	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
		Model: "anthropic/claude-opus-4",
		Messages: []warp.Message{
			{Role: "user", Content: "Tell me a very short story about AI."},
		},
		Temperature: warp.Float64Ptr(0.8),
		MaxTokens:   warp.IntPtr(150),
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
			return fmt.Errorf("stream error: %w", err)
		}

		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			fmt.Print(chunk.Choices[0].Delta.Content)
		}
	}
	fmt.Println()

	return nil
}

// embeddings demonstrates text embedding generation
func embeddings(ctx context.Context, provider *openrouter.Provider) error {
	resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
		Model: "openai/text-embedding-ada-002",
		Input: "OpenRouter provides access to multiple LLM providers through a single API.",
	})
	if err != nil {
		return fmt.Errorf("embedding failed: %w", err)
	}

	fmt.Printf("Generated embedding with %d dimensions\n", len(resp.Data[0].Embedding))
	fmt.Printf("First 5 values: %v\n", resp.Data[0].Embedding[:5])
	fmt.Printf("Tokens used: %d\n", resp.Usage.TotalTokens)

	// Demonstrate batch embeddings
	batchResp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
		Model: "sentence-transformers/all-mpnet-base-v2",
		Input: []string{
			"First sentence",
			"Second sentence",
			"Third sentence",
		},
	})
	if err != nil {
		return fmt.Errorf("batch embedding failed: %w", err)
	}

	fmt.Printf("Batch embeddings: generated %d embeddings\n", len(batchResp.Data))

	return nil
}

// functionCalling demonstrates function calling
func functionCalling(ctx context.Context, provider *openrouter.Provider) error {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "openai/gpt-4o",
		Messages: []warp.Message{
			{Role: "user", Content: "What's the weather like in San Francisco?"},
		},
		Tools: []warp.Tool{
			{
				Type: "function",
				Function: warp.Function{
					Name:        "get_weather",
					Description: "Get the current weather in a location",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
							"unit": map[string]any{
								"type": "string",
								"enum": []string{"celsius", "fahrenheit"},
							},
						},
						"required": []string{"location"},
					},
				},
			},
		},
		ToolChoice: &warp.ToolChoice{Type: "auto"},
	})
	if err != nil {
		return fmt.Errorf("function calling failed: %w", err)
	}

	if len(resp.Choices[0].Message.ToolCalls) > 0 {
		toolCall := resp.Choices[0].Message.ToolCalls[0]
		fmt.Printf("Function called: %s\n", toolCall.Function.Name)
		fmt.Printf("Arguments: %s\n", toolCall.Function.Arguments)
	} else {
		fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
	}

	return nil
}

// vision demonstrates multimodal image understanding
func vision(ctx context.Context, provider *openrouter.Provider) error {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "openai/gpt-4o",
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
							URL: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
						},
					},
				},
			},
		},
		MaxTokens: warp.IntPtr(300),
	})
	if err != nil {
		return fmt.Errorf("vision failed: %w", err)
	}

	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)

	return nil
}

// autoRouter demonstrates OpenRouter's intelligent model selection
func autoRouter(ctx context.Context, provider *openrouter.Provider) error {
	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
		Model: "openrouter/auto", // Auto selects best available model
		Messages: []warp.Message{
			{Role: "user", Content: "Explain quantum computing in simple terms."},
		},
		MaxTokens: warp.IntPtr(200),
	})
	if err != nil {
		return fmt.Errorf("auto router failed: %w", err)
	}

	fmt.Printf("Auto-selected model: %s\n", resp.Model)
	fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)

	return nil
}

// listModels demonstrates listing available models
func listModels(provider *openrouter.Provider) {
	models := provider.ListModels()

	fmt.Printf("Total models available: %d\n\n", len(models))

	// Show chat models
	fmt.Println("Chat Models (first 10):")
	chatCount := 0
	for _, model := range models {
		if model.Capabilities.Completion && chatCount < 10 {
			fmt.Printf("  - %s (context: %d, input: $%.2f/1M, output: $%.2f/1M)\n",
				model.Name,
				model.ContextWindow,
				model.InputCostPer1M,
				model.OutputCostPer1M,
			)
			chatCount++
		}
	}

	// Show embedding models
	fmt.Println("\nEmbedding Models:")
	for _, model := range models {
		if model.Capabilities.Embedding {
			fmt.Printf("  - %s (context: %d, cost: $%.2f/1M)\n",
				model.Name,
				model.ContextWindow,
				model.InputCostPer1M,
			)
		}
	}
}
