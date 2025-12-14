// Package main demonstrates advanced callback usage with cost tracking
// and request validation.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/callback"
)

// CostTracker tracks API costs across requests
type CostTracker struct {
	totalCost    float64
	requestCount int64
	mu           sync.Mutex
}

func (ct *CostTracker) AddCost(cost float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.totalCost += cost
	atomic.AddInt64(&ct.requestCount, 1)
}

func (ct *CostTracker) GetStats() (totalCost float64, requests int64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.totalCost, ct.requestCount
}

// RequestValidator validates requests before sending
type RequestValidator struct {
	maxTokens int
}

func (rv *RequestValidator) Validate(req *warp.CompletionRequest) error {
	// Check for empty messages
	if len(req.Messages) == 0 {
		return errors.New("request must have at least one message")
	}

	// Check max tokens limit
	if req.MaxTokens != nil && *req.MaxTokens > rv.maxTokens {
		return fmt.Errorf("max tokens %d exceeds limit of %d", *req.MaxTokens, rv.maxTokens)
	}

	return nil
}

func main() {
	// Initialize components
	costTracker := &CostTracker{}
	validator := &RequestValidator{maxTokens: 4096}

	// Create client with advanced callbacks
	client, err := warp.NewClient(
		warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
		warp.WithCostTracking(true),

		// Before-request: Validate request
		warp.WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			log.Printf("Validating request to %s/%s", event.Provider, event.Model)

			// Validate request
			if req, ok := event.Request.(*warp.CompletionRequest); ok {
				if err := validator.Validate(req); err != nil {
					return fmt.Errorf("validation failed: %w", err)
				}
			}

			return nil
		}),

		// Before-request: Log request metadata
		warp.WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
			log.Printf("Sending request %s to %s/%s",
				event.RequestID, event.Provider, event.Model)
			return nil
		}),

		// Success: Track costs
		warp.WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			costTracker.AddCost(event.Cost)
			log.Printf("Request %s succeeded: $%.4f, %d tokens, %v duration",
				event.RequestID, event.Cost, event.Tokens, event.Duration)
		}),

		// Success: Check budget alerts
		warp.WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
			totalCost, requests := costTracker.GetStats()
			if totalCost > 10.0 {
				log.Printf("WARNING: Total cost $%.2f exceeds budget threshold", totalCost)
			}

			avgCost := totalCost / float64(requests)
			log.Printf("Running average: $%.4f per request", avgCost)
		}),

		// Failure: Log detailed error information
		warp.WithFailureCallback(func(ctx context.Context, event *callback.FailureEvent) {
			log.Printf("Request %s failed after %v: %v",
				event.RequestID, event.Duration, event.Error)
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Example 1: Valid request
	log.Println("\n=== Example 1: Valid Request ===")
	resp1, err := client.Completion(context.Background(), &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{Role: "user", Content: "What is machine learning?"},
		},
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Printf("Response: %s\n", resp1.Choices[0].Message.Content)
	}

	// Example 2: Request with excessive tokens (will be rejected)
	log.Println("\n=== Example 2: Invalid Request (too many tokens) ===")
	maxTokens := 10000
	_, err = client.Completion(context.Background(), &warp.CompletionRequest{
		Model:     "openai/gpt-3.5-turbo",
		MaxTokens: &maxTokens,
		Messages: []warp.Message{
			{Role: "user", Content: "Generate a long story"},
		},
	})
	if err != nil {
		log.Printf("Expected error: %v", err)
	}

	// Example 3: Multiple requests to track cumulative cost
	log.Println("\n=== Example 3: Multiple Requests ===")
	for i := 0; i < 3; i++ {
		_, err := client.Completion(context.Background(), &warp.CompletionRequest{
			Model: "openai/gpt-3.5-turbo",
			Messages: []warp.Message{
				{Role: "user", Content: fmt.Sprintf("Request %d", i+1)},
			},
		})
		if err != nil {
			log.Printf("Request %d failed: %v", i+1, err)
		}
	}

	// Print final stats
	totalCost, requests := costTracker.GetStats()
	log.Printf("\n=== Final Stats ===")
	log.Printf("Total requests: %d", requests)
	log.Printf("Total cost: $%.4f", totalCost)
	if requests > 0 {
		log.Printf("Average cost: $%.4f per request", totalCost/float64(requests))
	}
}
