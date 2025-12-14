// Package main demonstrates how to use the Warp moderation API
// to check content for policy violations.
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

	// Create Warp client
	client, err := warp.NewClient()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Register the provider
	if err := client.RegisterProvider(provider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	ctx := context.Background()

	// Example 1: Check a single text
	fmt.Println("=== Example 1: Single Text Moderation ===")
	checkSingleText(ctx, client)

	// Example 2: Check multiple texts in one request
	fmt.Println("\n=== Example 2: Batch Moderation ===")
	checkMultipleTexts(ctx, client)

	// Example 3: Interpret category scores
	fmt.Println("\n=== Example 3: Detailed Category Analysis ===")
	analyzeCategoryScores(ctx, client)

	// Example 4: Using different moderation models
	fmt.Println("\n=== Example 4: Using Different Models ===")
	useDifferentModels(ctx, client)
}

// checkSingleText demonstrates checking a single piece of text.
func checkSingleText(ctx context.Context, client warp.Client) {
	resp, err := client.Moderation(ctx, &warp.ModerationRequest{
		Input: "Hello, how are you today?",
	})
	if err != nil {
		log.Fatalf("Moderation failed: %v", err)
	}

	result := resp.Results[0]
	fmt.Printf("Text: \"Hello, how are you today?\"\n")
	fmt.Printf("Flagged: %v\n", result.Flagged)
	if result.Flagged {
		fmt.Println("⚠️  Content flagged for policy violations")
		printFlaggedCategories(result.Categories)
	} else {
		fmt.Println("✓ Content is safe")
	}
}

// checkMultipleTexts demonstrates batch moderation of multiple texts.
func checkMultipleTexts(ctx context.Context, client warp.Client) {
	texts := []string{
		"This is a normal, friendly message.",
		"I want to hurt someone.",
		"The weather is nice today.",
	}

	resp, err := client.Moderation(ctx, &warp.ModerationRequest{
		Input: texts,
	})
	if err != nil {
		log.Fatalf("Moderation failed: %v", err)
	}

	for i, result := range resp.Results {
		fmt.Printf("\nText %d: \"%s\"\n", i+1, texts[i])
		fmt.Printf("Flagged: %v\n", result.Flagged)
		if result.Flagged {
			fmt.Println("⚠️  Content flagged:")
			printFlaggedCategories(result.Categories)
		} else {
			fmt.Println("✓ Content is safe")
		}
	}
}

// analyzeCategoryScores demonstrates detailed category score analysis.
func analyzeCategoryScores(ctx context.Context, client warp.Client) {
	resp, err := client.Moderation(ctx, &warp.ModerationRequest{
		Input: "I'm feeling angry and want to lash out.",
	})
	if err != nil {
		log.Fatalf("Moderation failed: %v", err)
	}

	result := resp.Results[0]
	fmt.Printf("Text: \"I'm feeling angry and want to lash out.\"\n")
	fmt.Printf("Flagged: %v\n\n", result.Flagged)

	fmt.Println("Category Scores (0.0 to 1.0, higher = more likely violation):")
	scores := result.CategoryScores
	printScore("Sexual", scores.Sexual, result.Categories.Sexual)
	printScore("Hate", scores.Hate, result.Categories.Hate)
	printScore("Harassment", scores.Harassment, result.Categories.Harassment)
	printScore("Self-harm", scores.SelfHarm, result.Categories.SelfHarm)
	printScore("Sexual/minors", scores.SexualMinors, result.Categories.SexualMinors)
	printScore("Hate/threatening", scores.HateThreatening, result.Categories.HateThreatening)
	printScore("Violence/graphic", scores.ViolenceGraphic, result.Categories.ViolenceGraphic)
	printScore("Self-harm/intent", scores.SelfHarmIntent, result.Categories.SelfHarmIntent)
	printScore("Self-harm/instructions", scores.SelfHarmInstructions, result.Categories.SelfHarmInstructions)
	printScore("Harassment/threatening", scores.HarassmentThreatening, result.Categories.HarassmentThreatening)
	printScore("Violence", scores.Violence, result.Categories.Violence)
}

// useDifferentModels demonstrates using different moderation models.
func useDifferentModels(ctx context.Context, client warp.Client) {
	text := "This is a test message."

	// Use latest model
	respLatest, err := client.Moderation(ctx, &warp.ModerationRequest{
		Model: "openai/text-moderation-latest",
		Input: text,
	})
	if err != nil {
		log.Fatalf("Moderation (latest) failed: %v", err)
	}

	// Use stable model
	respStable, err := client.Moderation(ctx, &warp.ModerationRequest{
		Model: "openai/text-moderation-stable",
		Input: text,
	})
	if err != nil {
		log.Fatalf("Moderation (stable) failed: %v", err)
	}

	fmt.Printf("Text: \"%s\"\n\n", text)
	fmt.Printf("Model: text-moderation-latest\n")
	fmt.Printf("  Flagged: %v\n", respLatest.Results[0].Flagged)
	fmt.Printf("  Violence score: %.4f\n\n", respLatest.Results[0].CategoryScores.Violence)

	fmt.Printf("Model: text-moderation-stable\n")
	fmt.Printf("  Flagged: %v\n", respStable.Results[0].Flagged)
	fmt.Printf("  Violence score: %.4f\n", respStable.Results[0].CategoryScores.Violence)
}

// printFlaggedCategories prints which categories were flagged.
func printFlaggedCategories(categories warp.ModerationCategories) {
	flagged := []string{}

	if categories.Sexual {
		flagged = append(flagged, "Sexual")
	}
	if categories.Hate {
		flagged = append(flagged, "Hate")
	}
	if categories.Harassment {
		flagged = append(flagged, "Harassment")
	}
	if categories.SelfHarm {
		flagged = append(flagged, "Self-harm")
	}
	if categories.SexualMinors {
		flagged = append(flagged, "Sexual/minors")
	}
	if categories.HateThreatening {
		flagged = append(flagged, "Hate/threatening")
	}
	if categories.ViolenceGraphic {
		flagged = append(flagged, "Violence/graphic")
	}
	if categories.SelfHarmIntent {
		flagged = append(flagged, "Self-harm/intent")
	}
	if categories.SelfHarmInstructions {
		flagged = append(flagged, "Self-harm/instructions")
	}
	if categories.HarassmentThreatening {
		flagged = append(flagged, "Harassment/threatening")
	}
	if categories.Violence {
		flagged = append(flagged, "Violence")
	}

	for _, category := range flagged {
		fmt.Printf("  • %s\n", category)
	}
}

// printScore prints a category score with formatting.
func printScore(name string, score float64, flagged bool) {
	flag := " "
	if flagged {
		flag = "⚠"
	}
	fmt.Printf("  %s %-30s %.6f\n", flag, name+":", score)
}
