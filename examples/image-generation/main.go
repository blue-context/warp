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

	// Register provider
	if err := client.RegisterProvider(provider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	// Example 1: Basic image generation with DALL-E 3
	fmt.Println("Example 1: Basic DALL-E 3 image generation")
	fmt.Println("===========================================")

	resp1, err := client.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:   "openai/dall-e-3",
		Prompt:  "A cute baby sea otter floating on its back in the ocean, watercolor style",
		Size:    "1024x1024",
		Quality: "standard",
	})
	if err != nil {
		log.Fatalf("Image generation failed: %v", err)
	}

	fmt.Printf("Generated %d image(s)\n", len(resp1.Data))
	for i, img := range resp1.Data {
		fmt.Printf("\nImage %d:\n", i+1)
		fmt.Printf("  URL: %s\n", img.URL)
		if img.RevisedPrompt != "" {
			fmt.Printf("  Revised Prompt: %s\n", img.RevisedPrompt)
		}

		// Save to file
		filename := fmt.Sprintf("otter_%d.png", i+1)
		if err := img.SaveToFile(context.Background(), filename); err != nil {
			log.Printf("Failed to save image: %v", err)
		} else {
			fmt.Printf("  Saved to: %s\n", filename)
		}
	}

	// Example 2: High-definition image with vivid style
	fmt.Println("\n\nExample 2: HD image with vivid style")
	fmt.Println("=====================================")

	resp2, err := client.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:   "openai/dall-e-3",
		Prompt:  "A futuristic cityscape at sunset with flying cars and neon lights",
		Size:    "1792x1024", // Wide format
		Quality: "hd",
		Style:   "vivid",
	})
	if err != nil {
		log.Fatalf("Image generation failed: %v", err)
	}

	fmt.Printf("Generated HD image\n")
	fmt.Printf("  URL: %s\n", resp2.Data[0].URL)
	if resp2.Data[0].RevisedPrompt != "" {
		fmt.Printf("  Revised Prompt: %s\n", resp2.Data[0].RevisedPrompt)
	}

	// Save to file
	if err := resp2.Data[0].SaveToFile(context.Background(), "cityscape_hd.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: cityscape_hd.png")
	}

	// Example 3: Natural style portrait
	fmt.Println("\n\nExample 3: Natural style portrait")
	fmt.Println("==================================")

	resp3, err := client.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:  "openai/dall-e-3",
		Prompt: "A professional portrait photograph of a person working at a modern office desk",
		Size:   "1024x1024",
		Style:  "natural",
	})
	if err != nil {
		log.Fatalf("Image generation failed: %v", err)
	}

	fmt.Printf("Generated natural style image\n")
	fmt.Printf("  URL: %s\n", resp3.Data[0].URL)

	// Save to file
	if err := resp3.Data[0].SaveToFile(context.Background(), "portrait_natural.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: portrait_natural.png")
	}

	// Example 4: Multiple images with DALL-E 2 (supports N > 1)
	fmt.Println("\n\nExample 4: Multiple images with DALL-E 2")
	fmt.Println("=========================================")

	n := 4
	resp4, err := client.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:  "openai/dall-e-2",
		Prompt: "Abstract geometric pattern in blue and gold",
		Size:   "512x512",
		N:      &n,
	})
	if err != nil {
		log.Fatalf("Image generation failed: %v", err)
	}

	fmt.Printf("Generated %d images\n", len(resp4.Data))
	for i, img := range resp4.Data {
		fmt.Printf("  Image %d URL: %s\n", i+1, img.URL)

		// Save to file
		filename := fmt.Sprintf("pattern_%d.png", i+1)
		if err := img.SaveToFile(context.Background(), filename); err != nil {
			log.Printf("Failed to save image: %v", err)
		} else {
			fmt.Printf("  Saved to: %s\n", filename)
		}
	}

	// Example 5: Base64-encoded response
	fmt.Println("\n\nExample 5: Base64-encoded response")
	fmt.Println("===================================")

	resp5, err := client.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:          "openai/dall-e-2",
		Prompt:         "A simple icon of a rocket ship",
		Size:           "256x256",
		ResponseFormat: "b64_json",
	})
	if err != nil {
		log.Fatalf("Image generation failed: %v", err)
	}

	fmt.Printf("Generated base64 image\n")
	fmt.Printf("  Base64 length: %d characters\n", len(resp5.Data[0].B64JSON))

	// Save to file
	if err := resp5.Data[0].SaveToFile(context.Background(), "rocket_icon.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: rocket_icon.png")
	}

	fmt.Println("\n\nAll examples completed successfully!")
	fmt.Println("Generated images have been saved to the current directory.")
}
