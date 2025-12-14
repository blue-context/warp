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

	// Example 1: Basic image variation with default model
	// This example requires an existing PNG image file
	fmt.Println("Example 1: Basic image variation (default model)")
	fmt.Println("=================================================")

	// Open the image file
	imageFile, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v\nPlease provide an 'original.png' file in the current directory", err)
	}
	defer imageFile.Close()

	// Note: Model defaults to "openai/dall-e-2" if not specified
	resp1, err := client.ImageVariation(context.Background(), &warp.ImageVariationRequest{
		Image:         imageFile,
		ImageFilename: "original.png",
	})
	if err != nil {
		log.Fatalf("Image variation failed: %v", err)
	}

	fmt.Printf("Generated %d variation(s)\n", len(resp1.Data))
	fmt.Printf("  URL: %s\n", resp1.Data[0].URL)

	// Save to file
	if err := resp1.Data[0].SaveToFile(context.Background(), "variation_1.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: variation_1.png")
	}

	// Example 2: Multiple variations with custom size
	fmt.Println("\n\nExample 2: Multiple variations with custom size")
	fmt.Println("================================================")

	// Reopen the image file (since it was consumed in Example 1)
	imageFile2, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v", err)
	}
	defer imageFile2.Close()

	n := 3
	resp2, err := client.ImageVariation(context.Background(), &warp.ImageVariationRequest{
		Model:         "openai/dall-e-2",
		Image:         imageFile2,
		ImageFilename: "original.png",
		N:             &n,
		Size:          "512x512",
	})
	if err != nil {
		log.Fatalf("Image variation failed: %v", err)
	}

	fmt.Printf("Generated %d variations\n", len(resp2.Data))
	for i, img := range resp2.Data {
		fmt.Printf("  Variation %d URL: %s\n", i+1, img.URL)

		// Save to file
		filename := fmt.Sprintf("variation_2_%d.png", i+1)
		if err := img.SaveToFile(context.Background(), filename); err != nil {
			log.Printf("Failed to save image: %v", err)
		} else {
			fmt.Printf("  Saved to: %s\n", filename)
		}
	}

	// Example 3: High-resolution variations
	fmt.Println("\n\nExample 3: High-resolution variations")
	fmt.Println("======================================")

	// Reopen the image file
	imageFile3, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v", err)
	}
	defer imageFile3.Close()

	resp3, err := client.ImageVariation(context.Background(), &warp.ImageVariationRequest{
		Model:         "openai/dall-e-2",
		Image:         imageFile3,
		ImageFilename: "original.png",
		Size:          "1024x1024",
	})
	if err != nil {
		log.Fatalf("Image variation failed: %v", err)
	}

	fmt.Printf("Generated high-resolution variation\n")
	fmt.Printf("  Size: 1024x1024\n")
	fmt.Printf("  URL: %s\n", resp3.Data[0].URL)

	// Save to file
	if err := resp3.Data[0].SaveToFile(context.Background(), "variation_hires.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: variation_hires.png")
	}

	// Example 4: Base64-encoded response
	fmt.Println("\n\nExample 4: Base64-encoded response")
	fmt.Println("===================================")

	// Reopen the image file
	imageFile4, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v", err)
	}
	defer imageFile4.Close()

	resp4, err := client.ImageVariation(context.Background(), &warp.ImageVariationRequest{
		Model:          "openai/dall-e-2",
		Image:          imageFile4,
		ImageFilename:  "original.png",
		Size:           "256x256",
		ResponseFormat: "b64_json",
	})
	if err != nil {
		log.Fatalf("Image variation failed: %v", err)
	}

	fmt.Printf("Generated base64-encoded variation\n")
	fmt.Printf("  Base64 length: %d characters\n", len(resp4.Data[0].B64JSON))

	// Save to file
	if err := resp4.Data[0].SaveToFile(context.Background(), "variation_base64.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: variation_base64.png")
	}

	fmt.Println("\n\nAll examples completed successfully!")
	fmt.Println("Image variations have been saved to the current directory.")
	fmt.Println("\nNote: Image variations work best with:")
	fmt.Println("  - PNG images with clear subjects")
	fmt.Println("  - Images less than 4MB in size")
	fmt.Println("  - Square images (or they will be center-cropped)")
	fmt.Println("  - DALL-E 2 model (DALL-E 3 does not support variations)")
}
