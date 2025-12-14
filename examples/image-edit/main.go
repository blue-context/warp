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

	// Example 1: Basic image edit without mask
	// This example requires an existing PNG image file
	fmt.Println("Example 1: Basic image edit (without mask)")
	fmt.Println("==========================================")

	// Open the image file
	imageFile, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v\nPlease provide an 'original.png' file in the current directory", err)
	}
	defer imageFile.Close()

	resp1, err := client.ImageEdit(context.Background(), &warp.ImageEditRequest{
		Model:         "openai/dall-e-2",
		Image:         imageFile,
		ImageFilename: "original.png",
		Prompt:        "Add a party hat to the subject",
	})
	if err != nil {
		log.Fatalf("Image edit failed: %v", err)
	}

	fmt.Printf("Edited image successfully\n")
	fmt.Printf("  URL: %s\n", resp1.Data[0].URL)

	// Save to file
	if err := resp1.Data[0].SaveToFile(context.Background(), "edited_with_hat.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: edited_with_hat.png")
	}

	// Example 2: Image edit with mask for targeted editing
	// The mask should be a PNG with transparent areas indicating where to edit
	fmt.Println("\n\nExample 2: Targeted edit with mask")
	fmt.Println("===================================")

	// Reopen the image file (since it was consumed in Example 1)
	imageFile2, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v", err)
	}
	defer imageFile2.Close()

	// Open the mask file
	maskFile, err := os.Open("mask.png")
	if err != nil {
		log.Printf("Skipping Example 2: mask.png not found")
		log.Printf("To run this example, create a mask.png file with transparent areas where you want edits\n")
	} else {
		defer maskFile.Close()

		resp2, err := client.ImageEdit(context.Background(), &warp.ImageEditRequest{
			Model:         "openai/dall-e-2",
			Image:         imageFile2,
			ImageFilename: "original.png",
			Mask:          maskFile,
			MaskFilename:  "mask.png",
			Prompt:        "Change the background to a sunny beach",
		})
		if err != nil {
			log.Fatalf("Image edit with mask failed: %v", err)
		}

		fmt.Printf("Edited image with mask successfully\n")
		fmt.Printf("  URL: %s\n", resp2.Data[0].URL)

		// Save to file
		if err := resp2.Data[0].SaveToFile(context.Background(), "edited_with_mask.png"); err != nil {
			log.Printf("Failed to save image: %v", err)
		} else {
			fmt.Println("  Saved to: edited_with_mask.png")
		}
	}

	// Example 3: Multiple edited images with custom size
	fmt.Println("\n\nExample 3: Multiple edited images")
	fmt.Println("==================================")

	// Reopen the image file
	imageFile3, err := os.Open("original.png")
	if err != nil {
		log.Fatalf("Failed to open image file: %v", err)
	}
	defer imageFile3.Close()

	n := 3
	resp3, err := client.ImageEdit(context.Background(), &warp.ImageEditRequest{
		Model:         "openai/dall-e-2",
		Image:         imageFile3,
		ImageFilename: "original.png",
		Prompt:        "Add colorful confetti falling around the subject",
		N:             &n,
		Size:          "512x512",
	})
	if err != nil {
		log.Fatalf("Image edit failed: %v", err)
	}

	fmt.Printf("Generated %d edited images\n", len(resp3.Data))
	for i, img := range resp3.Data {
		fmt.Printf("  Image %d URL: %s\n", i+1, img.URL)

		// Save to file
		filename := fmt.Sprintf("edited_confetti_%d.png", i+1)
		if err := img.SaveToFile(context.Background(), filename); err != nil {
			log.Printf("Failed to save image: %v", err)
		} else {
			fmt.Printf("  Saved to: %s\n", filename)
		}
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

	resp4, err := client.ImageEdit(context.Background(), &warp.ImageEditRequest{
		Model:          "openai/dall-e-2",
		Image:          imageFile4,
		ImageFilename:  "original.png",
		Prompt:         "Add sparkles and glitter effects",
		Size:           "256x256",
		ResponseFormat: "b64_json",
	})
	if err != nil {
		log.Fatalf("Image edit failed: %v", err)
	}

	fmt.Printf("Generated base64-encoded edited image\n")
	fmt.Printf("  Base64 length: %d characters\n", len(resp4.Data[0].B64JSON))

	// Save to file
	if err := resp4.Data[0].SaveToFile(context.Background(), "edited_sparkles.png"); err != nil {
		log.Printf("Failed to save image: %v", err)
	} else {
		fmt.Println("  Saved to: edited_sparkles.png")
	}

	fmt.Println("\n\nAll examples completed successfully!")
	fmt.Println("Edited images have been saved to the current directory.")
	fmt.Println("\nNote: Image editing works best with:")
	fmt.Println("  - PNG images with transparent backgrounds")
	fmt.Println("  - Images less than 4MB in size")
	fmt.Println("  - Clear, specific editing prompts")
	fmt.Println("  - Masks with transparent areas indicating where to edit (for targeted edits)")
}
