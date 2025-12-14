// Package main demonstrates text-to-speech functionality using the Warp Go SDK.
//
// This example shows:
// - Basic text-to-speech
// - Saving audio to file
// - Using different voices
// - Using different audio formats
// - Adjusting playback speed
//
// Usage:
//
//	export OPENAI_API_KEY="sk-..."
//	go run main.go
package main

import (
	"context"
	"fmt"
	"io"
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
		log.Fatal(err)
	}

	// Create client
	client, err := warp.NewClient()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Register provider
	if err := client.RegisterProvider(provider); err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Example 1: Basic text-to-speech
	fmt.Println("Example 1: Basic text-to-speech")
	basicTTS(ctx, client)

	// Example 2: Different voices
	fmt.Println("\nExample 2: Different voices")
	differentVoices(ctx, client)

	// Example 3: Different formats
	fmt.Println("\nExample 3: Different audio formats")
	differentFormats(ctx, client)

	// Example 4: Speed control
	fmt.Println("\nExample 4: Speed control")
	speedControl(ctx, client)

	// Example 5: High-definition audio
	fmt.Println("\nExample 5: High-definition audio")
	hdAudio(ctx, client)

	fmt.Println("\nAll examples completed successfully!")
}

// basicTTS demonstrates basic text-to-speech and saving to file.
func basicTTS(ctx context.Context, client warp.Client) {
	audio, err := client.Speech(ctx, &warp.SpeechRequest{
		Model: "openai/tts-1",
		Input: "Hello! Welcome to the Warp Go SDK text-to-speech demonstration.",
		Voice: "alloy",
	})
	if err != nil {
		log.Printf("Error generating speech: %v", err)
		return
	}
	defer audio.Close()

	// Save to file
	out, err := os.Create("basic.mp3")
	if err != nil {
		log.Printf("Error creating file: %v", err)
		return
	}
	defer out.Close()

	written, err := io.Copy(out, audio)
	if err != nil {
		log.Printf("Error writing audio: %v", err)
		return
	}

	fmt.Printf("✓ Generated basic.mp3 (%d bytes)\n", written)
}

// differentVoices demonstrates all available voices.
func differentVoices(ctx context.Context, client warp.Client) {
	voices := []string{"alloy", "echo", "fable", "onyx", "nova", "shimmer"}

	for _, voice := range voices {
		audio, err := client.Speech(ctx, &warp.SpeechRequest{
			Model: "openai/tts-1",
			Input: fmt.Sprintf("This is the %s voice.", voice),
			Voice: voice,
		})
		if err != nil {
			log.Printf("Error generating speech for %s: %v", voice, err)
			continue
		}

		filename := fmt.Sprintf("voice_%s.mp3", voice)
		out, err := os.Create(filename)
		if err != nil {
			audio.Close()
			log.Printf("Error creating file: %v", err)
			continue
		}

		written, err := io.Copy(out, audio)
		audio.Close()
		out.Close()

		if err != nil {
			log.Printf("Error writing audio: %v", err)
			continue
		}

		fmt.Printf("✓ Generated %s (%d bytes)\n", filename, written)
	}
}

// differentFormats demonstrates different audio formats.
func differentFormats(ctx context.Context, client warp.Client) {
	formats := []string{"mp3", "opus", "aac", "flac"}

	for _, format := range formats {
		audio, err := client.Speech(ctx, &warp.SpeechRequest{
			Model:          "openai/tts-1",
			Input:          fmt.Sprintf("This is %s format audio.", format),
			Voice:          "nova",
			ResponseFormat: format,
		})
		if err != nil {
			log.Printf("Error generating %s: %v", format, err)
			continue
		}

		extension := format
		if format == "aac" {
			extension = "m4a" // AAC typically uses .m4a extension
		}

		filename := fmt.Sprintf("format_%s.%s", format, extension)
		out, err := os.Create(filename)
		if err != nil {
			audio.Close()
			log.Printf("Error creating file: %v", err)
			continue
		}

		written, err := io.Copy(out, audio)
		audio.Close()
		out.Close()

		if err != nil {
			log.Printf("Error writing audio: %v", err)
			continue
		}

		fmt.Printf("✓ Generated %s (%d bytes)\n", filename, written)
	}
}

// speedControl demonstrates playback speed adjustment.
func speedControl(ctx context.Context, client warp.Client) {
	speeds := []float64{0.25, 0.5, 1.0, 1.5, 2.0, 4.0}

	for _, speed := range speeds {
		audio, err := client.Speech(ctx, &warp.SpeechRequest{
			Model: "openai/tts-1",
			Input: fmt.Sprintf("This audio is playing at %.2fx speed.", speed),
			Voice: "echo",
			Speed: &speed,
		})
		if err != nil {
			log.Printf("Error generating speech at %.2fx: %v", speed, err)
			continue
		}

		filename := fmt.Sprintf("speed_%.2fx.mp3", speed)
		out, err := os.Create(filename)
		if err != nil {
			audio.Close()
			log.Printf("Error creating file: %v", err)
			continue
		}

		written, err := io.Copy(out, audio)
		audio.Close()
		out.Close()

		if err != nil {
			log.Printf("Error writing audio: %v", err)
			continue
		}

		fmt.Printf("✓ Generated %s (%d bytes)\n", filename, written)
	}
}

// hdAudio demonstrates high-definition audio quality.
func hdAudio(ctx context.Context, client warp.Client) {
	audio, err := client.Speech(ctx, &warp.SpeechRequest{
		Model: "openai/tts-1-hd",
		Input: "This is high-definition audio with superior quality and clarity.",
		Voice: "shimmer",
	})
	if err != nil {
		log.Printf("Error generating HD audio: %v", err)
		return
	}
	defer audio.Close()

	out, err := os.Create("hd_quality.mp3")
	if err != nil {
		log.Printf("Error creating file: %v", err)
		return
	}
	defer out.Close()

	written, err := io.Copy(out, audio)
	if err != nil {
		log.Printf("Error writing audio: %v", err)
		return
	}

	fmt.Printf("✓ Generated hd_quality.mp3 (%d bytes)\n", written)
}
