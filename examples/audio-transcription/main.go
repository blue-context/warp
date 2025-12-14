// Package main demonstrates audio transcription using the Warp Go SDK.
//
// This example shows how to:
//   - Transcribe audio files using OpenAI's Whisper model
//   - Use different response formats (json, text, srt, vtt, verbose_json)
//   - Add timestamp granularities (word-level and segment-level)
//   - Provide context hints with prompts
//   - Handle language specification
//
// Prerequisites:
//   - OpenAI API key set in OPENAI_API_KEY environment variable
//   - Audio file (MP3, WAV, M4A, etc.) to transcribe
//
// Usage:
//
//	export OPENAI_API_KEY=your-api-key
//	go run main.go path/to/audio.mp3
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
	// Check for audio file argument
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <audio-file>")
		fmt.Println("Example: go run main.go audio.mp3")
		os.Exit(1)
	}

	audioPath := os.Args[1]

	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Create client
	client, err := warp.NewClient()
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Create and register OpenAI provider
	provider, err := openai.NewProvider(
		openai.WithAPIKey(apiKey),
	)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	if err := client.RegisterProvider(provider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	ctx := context.Background()

	// Example 1: Basic transcription with JSON response
	fmt.Println("=== Example 1: Basic Transcription (JSON) ===")
	basicTranscription(ctx, client, audioPath)

	// Example 2: Transcription with language hint
	fmt.Println("\n=== Example 2: Transcription with Language Hint ===")
	transcriptionWithLanguage(ctx, client, audioPath)

	// Example 3: Transcription with verbose JSON (includes timestamps)
	fmt.Println("\n=== Example 3: Verbose JSON with Timestamps ===")
	verboseTranscription(ctx, client, audioPath)

	// Example 4: Plain text transcription
	fmt.Println("\n=== Example 4: Plain Text Response ===")
	plainTextTranscription(ctx, client, audioPath)

	// Example 5: SRT subtitle format
	fmt.Println("\n=== Example 5: SRT Subtitle Format ===")
	srtTranscription(ctx, client, audioPath)
}

// basicTranscription demonstrates a simple transcription with JSON response
func basicTranscription(ctx context.Context, client warp.Client, audioPath string) {
	// Open audio file
	file, err := os.Open(audioPath)
	if err != nil {
		log.Printf("Failed to open audio file: %v", err)
		return
	}
	defer file.Close()

	// Transcribe audio
	resp, err := client.Transcription(ctx, &warp.TranscriptionRequest{
		Model:          "openai/whisper-1",
		File:           file,
		Filename:       audioPath,
		ResponseFormat: "json",
	})
	if err != nil {
		log.Printf("Transcription failed: %v", err)
		return
	}

	fmt.Printf("Text: %s\n", resp.Text)
	if resp.Language != "" {
		fmt.Printf("Language: %s\n", resp.Language)
	}
}

// transcriptionWithLanguage demonstrates providing a language hint
func transcriptionWithLanguage(ctx context.Context, client warp.Client, audioPath string) {
	file, err := os.Open(audioPath)
	if err != nil {
		log.Printf("Failed to open audio file: %v", err)
		return
	}
	defer file.Close()

	// Specify language (ISO 639-1 code) for improved accuracy
	resp, err := client.Transcription(ctx, &warp.TranscriptionRequest{
		Model:          "openai/whisper-1",
		File:           file,
		Filename:       audioPath,
		Language:       "en", // English
		ResponseFormat: "json",
		Prompt:         "This is a conversation about AI and machine learning.", // Context hint
	})
	if err != nil {
		log.Printf("Transcription failed: %v", err)
		return
	}

	fmt.Printf("Text: %s\n", resp.Text)
	fmt.Printf("Language: %s\n", resp.Language)
}

// verboseTranscription demonstrates getting detailed segment and word-level timestamps
func verboseTranscription(ctx context.Context, client warp.Client, audioPath string) {
	file, err := os.Open(audioPath)
	if err != nil {
		log.Printf("Failed to open audio file: %v", err)
		return
	}
	defer file.Close()

	// Request verbose JSON with word and segment timestamps
	resp, err := client.Transcription(ctx, &warp.TranscriptionRequest{
		Model:                  "openai/whisper-1",
		File:                   file,
		Filename:               audioPath,
		ResponseFormat:         "verbose_json",
		TimestampGranularities: []string{"word", "segment"},
	})
	if err != nil {
		log.Printf("Transcription failed: %v", err)
		return
	}

	fmt.Printf("Text: %s\n", resp.Text)
	fmt.Printf("Duration: %.2f seconds\n", resp.Duration)
	fmt.Printf("Language: %s\n\n", resp.Language)

	// Print word-level timestamps
	if len(resp.Words) > 0 {
		fmt.Println("Word-level timestamps (first 10 words):")
		maxWords := 10
		if len(resp.Words) < maxWords {
			maxWords = len(resp.Words)
		}
		for i := 0; i < maxWords; i++ {
			word := resp.Words[i]
			fmt.Printf("  [%.2fs-%.2fs] %s\n", word.Start, word.End, word.Word)
		}
		if len(resp.Words) > 10 {
			fmt.Printf("  ... and %d more words\n", len(resp.Words)-10)
		}
		fmt.Println()
	}

	// Print segment information
	if len(resp.Segments) > 0 {
		fmt.Println("Segments:")
		for _, seg := range resp.Segments {
			fmt.Printf("  [%.2fs-%.2fs] %s\n", seg.Start, seg.End, seg.Text)
		}
	}
}

// plainTextTranscription demonstrates getting plain text output
func plainTextTranscription(ctx context.Context, client warp.Client, audioPath string) {
	file, err := os.Open(audioPath)
	if err != nil {
		log.Printf("Failed to open audio file: %v", err)
		return
	}
	defer file.Close()

	// Request plain text response
	resp, err := client.Transcription(ctx, &warp.TranscriptionRequest{
		Model:          "openai/whisper-1",
		File:           file,
		Filename:       audioPath,
		ResponseFormat: "text",
	})
	if err != nil {
		log.Printf("Transcription failed: %v", err)
		return
	}

	fmt.Println(resp.Text)
}

// srtTranscription demonstrates getting SRT subtitle format
func srtTranscription(ctx context.Context, client warp.Client, audioPath string) {
	file, err := os.Open(audioPath)
	if err != nil {
		log.Printf("Failed to open audio file: %v", err)
		return
	}
	defer file.Close()

	// Request SRT subtitle format
	resp, err := client.Transcription(ctx, &warp.TranscriptionRequest{
		Model:          "openai/whisper-1",
		File:           file,
		Filename:       audioPath,
		ResponseFormat: "srt",
	})
	if err != nil {
		log.Printf("Transcription failed: %v", err)
		return
	}

	// Print SRT content (limited to first 500 characters)
	maxLen := 500
	if len(resp.Text) < maxLen {
		maxLen = len(resp.Text)
	}
	fmt.Println(resp.Text[:maxLen])
	if len(resp.Text) > 500 {
		fmt.Println("... (truncated)")
	}
}
