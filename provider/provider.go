// Package provider defines the interface that all LLM providers must implement
// and provides utilities for managing provider capabilities.
//
// Each LLM provider (OpenAI, Anthropic, Azure, etc.) implements the Provider
// interface to provide a unified API for calling different LLM services.
package provider

import (
	"context"
	"io"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/types"
)

// ModelInfo is an alias for types.ModelInfo to maintain backward compatibility.
// Use types.ModelInfo directly in new code.
type ModelInfo = types.ModelInfo

// Capabilities is an alias for types.Capabilities to maintain backward compatibility.
// Use types.Capabilities directly in new code.
type Capabilities = types.Capabilities

// Provider defines the interface that all LLM providers must implement.
//
// Each provider (OpenAI, Anthropic, Azure, etc.) implements this interface
// to provide a unified API for calling different LLM services.
//
// Thread Safety: Implementations must be safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
//
// Example:
//
//	type OpenAIProvider struct {
//	    apiKey string
//	}
//
//	func (p *OpenAIProvider) Name() string {
//	    return "openai"
//	}
//
//	func (p *OpenAIProvider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
//	    // Implementation...
//	}
type Provider interface {
	// Name returns the provider name (e.g., "openai", "anthropic", "azure").
	//
	// The name should be lowercase and unique across all providers.
	// It is used as the provider identifier in the registry.
	Name() string

	// Completion sends a chat completion request.
	//
	// Returns the complete response from the LLM provider.
	//
	// Example:
	//   resp, err := provider.Completion(ctx, &warp.CompletionRequest{
	//       Model: "gpt-4",
	//       Messages: []warp.Message{
	//           {Role: "user", Content: "Hello!"},
	//       },
	//   })
	Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error)

	// CompletionStream sends a streaming chat completion request.
	//
	// Returns a Stream that must be closed by the caller.
	// The stream delivers response chunks incrementally as they become available.
	//
	// Example:
	//   stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
	//       Model: "gpt-4",
	//       Messages: []warp.Message{
	//           {Role: "user", Content: "Hello!"},
	//       },
	//   })
	//   if err != nil {
	//       return err
	//   }
	//   defer stream.Close()
	//
	//   for {
	//       chunk, err := stream.Recv()
	//       if err == io.EOF {
	//           break
	//       }
	//       if err != nil {
	//           return err
	//       }
	//       fmt.Print(chunk.Choices[0].Delta.Content)
	//   }
	CompletionStream(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error)

	// Embedding sends an embedding request.
	//
	// Returns an error if the provider doesn't support embeddings.
	// Check Supports().Embedding before calling.
	//
	// Example:
	//   resp, err := provider.Embedding(ctx, &warp.EmbeddingRequest{
	//       Model: "text-embedding-ada-002",
	//       Input: "Hello, world!",
	//   })
	Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error)

	// ImageGeneration generates images from text prompts.
	//
	// Returns an error if the provider doesn't support image generation.
	// Check Supports().ImageGeneration before calling.
	//
	// Example:
	//   resp, err := provider.ImageGeneration(ctx, &warp.ImageGenerationRequest{
	//       Model: "dall-e-3",
	//       Prompt: "A cute baby sea otter",
	//       Size: "1024x1024",
	//   })
	ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error)

	// ImageEdit edits an image using AI based on a text prompt.
	//
	// Returns an error if the provider doesn't support image editing.
	// Check Supports().ImageEdit before calling.
	//
	// Example:
	//   imageFile, _ := os.Open("original.png")
	//   defer imageFile.Close()
	//   resp, err := provider.ImageEdit(ctx, &warp.ImageEditRequest{
	//       Model: "dall-e-2",
	//       Image: imageFile,
	//       ImageFilename: "original.png",
	//       Prompt: "Add a party hat to the cat",
	//   })
	ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error)

	// ImageVariation creates variations of an existing image.
	//
	// Returns an error if the provider doesn't support image variation.
	// Check Supports().ImageVariation before calling.
	//
	// Example:
	//   imageFile, _ := os.Open("original.png")
	//   defer imageFile.Close()
	//   resp, err := provider.ImageVariation(ctx, &warp.ImageVariationRequest{
	//       Model: "dall-e-2",
	//       Image: imageFile,
	//       ImageFilename: "original.png",
	//       N: warp.IntPtr(3),
	//       Size: "512x512",
	//   })
	ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error)

	// Transcription transcribes audio to text.
	//
	// Returns an error if the provider doesn't support transcription.
	// Check Supports().Transcription before calling.
	//
	// Example:
	//   f, _ := os.Open("audio.mp3")
	//   defer f.Close()
	//   resp, err := provider.Transcription(ctx, &warp.TranscriptionRequest{
	//       Model: "whisper-1",
	//       File: f,
	//       Filename: "audio.mp3",
	//   })
	Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error)

	// Speech converts text to speech.
	//
	// Returns an io.ReadCloser containing audio data.
	// The caller MUST close the reader when done.
	//
	// Returns an error if the provider doesn't support text-to-speech.
	// Check Supports().Speech before calling.
	//
	// Example:
	//   audio, err := provider.Speech(ctx, &warp.SpeechRequest{
	//       Model: "tts-1",
	//       Input: "Hello, world!",
	//       Voice: "alloy",
	//   })
	//   if err != nil {
	//       return err
	//   }
	//   defer audio.Close()
	//
	//   // Write to file
	//   f, _ := os.Create("output.mp3")
	//   defer f.Close()
	//   io.Copy(f, audio)
	Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error)

	// Moderation checks content for policy violations.
	//
	// Returns moderation results indicating whether content is flagged
	// for various content policy categories.
	//
	// Returns an error if the provider doesn't support moderation.
	// Check Supports().Moderation before calling.
	//
	// Example:
	//   resp, err := provider.Moderation(ctx, &warp.ModerationRequest{
	//       Model: "text-moderation-latest",
	//       Input: "I want to hurt someone",
	//   })
	//   if err != nil {
	//       return err
	//   }
	//   if resp.Results[0].Flagged {
	//       fmt.Println("Content flagged")
	//   }
	Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error)

	// Rerank ranks documents by relevance to a query.
	//
	// Used for RAG (Retrieval-Augmented Generation) applications to rank
	// retrieved documents before feeding them to an LLM.
	//
	// Returns an error if the provider doesn't support reranking.
	// Check Supports().Rerank before calling.
	//
	// Example:
	//   resp, err := provider.Rerank(ctx, &warp.RerankRequest{
	//       Model: "rerank-english-v3.0",
	//       Query: "What is the capital of France?",
	//       Documents: []string{
	//           "Paris is the capital of France",
	//           "London is the capital of England",
	//       },
	//       TopN: warp.IntPtr(1),
	//   })
	//   if err != nil {
	//       return err
	//   }
	//   for _, result := range resp.Results {
	//       fmt.Printf("Document %d: score=%.3f\n", result.Index, result.RelevanceScore)
	//   }
	Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error)

	// Supports returns the capabilities supported by this provider.
	//
	// Use this to check which features the provider supports before calling them.
	//
	// Example:
	//   caps := provider.Supports().(Capabilities)
	//   if caps.Streaming {
	//       // Use streaming
	//   }
	Supports() interface{}

	// GetModelInfo returns metadata for a specific model.
	//
	// Returns nil if the model is unknown to this provider.
	// This method is used by the cost calculator to retrieve pricing and capability information.
	//
	// Thread Safety: Must be safe for concurrent use.
	//
	// Example:
	//   info := provider.GetModelInfo("gpt-4")
	//   if info != nil {
	//       fmt.Printf("Input cost: $%.2f per 1M tokens\n", info.InputCostPer1M)
	//   }
	GetModelInfo(model string) *ModelInfo

	// ListModels returns all models supported by this provider.
	//
	// Returns a slice of all model names that this provider can handle.
	// The slice should be sorted alphabetically for consistent output.
	//
	// Thread Safety: Must be safe for concurrent use.
	//
	// Example:
	//   models := provider.ListModels()
	//   for _, model := range models {
	//       fmt.Println(model)
	//   }
	ListModels() []*ModelInfo
}

// Note: Stream interface is defined in the warp package to avoid duplication.
// Providers should return warp.Stream from CompletionStream method.

// AllSupported returns Capabilities with all features enabled.
//
// Useful for testing or providers that support everything.
func AllSupported() Capabilities {
	return Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: true,
		Transcription:   true,
		Speech:          true,
		Moderation:      true,
		FunctionCalling: true,
		Vision:          true,
		JSON:            true,
	}
}

// NoneSupported returns Capabilities with all features disabled.
//
// Useful as a base for providers that only support specific features.
func NoneSupported() Capabilities {
	return Capabilities{}
}
