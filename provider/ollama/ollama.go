// Package ollama implements the Ollama provider for Warp.
//
// Ollama runs large language models locally. It has no authentication
// and runs on localhost by default (http://localhost:11434).
//
// Supported models: Any model pulled locally (llama3, mistral, codellama, etc.)
//
// Basic usage:
//
//	provider, err := ollama.NewProvider(
//	    ollama.WithBaseURL("http://localhost:11434"),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "llama3",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package ollama

import (
	"context"
	"io"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Ollama.
//
// Ollama is a local LLM server that doesn't require authentication.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	baseURL    string
	httpClient warp.HTTPClient
}

// Compile-time interface check
var _ provider.Provider = (*Provider)(nil)

// Option is a functional option for configuring the Ollama provider.
type Option func(*Provider)

// NewProvider creates a new Ollama provider with the given options.
//
// No API key is required since Ollama runs locally.
// The default base URL is http://localhost:11434.
//
// Example:
//
//	provider, err := ollama.NewProvider(
//	    ollama.WithBaseURL("http://localhost:11434"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		baseURL:    "http://localhost:11434",
		httpClient: &http.Client{Timeout: 120 * time.Second}, // Longer timeout for local generation
	}

	for _, opt := range opts {
		opt(p)
	}

	return p, nil
}

// WithBaseURL sets a custom base URL.
//
// This is useful if Ollama is running on a different host or port.
// The default is "http://localhost:11434".
//
// Example:
//
//	provider, err := ollama.NewProvider(
//	    ollama.WithBaseURL("http://192.168.1.100:11434"),
//	)
func WithBaseURL(url string) Option {
	return func(p *Provider) {
		p.baseURL = url
	}
}

// WithHTTPClient sets a custom HTTP client.
//
// This is useful for configuring custom timeouts, transport settings,
// or injecting mock clients for testing.
//
// Example:
//
//	customClient := &http.Client{
//	    Timeout: 180 * time.Second,
//	}
//	provider, err := ollama.NewProvider(
//	    ollama.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "ollama".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "ollama"
}

// Supports returns the capabilities supported by Ollama.
//
// Ollama supports completion and streaming.
// Ollama does not support embeddings, function calling, or other advanced features.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       false,
		ImageGeneration: false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: false,
		Vision:          false,
		JSON:            false,
	}
}

// Rerank ranks documents by relevance to a query.
//
// This provider does not support document reranking.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, &warp.WarpError{
		Message:  "rerank is not supported by Ollama",
		Provider: "ollama",
	}
}

// Moderation checks content for policy violations.
//
// Ollama does not support content moderation.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "moderation is not supported by Ollama",
		Provider: "ollama",
	}
}

// Transcription transcribes audio to text.
//
// Ollama does not support audio transcription.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by Ollama",
		Provider: "ollama",
	}
}

// Speech converts text to speech.
//
// Ollama does not support text-to-speech.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by Ollama",
		Provider: "ollama",
	}
}

// ImageGeneration generates images from text prompts.
//
// Ollama does not support image generation.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image generation is not supported by Ollama",
		Provider: "ollama",
	}
}

// ImageEdit edits an image using AI based on a text prompt.
//
// Ollama does not support image editing.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by Ollama",
		Provider: "ollama",
	}
}

// ImageVariation creates variations of an existing image.
//
// Ollama does not support image variation.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by Ollama",
		Provider: "ollama",
	}
}
