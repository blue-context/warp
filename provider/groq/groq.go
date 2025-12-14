// Package groq implements the Groq provider for Warp.
//
// Groq provides ultra-fast LLM inference using custom LPU hardware.
// The API is OpenAI-compatible, making integration straightforward.
//
// Supported models: llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768, gemma-7b-it
//
// Basic usage:
//
//	provider, err := groq.NewProvider(
//	    groq.WithAPIKey("gsk_..."),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "llama3-70b-8192",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package groq

import (
	"context"
	"io"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Groq.
//
// Groq uses an OpenAI-compatible API, so the implementation is very similar
// to the OpenAI provider, just with different endpoints and models.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	apiKey     string
	apiBase    string
	httpClient warp.HTTPClient
}

// Compile-time interface check
var _ provider.Provider = (*Provider)(nil)

// Option is a functional option for configuring the Groq provider.
type Option func(*Provider)

// NewProvider creates a new Groq provider with the given options.
//
// The provider requires an API key to be set via WithAPIKey option.
// Other options are optional and have sensible defaults.
//
// Example:
//
//	provider, err := groq.NewProvider(
//	    groq.WithAPIKey("gsk_..."),
//	    groq.WithAPIBase("https://api.groq.com/openai/v1"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiBase:    "https://api.groq.com/openai/v1",
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "Groq API key is required",
			Provider: "groq",
		}
	}

	return p, nil
}

// WithAPIKey sets the Groq API key.
//
// This option is required. Without it, NewProvider will return an error.
//
// Example:
//
//	provider, err := groq.NewProvider(
//	    groq.WithAPIKey(os.Getenv("GROQ_API_KEY")),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAPIBase sets a custom API base URL.
//
// This is useful for using proxies or alternative endpoints.
// The default is "https://api.groq.com/openai/v1".
//
// Example:
//
//	provider, err := groq.NewProvider(
//	    groq.WithAPIKey("gsk_..."),
//	    groq.WithAPIBase("https://my-proxy.example.com/v1"),
//	)
func WithAPIBase(base string) Option {
	return func(p *Provider) {
		p.apiBase = base
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
//	    Timeout: 120 * time.Second,
//	    Transport: customTransport,
//	}
//	provider, err := groq.NewProvider(
//	    groq.WithAPIKey("gsk_..."),
//	    groq.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "groq".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "groq"
}

// Supports returns the capabilities supported by Groq.
//
// Groq supports completion, streaming, function calling, and JSON mode.
// Groq does not currently support embeddings, image generation, or other features.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       false,
		ImageGeneration: false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true,
		Vision:          false,
		JSON:            true,
	}
}

// Rerank ranks documents by relevance to a query.
//
// This provider does not support document reranking.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, &warp.WarpError{
		Message:  "rerank is not supported by Groq",
		Provider: "groq",
	}
}

// Moderation checks content for policy violations.
//
// Groq does not support content moderation.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "moderation is not supported by Groq",
		Provider: "groq",
	}
}

// Transcription transcribes audio to text.
//
// Groq does not support audio transcription.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by Groq",
		Provider: "groq",
	}
}

// Speech converts text to speech.
//
// Groq does not support text-to-speech.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by Groq",
		Provider: "groq",
	}
}

// ImageGeneration generates images from text prompts.
//
// Groq does not support image generation.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image generation is not supported by Groq",
		Provider: "groq",
	}
}

// ImageEdit edits an image using AI based on a text prompt.
//
// Groq does not support image editing.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by Groq",
		Provider: "groq",
	}
}

// ImageVariation creates variations of an existing image.
//
// Groq does not support image variation.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by Groq",
		Provider: "groq",
	}
}
