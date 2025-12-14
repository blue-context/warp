// Package together implements the Together AI provider for Warp.
//
// Together AI provides access to hundreds of open source models including
// Meta-Llama, Mistral, Qwen, and more through an OpenAI-compatible API.
//
// Supported models: meta-llama/Llama-3-70b-chat-hf, mistralai/Mixtral-8x7B-Instruct-v0.1,
// Qwen/Qwen2-72B-Instruct, and many others.
//
// Basic usage:
//
//	provider, err := together.NewProvider(
//	    together.WithAPIKey("..."),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "meta-llama/Llama-3-70b-chat-hf",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package together

import (
	"context"
	"io"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Together AI.
//
// Together AI uses an OpenAI-compatible API, making integration straightforward.
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

// Option is a functional option for configuring the Together AI provider.
type Option func(*Provider)

// NewProvider creates a new Together AI provider with the given options.
//
// The provider requires an API key to be set via WithAPIKey option.
// Other options are optional and have sensible defaults.
//
// Example:
//
//	provider, err := together.NewProvider(
//	    together.WithAPIKey("..."),
//	    together.WithAPIBase("https://api.together.xyz/v1"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiBase:    "https://api.together.xyz/v1",
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "Together AI API key is required",
			Provider: "together",
		}
	}

	return p, nil
}

// WithAPIKey sets the Together AI API key.
//
// This option is required. Without it, NewProvider will return an error.
//
// Example:
//
//	provider, err := together.NewProvider(
//	    together.WithAPIKey(os.Getenv("TOGETHER_API_KEY")),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAPIBase sets a custom API base URL.
//
// This is useful for using proxies or alternative endpoints.
// The default is "https://api.together.xyz/v1".
//
// Example:
//
//	provider, err := together.NewProvider(
//	    together.WithAPIKey("..."),
//	    together.WithAPIBase("https://my-proxy.example.com/v1"),
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
//	provider, err := together.NewProvider(
//	    together.WithAPIKey("..."),
//	    together.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "together".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "together"
}

// Supports returns the capabilities supported by Together AI.
//
// Together AI supports completion, streaming, function calling, and JSON mode.
// Together AI does not currently support embeddings, image generation, or other features.
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
		Message:  "rerank is not supported by Together",
		Provider: "together",
	}
}

// Moderation checks content for policy violations.
//
// Together does not support content moderation.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "moderation is not supported by Together",
		Provider: "together",
	}
}

// Transcription transcribes audio to text.
//
// Together does not support audio transcription.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by Together",
		Provider: "together",
	}
}

// Speech converts text to speech.
//
// Together does not support text-to-speech.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by Together",
		Provider: "together",
	}
}

// ImageGeneration generates images from text prompts.
//
// Together does not support image generation.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image generation is not supported by Together",
		Provider: "together",
	}
}

// ImageEdit edits an image using AI based on a text prompt.
//
// Together does not support image editing.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by Together",
		Provider: "together",
	}
}

// ImageVariation creates variations of an existing image.
//
// Together does not support image variation.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by Together",
		Provider: "together",
	}
}
