// Package openai implements the OpenAI provider for Warp.
//
// This package provides a complete implementation of the provider.Provider interface
// for OpenAI's API, supporting chat completions, streaming, and embeddings.
//
// Basic usage:
//
//	provider, err := openai.NewProvider(
//	    openai.WithAPIKey("sk-..."),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "gpt-4",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package openai

import (
	"context"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for OpenAI.
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

// Option is a functional option for configuring the OpenAI provider.
type Option func(*Provider)

// NewProvider creates a new OpenAI provider with the given options.
//
// The provider requires an API key to be set via WithAPIKey option.
// Other options are optional and have sensible defaults.
//
// Example:
//
//	provider, err := openai.NewProvider(
//	    openai.WithAPIKey("sk-..."),
//	    openai.WithAPIBase("https://api.openai.com/v1"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiBase:    "https://api.openai.com/v1",
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "OpenAI API key is required",
			Provider: "openai",
		}
	}

	return p, nil
}

// WithAPIKey sets the OpenAI API key.
//
// This option is required. Without it, NewProvider will return an error.
//
// Example:
//
//	provider, err := openai.NewProvider(
//	    openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAPIBase sets a custom API base URL.
//
// This is useful for using OpenAI-compatible endpoints or proxies.
// The default is "https://api.openai.com/v1".
//
// Example:
//
//	provider, err := openai.NewProvider(
//	    openai.WithAPIKey("sk-..."),
//	    openai.WithAPIBase("https://my-proxy.example.com/v1"),
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
//	provider, err := openai.NewProvider(
//	    openai.WithAPIKey("sk-..."),
//	    openai.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "openai".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "openai"
}

// Supports returns the capabilities supported by OpenAI.
//
// OpenAI supports all major features including completion, streaming,
// embeddings, function calling, vision, and JSON mode.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: true,
		ImageEdit:       true,
		ImageVariation:  true,
		Transcription:   true,
		Speech:          true,
		Moderation:      true,
		FunctionCalling: true,
		Vision:          true,
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
		Message:  "rerank is not supported by OpenAI",
		Provider: "openai",
	}
}
