// Package anthropic implements the Anthropic provider for Warp.
//
// This package provides a complete implementation of the provider.Provider interface
// for Anthropic's Claude API, supporting chat completions and streaming.
//
// Basic usage:
//
//	provider, err := anthropic.NewProvider(
//	    anthropic.WithAPIKey("sk-ant-..."),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "claude-3-opus-20240229",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Anthropic Claude.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	apiKey     string
	apiBase    string
	apiVersion string
	httpClient warp.HTTPClient
}

// Option is a functional option for configuring the Anthropic provider.
type Option func(*Provider)

// NewProvider creates a new Anthropic provider with the given options.
//
// The provider requires an API key to be set via WithAPIKey option.
// Other options are optional and have sensible defaults.
//
// Example:
//
//	provider, err := anthropic.NewProvider(
//	    anthropic.WithAPIKey("sk-ant-..."),
//	    anthropic.WithAPIBase("https://api.anthropic.com"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiBase:    "https://api.anthropic.com",
		apiVersion: "2023-06-01",
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "Anthropic API key is required",
			Provider: "anthropic",
		}
	}

	return p, nil
}

// WithAPIKey sets the Anthropic API key.
//
// This option is required. Without it, NewProvider will return an error.
//
// Example:
//
//	provider, err := anthropic.NewProvider(
//	    anthropic.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAPIBase sets a custom API base URL.
//
// This is useful for using Anthropic-compatible endpoints or proxies.
// The default is "https://api.anthropic.com".
//
// Example:
//
//	provider, err := anthropic.NewProvider(
//	    anthropic.WithAPIKey("sk-ant-..."),
//	    anthropic.WithAPIBase("https://my-proxy.example.com"),
//	)
func WithAPIBase(base string) Option {
	return func(p *Provider) {
		p.apiBase = base
	}
}

// WithAPIVersion sets the Anthropic API version.
//
// The default is "2023-06-01".
//
// Example:
//
//	provider, err := anthropic.NewProvider(
//	    anthropic.WithAPIKey("sk-ant-..."),
//	    anthropic.WithAPIVersion("2023-06-01"),
//	)
func WithAPIVersion(version string) Option {
	return func(p *Provider) {
		p.apiVersion = version
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
//	provider, err := anthropic.NewProvider(
//	    anthropic.WithAPIKey("sk-ant-..."),
//	    anthropic.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "anthropic".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "anthropic"
}

// Supports returns the capabilities supported by Anthropic.
//
// Anthropic supports completion, streaming, function calling, and vision.
// It does not support embeddings, image generation, transcription, speech, or moderation.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       false, // Anthropic doesn't provide embeddings
		ImageGeneration: false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true,
		Vision:          true,
		JSON:            false, // Anthropic doesn't have explicit JSON mode
	}
}

// Embedding returns an error as Anthropic doesn't support embeddings.
//
// This method exists to satisfy the provider.Provider interface.
// Always returns an error indicating embeddings are not supported.
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	return nil, &warp.WarpError{
		Message:  "Anthropic does not support embeddings",
		Provider: "anthropic",
	}
}

// Rerank ranks documents by relevance to a query.
//
// This provider does not support document reranking.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, fmt.Errorf("rerank not supported by anthropic provider")
}
