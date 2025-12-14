// Package openrouter implements the OpenRouter provider for Warp.
//
// OpenRouter is a unified LLM API gateway providing access to 300+ models from
// multiple providers (OpenAI, Anthropic, Google, Meta, Mistral, etc.) through
// a single OpenAI-compatible API.
//
// Key Features:
//   - OpenAI-compatible API (drop-in replacement)
//   - Access to 300+ models from multiple providers
//   - Automatic failover and provider routing
//   - Cost optimization across providers
//   - Support for chat, streaming, embeddings, vision, and function calling
//
// Basic usage:
//
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey("sk-or-v1-..."),
//	    openrouter.WithHTTPReferer("https://myapp.com"),
//	    openrouter.WithAppTitle("My App"),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "openai/gpt-4o",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package openrouter

import (
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for OpenRouter.
//
// OpenRouter provides unified access to 300+ models from multiple providers
// through an OpenAI-compatible API with automatic failover and cost optimization.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	apiKey     string
	apiBase    string
	httpClient warp.HTTPClient

	// Optional custom headers for rankings and analytics
	httpReferer string // HTTP-Referer header for site identification
	appTitle    string // X-Title header for app name
}

// Option is a functional option for configuring the OpenRouter provider.
type Option func(*Provider)

// NewProvider creates a new OpenRouter provider with the given options.
//
// The provider requires an API key to be set via WithAPIKey option.
// Optional headers (HTTP-Referer, X-Title) are recommended for rankings
// on openrouter.ai and better analytics.
//
// Example:
//
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey("sk-or-v1-..."),
//	    openrouter.WithHTTPReferer("https://myapp.com"),
//	    openrouter.WithAppTitle("My App"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiBase:    "https://openrouter.ai/api/v1",
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "OpenRouter API key is required",
			Provider: "openrouter",
		}
	}

	return p, nil
}

// WithAPIKey sets the OpenRouter API key.
//
// This option is required. Without it, NewProvider will return an error.
// Get your API key from https://openrouter.ai/keys
//
// Example:
//
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAPIBase sets a custom API base URL.
//
// This is useful for testing or using alternative OpenRouter endpoints.
// The default is "https://openrouter.ai/api/v1".
//
// Example:
//
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey("sk-or-v1-..."),
//	    openrouter.WithAPIBase("https://test.openrouter.ai/api/v1"),
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
//	    Timeout: 180 * time.Second,
//	    Transport: customTransport,
//	}
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey("sk-or-v1-..."),
//	    openrouter.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// WithHTTPReferer sets the HTTP-Referer header.
//
// Optional but recommended for rankings on openrouter.ai.
// This helps identify your application and improves visibility in the ecosystem.
//
// Example:
//
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey("sk-or-v1-..."),
//	    openrouter.WithHTTPReferer("https://myapp.com"),
//	)
func WithHTTPReferer(referer string) Option {
	return func(p *Provider) {
		p.httpReferer = referer
	}
}

// WithAppTitle sets the X-Title header.
//
// Optional but recommended for app identification on openrouter.ai.
// This sets a human-readable name for your application in analytics.
//
// Example:
//
//	provider, err := openrouter.NewProvider(
//	    openrouter.WithAPIKey("sk-or-v1-..."),
//	    openrouter.WithAppTitle("My AI Assistant"),
//	)
func WithAppTitle(title string) Option {
	return func(p *Provider) {
		p.appTitle = title
	}
}

// Name returns the provider name "openrouter".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "openrouter"
}

// Supports returns the capabilities supported by OpenRouter.
//
// OpenRouter supports all major features including completion, streaming,
// embeddings, image generation, function calling, vision, and JSON mode.
// Capabilities may vary by specific model.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: true,
		FunctionCalling: true,
		Vision:          true,
		JSON:            true,
	}
}
