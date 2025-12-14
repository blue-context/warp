// Package cohere implements the Cohere provider for Warp.
//
// Cohere provides command models optimized for conversational AI.
// The API has a different format from OpenAI, requiring transformation.
//
// Supported models: command-r, command-r-plus, command-light
//
// Basic usage:
//
//	provider, err := cohere.NewProvider(
//	    cohere.WithAPIKey("..."),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "command-r",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package cohere

import (
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Cohere.
//
// Cohere uses a different API format from OpenAI, requiring request/response transformation.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	apiKey     string
	apiBase    string
	httpClient warp.HTTPClient
}

// Option is a functional option for configuring the Cohere provider.
type Option func(*Provider)

// NewProvider creates a new Cohere provider with the given options.
//
// The provider requires an API key to be set via WithAPIKey option.
// Other options are optional and have sensible defaults.
//
// Example:
//
//	provider, err := cohere.NewProvider(
//	    cohere.WithAPIKey("..."),
//	    cohere.WithAPIBase("https://api.cohere.ai/v1"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiBase:    "https://api.cohere.ai/v1",
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "Cohere API key is required",
			Provider: "cohere",
		}
	}

	return p, nil
}

// WithAPIKey sets the Cohere API key.
//
// This option is required. Without it, NewProvider will return an error.
//
// Example:
//
//	provider, err := cohere.NewProvider(
//	    cohere.WithAPIKey(os.Getenv("COHERE_API_KEY")),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithAPIBase sets a custom API base URL.
//
// This is useful for using proxies or alternative endpoints.
// The default is "https://api.cohere.ai/v1".
//
// Example:
//
//	provider, err := cohere.NewProvider(
//	    cohere.WithAPIKey("..."),
//	    cohere.WithAPIBase("https://my-proxy.example.com/v1"),
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
//	provider, err := cohere.NewProvider(
//	    cohere.WithAPIKey("..."),
//	    cohere.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "cohere".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "cohere"
}

// Supports returns the capabilities supported by Cohere.
//
// Cohere supports completion and reranking.
// Streaming is not currently implemented due to Cohere's different streaming format.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       false,
		Embedding:       false,
		ImageGeneration: false,
		ImageEdit:       false,
		ImageVariation:  false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: false, // Cohere has tools but different format
		Vision:          false,
		JSON:            false,
		Rerank:          true,
	}
}
