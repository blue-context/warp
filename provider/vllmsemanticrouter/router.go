// Package vllmsemanticrouter implements the vLLM Semantic Router provider for Warp.
//
// vLLM Semantic Router is an intelligent routing proxy that semantically routes
// requests to optimal backend models. It provides OpenAI-compatible API endpoints
// along with optional classification APIs for intent, PII, and security detection.
//
// The provider supports a special "auto" model parameter that enables semantic
// routing to automatically select the best backend model for the request.
//
// Basic usage:
//
//	provider, err := vllmsemanticrouter.NewProvider(
//	    vllmsemanticrouter.WithBaseURL("http://localhost:8801"),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Use semantic routing with "auto" model
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "auto",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package vllmsemanticrouter

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for vLLM Semantic Router.
//
// vLLM Semantic Router provides intelligent routing to backend models through
// an OpenAI-compatible API. It supports semantic routing via the "auto" model
// parameter and optional classification endpoints.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	baseURL           string
	classificationURL string
	apiKey            string
	httpClient        warp.HTTPClient
}

// Option is a functional option for configuring the vLLM Semantic Router provider.
type Option func(*Provider)

// NewProvider creates a new vLLM Semantic Router provider with the given options.
//
// The provider defaults to localhost:8801 for the main API and localhost:8080
// for classification endpoints. API key is optional as vLLM Semantic Router
// can be deployed without authentication.
//
// Example:
//
//	provider, err := vllmsemanticrouter.NewProvider(
//	    vllmsemanticrouter.WithBaseURL("http://localhost:8801"),
//	    vllmsemanticrouter.WithAPIKey("optional-key"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		baseURL:           "http://localhost:8801",
		classificationURL: "http://localhost:8080",
		httpClient:        &http.Client{Timeout: 120 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	return p, nil
}

// WithBaseURL sets the base URL for the main API endpoint.
//
// This is the Envoy public entry point for the OpenAI-compatible API.
// The default is "http://localhost:8801".
//
// Example:
//
//	provider, err := vllmsemanticrouter.NewProvider(
//	    vllmsemanticrouter.WithBaseURL("http://router.example.com:8801"),
//	)
func WithBaseURL(url string) Option {
	return func(p *Provider) {
		p.baseURL = url
	}
}

// WithClassificationURL sets the URL for classification endpoints.
//
// Classification endpoints provide intent detection, PII detection,
// and security/jailbreak detection. The default is "http://localhost:8080".
//
// Example:
//
//	provider, err := vllmsemanticrouter.NewProvider(
//	    vllmsemanticrouter.WithClassificationURL("http://classifier.example.com:8080"),
//	)
func WithClassificationURL(url string) Option {
	return func(p *Provider) {
		p.classificationURL = url
	}
}

// WithAPIKey sets an optional API key for authentication.
//
// vLLM Semantic Router can be deployed without authentication,
// so this option is optional.
//
// Example:
//
//	provider, err := vllmsemanticrouter.NewProvider(
//	    vllmsemanticrouter.WithAPIKey("my-secret-key"),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
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
//	provider, err := vllmsemanticrouter.NewProvider(
//	    vllmsemanticrouter.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "vllm-semantic-router".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "vllm-semantic-router"
}

// Supports returns the capabilities supported by vLLM Semantic Router.
//
// vLLM Semantic Router supports completion, streaming, function calling, and JSON mode
// through its OpenAI-compatible API. It does not support embeddings, image generation,
// or other specialized features.
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
	return nil, fmt.Errorf("rerank not supported by vllm-semantic-router provider")
}
