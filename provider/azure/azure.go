// Package azure implements the Azure OpenAI provider for Warp.
//
// This package provides a complete implementation of the provider.Provider interface
// for Azure OpenAI's API, supporting chat completions, streaming, and embeddings.
//
// Azure OpenAI differs from standard OpenAI in URL structure and authentication:
// - URL: {base}/openai/deployments/{deployment}/chat/completions?api-version={version}
// - Auth: api-key header (not Authorization: Bearer)
// - Uses deployment names instead of model names
//
// Basic usage:
//
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey("your-key"),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("gpt-4-deployment"),
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
package azure

import (
	"context"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Azure OpenAI.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	apiKey     string
	apiBase    string
	apiVersion string
	deployment string
	httpClient warp.HTTPClient
}

// Option is a functional option for configuring the Azure OpenAI provider.
type Option func(*Provider)

// NewProvider creates a new Azure OpenAI provider with the given options.
//
// The provider requires an API key, endpoint, and deployment name to be set.
// API version is optional and defaults to "2024-02-15-preview".
//
// Example:
//
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey("your-api-key"),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("gpt-4-deployment"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		apiVersion: "2024-02-15-preview",
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	// Validate required fields
	if p.apiKey == "" {
		return nil, &warp.WarpError{
			Message:  "Azure API key is required",
			Provider: "azure",
		}
	}

	if p.apiBase == "" {
		return nil, &warp.WarpError{
			Message:  "Azure endpoint is required",
			Provider: "azure",
		}
	}

	// Remove trailing slash from apiBase for consistent URL construction
	p.apiBase = strings.TrimSuffix(p.apiBase, "/")

	if p.deployment == "" {
		return nil, &warp.WarpError{
			Message:  "Azure deployment name is required",
			Provider: "azure",
		}
	}

	return p, nil
}

// WithAPIKey sets the Azure API key.
//
// This option is required. Without it, NewProvider will return an error.
//
// Example:
//
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey(os.Getenv("AZURE_OPENAI_API_KEY")),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("gpt-4-deployment"),
//	)
func WithAPIKey(key string) Option {
	return func(p *Provider) {
		p.apiKey = key
	}
}

// WithEndpoint sets the Azure OpenAI endpoint URL.
//
// This option is required. Without it, NewProvider will return an error.
// The endpoint should be your Azure resource URL.
//
// Example:
//
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey("your-key"),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("gpt-4-deployment"),
//	)
func WithEndpoint(endpoint string) Option {
	return func(p *Provider) {
		p.apiBase = endpoint
	}
}

// WithDeployment sets the Azure deployment name.
//
// This option is required. Without it, NewProvider will return an error.
// The deployment name is the name you gave your model deployment in Azure.
//
// Example:
//
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey("your-key"),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("my-gpt-4-deployment"),
//	)
func WithDeployment(deployment string) Option {
	return func(p *Provider) {
		p.deployment = deployment
	}
}

// WithAPIVersion sets the Azure OpenAI API version.
//
// This option is optional. The default is "2024-02-15-preview".
//
// Example:
//
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey("your-key"),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("gpt-4-deployment"),
//	    azure.WithAPIVersion("2024-02-01"),
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
//	provider, err := azure.NewProvider(
//	    azure.WithAPIKey("your-key"),
//	    azure.WithEndpoint("https://your-resource.openai.azure.com"),
//	    azure.WithDeployment("gpt-4-deployment"),
//	    azure.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "azure".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "azure"
}

// Supports returns the capabilities supported by Azure OpenAI.
//
// Azure OpenAI supports the same features as OpenAI including completion,
// streaming, embeddings, function calling, vision, and JSON mode.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: false, // Azure doesn't support DALL-E via this API
		Transcription:   false, // Whisper requires different endpoints
		Speech:          false, // TTS requires different endpoints
		Moderation:      false, // Moderation requires different endpoints
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
		Message:  "rerank is not supported by Azure",
		Provider: "azure",
	}
}

// Moderation checks content for policy violations.
//
// Azure does not support content moderation.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "moderation is not supported by Azure",
		Provider: "azure",
	}
}

// Transcription transcribes audio to text.
//
// Azure does not support audio transcription.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by Azure",
		Provider: "azure",
	}
}

// Speech converts text to speech.
//
// Azure does not support text-to-speech.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by Azure",
		Provider: "azure",
	}
}

// ImageGeneration generates images from text prompts.
//
// Azure does not support image generation.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image generation is not supported by Azure",
		Provider: "azure",
	}
}

// ImageEdit edits an image using AI based on a text prompt.
//
// Azure does not support image editing.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by Azure",
		Provider: "azure",
	}
}

// ImageVariation creates variations of an existing image.
//
// Azure does not support image variation.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by Azure",
		Provider: "azure",
	}
}
