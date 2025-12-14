// Package vllm implements the vLLM provider for Warp.
//
// vLLM is a self-hosted inference engine for large language models.
// It runs on localhost by default (http://localhost:8000) and does not require authentication.
//
// Supported models: Any model loaded in the vLLM server (meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1, etc.)
//
// Basic usage:
//
//	provider, err := vllm.NewProvider(
//	    vllm.WithBaseURL("http://localhost:8000"),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "meta-llama/Llama-2-7b-hf",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
package vllm

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for vLLM.
//
// vLLM is a self-hosted LLM inference engine that doesn't require authentication.
// It uses native endpoints (/inference/v1/generate) rather than OpenAI-compatible endpoints.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	baseURL    string
	apiKey     string // Optional for self-hosted deployments
	httpClient warp.HTTPClient
}

// Option is a functional option for configuring the vLLM provider.
type Option func(*Provider)

// NewProvider creates a new vLLM provider with the given options.
//
// No API key is required by default since vLLM runs locally and native endpoints
// don't enforce authentication. However, an API key can be provided for custom deployments.
// The default base URL is http://localhost:8000.
//
// Example:
//
//	provider, err := vllm.NewProvider(
//	    vllm.WithBaseURL("http://localhost:8000"),
//	)
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		baseURL:    "http://localhost:8000",
		httpClient: &http.Client{Timeout: 120 * time.Second}, // Longer timeout for local generation
	}

	for _, opt := range opts {
		opt(p)
	}

	return p, nil
}

// WithBaseURL sets a custom base URL.
//
// This is useful if vLLM is running on a different host or port.
// The default is "http://localhost:8000".
//
// Example:
//
//	provider, err := vllm.NewProvider(
//	    vllm.WithBaseURL("http://192.168.1.100:8000"),
//	)
func WithBaseURL(url string) Option {
	return func(p *Provider) {
		p.baseURL = url
	}
}

// WithAPIKey sets an optional API key.
//
// Note: vLLM's native endpoints (/inference/v1/generate) do not enforce
// API key authentication by default. This option is provided for custom
// deployments that may add authentication via a reverse proxy.
//
// Example:
//
//	provider, err := vllm.NewProvider(
//	    vllm.WithAPIKey("secret-token-abc123"),
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
//	}
//	provider, err := vllm.NewProvider(
//	    vllm.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "vllm".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "vllm"
}

// Supports returns the capabilities supported by vLLM.
//
// vLLM supports completion, streaming, embeddings (via /pooling), and reranking (via /rerank).
// vLLM does not support image generation, transcription, speech, or moderation.
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       true, // via /pooling endpoint
		ImageGeneration: false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: false,
		Vision:          false,
		JSON:            true, // vLLM supports JSON mode
		Rerank:          true, // via /rerank endpoint
	}
}

// ImageGeneration generates images from text prompts.
//
// This provider does not support image generation.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image generation not supported by vllm provider")
}

// ImageEdit edits an image using AI based on a text prompt.
//
// This provider does not support image editing.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image editing not supported by vllm provider")
}

// ImageVariation creates variations of an existing image.
//
// This provider does not support image variation.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image variation not supported by vllm provider")
}

// Transcription transcribes audio to text.
//
// This provider does not support transcription.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, fmt.Errorf("transcription not supported by vllm provider")
}

// Speech converts text to speech.
//
// This provider does not support text-to-speech.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("speech synthesis not supported by vllm provider")
}

// Moderation checks content for policy violations.
//
// This provider does not support moderation.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, fmt.Errorf("moderation not supported by vllm provider")
}
