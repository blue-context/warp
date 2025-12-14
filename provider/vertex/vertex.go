package vertex

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for Google Vertex AI.
//
// Vertex AI provides access to Google's Gemini models and other LLMs through
// a unified API. This implementation uses OAuth2 service account authentication
// with zero external dependencies.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	projectID     string
	location      string
	tokenProvider *TokenProvider
	httpClient    warp.HTTPClient
}

// Option is a functional option for configuring the Vertex AI provider.
type Option func(*Provider) error

// NewProvider creates a new Vertex AI provider with the given options.
//
// The provider requires:
//   - Project ID (via WithProjectID)
//   - Service account credentials (via WithServiceAccountKey)
//
// Optional configuration:
//   - Location (defaults to "us-central1")
//   - Custom HTTP client (defaults to 120s timeout)
//
// Example:
//
//	keyJSON, err := os.ReadFile("service-account-key.json")
//	if err != nil {
//	    return err
//	}
//
//	provider, err := vertex.NewProvider(
//	    vertex.WithProjectID("my-gcp-project"),
//	    vertex.WithLocation("us-central1"),
//	    vertex.WithServiceAccountKey(keyJSON),
//	)
//	if err != nil {
//	    return err
//	}
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		location:   "us-central1", // Default to us-central1
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}

	// Apply options
	for _, opt := range opts {
		if err := opt(p); err != nil {
			return nil, err
		}
	}

	// Validate required fields
	if p.projectID == "" {
		return nil, &warp.WarpError{
			Message:  "GCP project ID is required (use WithProjectID option)",
			Provider: "vertex",
		}
	}

	if p.tokenProvider == nil {
		return nil, &warp.WarpError{
			Message:  "service account credentials are required (use WithServiceAccountKey option)",
			Provider: "vertex",
		}
	}

	return p, nil
}

// WithProjectID sets the GCP project ID.
//
// This is required for Vertex AI API requests. The project ID can be found
// in the Google Cloud Console.
//
// Example:
//
//	provider, err := vertex.NewProvider(
//	    vertex.WithProjectID("my-gcp-project"),
//	    vertex.WithServiceAccountKey(keyJSON),
//	)
func WithProjectID(projectID string) Option {
	return func(p *Provider) error {
		if projectID == "" {
			return &warp.WarpError{
				Message:  "project ID cannot be empty",
				Provider: "vertex",
			}
		}
		p.projectID = projectID
		return nil
	}
}

// WithLocation sets the GCP location/region.
//
// The location determines which Vertex AI endpoint to use.
// Common locations: us-central1, us-east1, europe-west1, asia-southeast1
//
// Default: us-central1
//
// Example:
//
//	provider, err := vertex.NewProvider(
//	    vertex.WithProjectID("my-project"),
//	    vertex.WithLocation("europe-west1"),
//	    vertex.WithServiceAccountKey(keyJSON),
//	)
func WithLocation(location string) Option {
	return func(p *Provider) error {
		if location == "" {
			return &warp.WarpError{
				Message:  "location cannot be empty",
				Provider: "vertex",
			}
		}
		p.location = location
		return nil
	}
}

// WithServiceAccountKey sets credentials from a service account JSON key file.
//
// The keyJSON parameter should contain the contents of a GCP service account
// key file in JSON format. This file can be created in the Google Cloud Console
// under IAM & Admin > Service Accounts.
//
// This option is required for authentication.
//
// Example:
//
//	keyJSON, err := os.ReadFile("service-account-key.json")
//	if err != nil {
//	    return err
//	}
//
//	provider, err := vertex.NewProvider(
//	    vertex.WithProjectID("my-project"),
//	    vertex.WithServiceAccountKey(keyJSON),
//	)
func WithServiceAccountKey(keyJSON []byte) Option {
	return func(p *Provider) error {
		tp, err := NewTokenProvider(keyJSON)
		if err != nil {
			return fmt.Errorf("failed to create token provider: %w", err)
		}
		p.tokenProvider = tp
		return nil
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
//
//	provider, err := vertex.NewProvider(
//	    vertex.WithProjectID("my-project"),
//	    vertex.WithServiceAccountKey(keyJSON),
//	    vertex.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) error {
		if client == nil {
			return &warp.WarpError{
				Message:  "HTTP client cannot be nil",
				Provider: "vertex",
			}
		}
		p.httpClient = client
		return nil
	}
}

// Name returns the provider name "vertex".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "vertex"
}

// Supports returns the capabilities supported by Vertex AI.
//
// Vertex AI (Gemini) supports:
//   - Completion: Chat completion
//   - Streaming: Server-sent events streaming
//   - FunctionCalling: Function/tool calling
//   - Vision: Multimodal image understanding
//   - JSON: JSON mode for structured output
//
// Not supported via this provider:
//   - Embedding: Vertex has separate embedding models/API
//   - ImageGeneration: Vertex Imagen uses different API
//   - Transcription, Speech, Moderation: Not available
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       false, // Vertex embeddings use separate API
		ImageGeneration: false, // Vertex Imagen uses different endpoint
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true, // Gemini supports function calling
		Vision:          true, // Gemini supports vision
		JSON:            true, // Gemini supports JSON mode
	}
}

// Embedding returns an error as Vertex AI embeddings use a separate API.
//
// Vertex AI text embeddings (text-embedding-gecko, textembedding-gecko-multilingual)
// use a different API endpoint than the generateContent endpoint used for completions.
//
// This method exists to satisfy the provider.Provider interface.
// Always returns an error indicating embeddings are not supported via this provider.
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	return nil, &warp.WarpError{
		Message:  "Vertex AI embeddings require separate API endpoint (not yet implemented)",
		Provider: "vertex",
	}
}

// buildEndpoint constructs the Vertex AI API endpoint URL.
//
// Endpoint format for completion:
//
//	https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
//
// Endpoint format for streaming:
//
//	https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:streamGenerateContent
//
// The model parameter should be just the model name (e.g., "gemini-pro", "gemini-1.5-pro").
// The "vertex/" prefix is stripped if present.
func (p *Provider) buildEndpoint(model string, stream bool) string {
	// Strip "vertex/" prefix if present
	if len(model) > 7 && model[:7] == "vertex/" {
		model = model[7:]
	}

	action := "generateContent"
	if stream {
		action = "streamGenerateContent"
	}

	return fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:%s",
		p.location,
		p.projectID,
		p.location,
		model,
		action,
	)
}

// Rerank ranks documents by relevance to a query.
//
// This provider does not support document reranking.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, fmt.Errorf("rerank not supported by vertex provider")
}
