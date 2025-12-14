package bedrock

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider"
)

// Provider implements the provider.Provider interface for AWS Bedrock.
//
// AWS Bedrock provides access to foundation models from Anthropic, Meta, Amazon,
// Cohere, and other providers through a unified AWS API.
//
// This implementation uses zero dependencies by implementing AWS Signature Version 4
// signing using only the Go standard library.
//
// Thread Safety: Provider is safe for concurrent use.
// Multiple goroutines may call methods on the same Provider instance simultaneously.
type Provider struct {
	accessKeyID     string
	secretAccessKey string
	sessionToken    string
	region          string
	httpClient      warp.HTTPClient
	signer          *Signer
}

// Option is a functional option for configuring the Bedrock provider.
type Option func(*Provider)

// NewProvider creates a new AWS Bedrock provider with the given options.
//
// The provider requires AWS credentials (access key ID and secret access key)
// to be set via WithCredentials option. A region must also be specified via
// WithRegion option (defaults to us-east-1).
//
// Example:
//
//	provider, err := bedrock.NewProvider(
//	    bedrock.WithCredentials(
//	        os.Getenv("AWS_ACCESS_KEY_ID"),
//	        os.Getenv("AWS_SECRET_ACCESS_KEY"),
//	    ),
//	    bedrock.WithRegion("us-east-1"),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
func NewProvider(opts ...Option) (*Provider, error) {
	p := &Provider{
		region:     "us-east-1", // Default region
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}

	for _, opt := range opts {
		opt(p)
	}

	// Validate required fields
	if p.accessKeyID == "" {
		return nil, &warp.WarpError{
			Message:  "AWS access key ID is required",
			Provider: "bedrock",
		}
	}

	if p.secretAccessKey == "" {
		return nil, &warp.WarpError{
			Message:  "AWS secret access key is required",
			Provider: "bedrock",
		}
	}

	if p.region == "" {
		return nil, &warp.WarpError{
			Message:  "AWS region is required",
			Provider: "bedrock",
		}
	}

	// Create signer with session token if present
	var signerOpts []SignerOption
	if p.sessionToken != "" {
		signerOpts = append(signerOpts, WithSignerSessionToken(p.sessionToken))
	}
	p.signer = NewSigner(p.accessKeyID, p.secretAccessKey, p.region, signerOpts...)

	return p, nil
}

// WithCredentials sets AWS credentials.
//
// Both access key ID and secret access key are required for authentication.
// These can be obtained from the AWS IAM console.
//
// Example:
//
//	provider, err := bedrock.NewProvider(
//	    bedrock.WithCredentials(
//	        os.Getenv("AWS_ACCESS_KEY_ID"),
//	        os.Getenv("AWS_SECRET_ACCESS_KEY"),
//	    ),
//	)
func WithCredentials(accessKeyID, secretAccessKey string) Option {
	return func(p *Provider) {
		p.accessKeyID = accessKeyID
		p.secretAccessKey = secretAccessKey
	}
}

// WithSessionToken sets AWS session token for temporary credentials.
//
// Session tokens are used with temporary security credentials from AWS STS.
// This is optional and only needed when using temporary credentials.
//
// Example:
//
//	provider, err := bedrock.NewProvider(
//	    bedrock.WithCredentials(accessKey, secretKey),
//	    bedrock.WithSessionToken(sessionToken),
//	    bedrock.WithRegion("us-east-1"),
//	)
func WithSessionToken(token string) Option {
	return func(p *Provider) {
		p.sessionToken = token
	}
}

// WithRegion sets AWS region.
//
// The region determines which AWS Bedrock endpoint to use.
// Available regions include: us-east-1, us-west-2, eu-west-1, ap-southeast-1, etc.
//
// Default: us-east-1
//
// Example:
//
//	provider, err := bedrock.NewProvider(
//	    bedrock.WithCredentials(accessKey, secretKey),
//	    bedrock.WithRegion("us-west-2"),
//	)
func WithRegion(region string) Option {
	return func(p *Provider) {
		p.region = region
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
//	provider, err := bedrock.NewProvider(
//	    bedrock.WithCredentials(accessKey, secretKey),
//	    bedrock.WithHTTPClient(customClient),
//	)
func WithHTTPClient(client warp.HTTPClient) Option {
	return func(p *Provider) {
		p.httpClient = client
	}
}

// Name returns the provider name "bedrock".
//
// This is used for provider identification in the registry and error messages.
func (p *Provider) Name() string {
	return "bedrock"
}

// Supports returns the capabilities supported by AWS Bedrock.
//
// Bedrock supports completion and streaming for most models.
// Function calling and vision are supported for Claude models.
// Embeddings are handled separately and not supported via the completion endpoint.
//
// Specific capabilities vary by model family:
//   - Claude: completion, streaming, function calling, vision
//   - Llama: completion, streaming, (some models support vision)
//   - Titan: completion, streaming
//   - Cohere: completion, streaming
func (p *Provider) Supports() interface{} {
	return provider.Capabilities{
		Completion:      true,
		Streaming:       true,
		Embedding:       false, // Bedrock embeddings use separate endpoint
		ImageGeneration: false, // Stability AI image generation uses different endpoint
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true,  // Supported for Claude models
		Vision:          true,  // Supported for Claude and some Llama models
		JSON:            false, // No explicit JSON mode
	}
}

// Embedding returns an error as Bedrock embeddings use a separate endpoint.
//
// AWS Bedrock embeddings (Titan, Cohere) use a different API endpoint than
// the invoke endpoint used for completions.
//
// This method exists to satisfy the provider.Provider interface.
// Always returns an error indicating embeddings are not supported via this provider.
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	return nil, &warp.WarpError{
		Message:  "AWS Bedrock embeddings require separate endpoint implementation",
		Provider: "bedrock",
	}
}

// buildEndpoint constructs the Bedrock API endpoint URL.
//
// Bedrock endpoints follow the format:
//
//	https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/invoke
//
// For streaming requests, the endpoint is:
//
//	https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/invoke-with-response-stream
func (p *Provider) buildEndpoint(modelID string, stream bool) string {
	action := "invoke"
	if stream {
		action = "invoke-with-response-stream"
	}

	return fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/%s",
		p.region,
		modelID,
		action,
	)
}

// Rerank ranks documents by relevance to a query.
//
// This provider does not support document reranking.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, fmt.Errorf("rerank not supported by bedrock provider")
}
