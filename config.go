package warp

import (
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/blue-context/warp/cache"
	"github.com/blue-context/warp/callback"
)

// HTTPClient defines the interface for HTTP clients.
// This allows injection of custom clients or mocks for testing.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// ClientConfig holds all client configuration.
type ClientConfig struct {
	// APIKeys stores API keys by provider name
	APIKeys map[string]string

	// APIBases stores custom API base URLs by provider (optional overrides)
	APIBases map[string]string

	// DefaultTimeout is the default timeout for all requests
	DefaultTimeout time.Duration

	// MaxRetries is the maximum number of retry attempts
	MaxRetries int

	// RetryDelay is the initial delay between retries
	RetryDelay time.Duration

	// RetryMultiplier is the multiplier for exponential backoff
	RetryMultiplier float64

	// FallbackModels are models to try in order on failure
	FallbackModels []string

	// Debug enables debug mode logging
	Debug bool

	// TrackCost enables cost tracking
	TrackCost bool

	// MaxBudget is the maximum budget limit (0 means no limit)
	MaxBudget float64

	// HTTPClient is the HTTP client to use for requests (injectable for testing)
	HTTPClient HTTPClient

	// Cache is the cache implementation to use (nil disables caching)
	Cache cache.Cache

	// Callbacks is the callback registry for request lifecycle hooks
	Callbacks *callback.Registry
}

// ClientOption is a functional option for configuring the client.
type ClientOption func(*ClientConfig) error

// defaultConfig returns default configuration.
func defaultConfig() *ClientConfig {
	return &ClientConfig{
		APIKeys:         make(map[string]string),
		APIBases:        make(map[string]string),
		DefaultTimeout:  60 * time.Second,
		MaxRetries:      3,
		RetryDelay:      1 * time.Second,
		RetryMultiplier: 2.0,
		Debug:           false,
		TrackCost:       false,
		MaxBudget:       0,
		HTTPClient:      &http.Client{Timeout: 60 * time.Second},
	}
}

// WithAPIKey sets an API key for a provider.
//
// The provider name should be lowercase (e.g., "openai", "anthropic").
// Returns an error if provider or key is empty.
//
// Example:
//
//	client, err := warp.NewClient(
//	    warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
//	    warp.WithAPIKey("anthropic", os.Getenv("ANTHROPIC_API_KEY")),
//	)
func WithAPIKey(provider, key string) ClientOption {
	return func(c *ClientConfig) error {
		if provider == "" {
			return fmt.Errorf("provider name is required")
		}
		if key == "" {
			return fmt.Errorf("API key is required for provider %s", provider)
		}
		if c.APIKeys == nil {
			c.APIKeys = make(map[string]string)
		}
		c.APIKeys[provider] = key
		return nil
	}
}

// WithAPIBase sets a custom API base URL for a provider.
//
// The provider name should be lowercase (e.g., "openai", "anthropic").
// Returns an error if provider or base URL is empty.
//
// Example:
//
//	warp.WithAPIBase("openai", "https://custom.openai.proxy.com/v1")
func WithAPIBase(provider, base string) ClientOption {
	return func(c *ClientConfig) error {
		if provider == "" {
			return fmt.Errorf("provider name is required")
		}
		if base == "" {
			return fmt.Errorf("API base is required for provider %s", provider)
		}
		if c.APIBases == nil {
			c.APIBases = make(map[string]string)
		}
		c.APIBases[provider] = base
		return nil
	}
}

// WithTimeout sets the default timeout for requests.
//
// Timeout must be positive. Returns an error if timeout is zero or negative.
//
// Example:
//
//	warp.WithTimeout(30 * time.Second)
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *ClientConfig) error {
		if timeout <= 0 {
			return fmt.Errorf("timeout must be positive, got %v", timeout)
		}
		c.DefaultTimeout = timeout
		return nil
	}
}

// WithRetries configures retry behavior with all parameters.
//
// Parameters:
//   - maxRetries: Maximum number of retry attempts (must be non-negative)
//   - initialDelay: Initial delay between retries (must be non-negative)
//   - multiplier: Multiplier for exponential backoff (must be positive)
//
// Returns an error if any parameter is invalid.
//
// Example:
//
//	warp.WithRetries(5, 2*time.Second, 2.5)
func WithRetries(maxRetries int, initialDelay time.Duration, multiplier float64) ClientOption {
	return func(c *ClientConfig) error {
		if maxRetries < 0 {
			return fmt.Errorf("maxRetries must be non-negative, got %d", maxRetries)
		}
		if initialDelay < 0 {
			return fmt.Errorf("initialDelay must be non-negative, got %v", initialDelay)
		}
		if multiplier <= 0 {
			return fmt.Errorf("multiplier must be positive, got %f", multiplier)
		}
		c.MaxRetries = maxRetries
		c.RetryDelay = initialDelay
		c.RetryMultiplier = multiplier
		return nil
	}
}

// WithMaxRetries sets only the maximum retry count.
//
// Max retries must be non-negative. Returns an error if negative.
//
// Example:
//
//	warp.WithMaxRetries(5)
func WithMaxRetries(max int) ClientOption {
	return func(c *ClientConfig) error {
		if max < 0 {
			return fmt.Errorf("max retries must be non-negative, got %d", max)
		}
		c.MaxRetries = max
		return nil
	}
}

// WithFallbacks sets fallback models to try on failure.
//
// Models are tried in the order provided. At least one model is required.
// Returns an error if no models are provided.
//
// Example:
//
//	warp.WithFallbacks("anthropic/claude-3-sonnet", "openai/gpt-3.5-turbo")
func WithFallbacks(models ...string) ClientOption {
	return func(c *ClientConfig) error {
		if len(models) == 0 {
			return fmt.Errorf("at least one fallback model is required")
		}
		c.FallbackModels = models
		return nil
	}
}

// WithDebug enables or disables debug mode.
//
// In debug mode, all requests and responses are logged.
//
// Example:
//
//	warp.WithDebug(true)
func WithDebug(debug bool) ClientOption {
	return func(c *ClientConfig) error {
		c.Debug = debug
		return nil
	}
}

// WithCostTracking enables or disables cost tracking.
//
// When enabled, the client tracks API usage costs.
//
// Example:
//
//	warp.WithCostTracking(true)
func WithCostTracking(track bool) ClientOption {
	return func(c *ClientConfig) error {
		c.TrackCost = track
		return nil
	}
}

// WithMaxBudget sets a maximum budget limit.
//
// If cost exceeds this limit, requests will fail with a budget error.
// Set to 0 to disable budget limits.
// Returns an error if budget is negative.
//
// Example:
//
//	warp.WithMaxBudget(10.0) // $10 limit
func WithMaxBudget(budget float64) ClientOption {
	return func(c *ClientConfig) error {
		if budget < 0 {
			return fmt.Errorf("budget must be non-negative, got %f", budget)
		}
		c.MaxBudget = budget
		return nil
	}
}

// WithHTTPClient sets a custom HTTP client.
//
// This is useful for testing or for using custom transports.
// Returns an error if client is nil.
//
// Example:
//
//	customClient := &http.Client{
//	    Timeout: 30 * time.Second,
//	    Transport: customTransport,
//	}
//	warp.WithHTTPClient(customClient)
func WithHTTPClient(client HTTPClient) ClientOption {
	return func(c *ClientConfig) error {
		if client == nil {
			return fmt.Errorf("HTTP client cannot be nil")
		}
		c.HTTPClient = client
		return nil
	}
}

// WithCache sets a cache implementation for response caching.
//
// When enabled, identical completion requests will be served from cache
// instead of making API calls. This reduces costs and latency.
//
// Pass nil to disable caching (default behavior).
//
// Example with memory cache:
//
//	cache := cache.NewMemoryCache(100 * 1024 * 1024) // 100MB
//	warp.WithCache(cache)
//
// Example with no-op cache (disable caching):
//
//	warp.WithCache(cache.NewNoopCache())
func WithCache(c cache.Cache) ClientOption {
	return func(cfg *ClientConfig) error {
		cfg.Cache = c
		return nil
	}
}

// LoadConfigFromEnv loads configuration from environment variables.
//
// Supported environment variables:
//   - OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_API_KEY, etc.
//   - OPENAI_API_BASE, ANTHROPIC_API_BASE, etc.
//   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME (for Bedrock)
//   - VERTEX_PROJECT, VERTEX_LOCATION (for Google Vertex AI)
//
// Returns a slice of ClientOption functions that can be passed to NewClient.
//
// Example:
//
//	config, err := warp.LoadConfigFromEnv()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	client, err := warp.NewClient(config...)
func LoadConfigFromEnv() ([]ClientOption, error) {
	var opts []ClientOption

	// Provider list - all major LLM providers
	providers := []string{
		"OPENAI", "ANTHROPIC", "AZURE", "COHERE", "REPLICATE",
		"HUGGINGFACE", "TOGETHER", "OPENROUTER", "AI21", "BASETEN",
		"VLLM", "NLP_CLOUD", "ALEPH_ALPHA", "PETALS", "DEEPINFRA",
		"PERPLEXITY", "GROQ", "DEEPSEEK", "MISTRAL", "OLLAMA",
	}

	// Load API keys and bases for each provider
	for _, provider := range providers {
		providerLower := strings.ToLower(provider)

		// API Key
		keyEnv := provider + "_API_KEY"
		if key := os.Getenv(keyEnv); key != "" {
			opts = append(opts, WithAPIKey(providerLower, key))
		}

		// API Base
		baseEnv := provider + "_API_BASE"
		if base := os.Getenv(baseEnv); base != "" {
			opts = append(opts, WithAPIBase(providerLower, base))
		}
	}

	// AWS credentials for Bedrock
	if awsKey := os.Getenv("AWS_ACCESS_KEY_ID"); awsKey != "" {
		opts = append(opts, WithAPIKey("aws_access_key_id", awsKey))
	}
	if awsSecret := os.Getenv("AWS_SECRET_ACCESS_KEY"); awsSecret != "" {
		opts = append(opts, WithAPIKey("aws_secret_access_key", awsSecret))
	}
	if awsRegion := os.Getenv("AWS_REGION_NAME"); awsRegion != "" {
		opts = append(opts, WithAPIKey("aws_region_name", awsRegion))
	}

	// Google Vertex AI
	if vertexProject := os.Getenv("VERTEX_PROJECT"); vertexProject != "" {
		opts = append(opts, WithAPIKey("vertex_project", vertexProject))
	}
	if vertexLocation := os.Getenv("VERTEX_LOCATION"); vertexLocation != "" {
		opts = append(opts, WithAPIKey("vertex_location", vertexLocation))
	}

	return opts, nil
}

// WithBeforeRequestCallback registers a before-request callback.
//
// Before-request callbacks are executed before sending the request to the provider.
// They can inspect the request and return an error to abort the request.
//
// The callback registry is created automatically on first use.
// Returns an error if the callback is nil.
//
// Example:
//
//	warp.WithBeforeRequestCallback(func(ctx context.Context, event *callback.BeforeRequestEvent) error {
//	    log.Printf("Request to %s/%s", event.Provider, event.Model)
//	    return nil
//	})
func WithBeforeRequestCallback(cb callback.BeforeRequestCallback) ClientOption {
	return func(c *ClientConfig) error {
		if cb == nil {
			return fmt.Errorf("callback cannot be nil")
		}
		if c.Callbacks == nil {
			c.Callbacks = callback.NewRegistry()
		}
		c.Callbacks.RegisterBeforeRequest(cb)
		return nil
	}
}

// WithSuccessCallback registers a success callback.
//
// Success callbacks are executed after successful responses from the provider.
// They receive metadata including cost, token usage, and duration.
//
// The callback registry is created automatically on first use.
// Returns an error if the callback is nil.
//
// Example:
//
//	warp.WithSuccessCallback(func(ctx context.Context, event *callback.SuccessEvent) {
//	    log.Printf("Success! Cost: $%.4f, Tokens: %d", event.Cost, event.Tokens)
//	})
func WithSuccessCallback(cb callback.SuccessCallback) ClientOption {
	return func(c *ClientConfig) error {
		if cb == nil {
			return fmt.Errorf("callback cannot be nil")
		}
		if c.Callbacks == nil {
			c.Callbacks = callback.NewRegistry()
		}
		c.Callbacks.RegisterSuccess(cb)
		return nil
	}
}

// WithFailureCallback registers a failure callback.
//
// Failure callbacks are executed after failed requests.
// They receive the error and request metadata for logging or alerting.
//
// The callback registry is created automatically on first use.
// Returns an error if the callback is nil.
//
// Example:
//
//	warp.WithFailureCallback(func(ctx context.Context, event *callback.FailureEvent) {
//	    log.Printf("Request failed: %v", event.Error)
//	})
func WithFailureCallback(cb callback.FailureCallback) ClientOption {
	return func(c *ClientConfig) error {
		if cb == nil {
			return fmt.Errorf("callback cannot be nil")
		}
		if c.Callbacks == nil {
			c.Callbacks = callback.NewRegistry()
		}
		c.Callbacks.RegisterFailure(cb)
		return nil
	}
}

// WithStreamCallback registers a streaming callback.
//
// Stream callbacks are executed for each chunk received during streaming.
// They receive the chunk data and index for real-time processing.
//
// The callback registry is created automatically on first use.
// Returns an error if the callback is nil.
//
// Example:
//
//	warp.WithStreamCallback(func(ctx context.Context, event *callback.StreamEvent) {
//	    log.Printf("Chunk %d received", event.Index)
//	})
func WithStreamCallback(cb callback.StreamCallback) ClientOption {
	return func(c *ClientConfig) error {
		if cb == nil {
			return fmt.Errorf("callback cannot be nil")
		}
		if c.Callbacks == nil {
			c.Callbacks = callback.NewRegistry()
		}
		c.Callbacks.RegisterStream(cb)
		return nil
	}
}

// Validate validates the configuration.
//
// Returns an error if any configuration value is invalid.
//
// Checks:
//   - DefaultTimeout must be positive
//   - MaxRetries must be non-negative
//   - RetryDelay must be non-negative
//   - RetryMultiplier must be positive
//   - MaxBudget must be non-negative
//   - HTTPClient must not be nil
func (c *ClientConfig) Validate() error {
	if c.DefaultTimeout <= 0 {
		return fmt.Errorf("default timeout must be positive")
	}
	if c.MaxRetries < 0 {
		return fmt.Errorf("max retries must be non-negative")
	}
	if c.RetryDelay < 0 {
		return fmt.Errorf("retry delay must be non-negative")
	}
	if c.RetryMultiplier <= 0 {
		return fmt.Errorf("retry multiplier must be positive")
	}
	if c.MaxBudget < 0 {
		return fmt.Errorf("max budget must be non-negative")
	}
	if c.HTTPClient == nil {
		return fmt.Errorf("HTTP client cannot be nil")
	}
	return nil
}
