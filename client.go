package warp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/blue-context/warp/cache"
	"github.com/blue-context/warp/callback"
	"github.com/blue-context/warp/cost"
)

// Provider defines the interface that all LLM providers must implement.
//
// This is a minimal interface used by the client to call providers.
// The full Provider interface with additional capabilities is defined in the provider package.
type Provider interface {
	// Name returns the provider name (e.g., "openai", "anthropic").
	Name() string

	// Completion sends a chat completion request.
	Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// CompletionStream sends a streaming chat completion request.
	CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error)

	// Embedding sends an embedding request.
	Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)

	// Transcription transcribes audio to text.
	Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)

	// Speech converts text to speech.
	Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error)

	// Moderation checks content for policy violations.
	Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error)

	// Rerank ranks documents by relevance to a query.
	Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error)

	// Supports returns the capabilities supported by this provider.
	Supports() interface{} // Returns provider.Capabilities to avoid import cycle
}

// Client is the main interface for interacting with Warp.
//
// Thread Safety: Client is safe for concurrent use from multiple goroutines.
//
// Example:
//
//	client, err := warp.NewClient(
//	    warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
type Client interface {
	// Completion creates a chat completion
	Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)

	// CompletionStream creates a streaming chat completion
	CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error)

	// Embedding creates embeddings
	Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)

	// ImageGeneration generates images from text prompts
	ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error)

	// ImageEdit edits an image using AI based on a text prompt
	ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageGenerationResponse, error)

	// ImageVariation creates variations of an existing image
	ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageGenerationResponse, error)

	// Transcription transcribes audio to text
	Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)

	// Speech converts text to speech.
	//
	// Returns an io.ReadCloser containing the audio data.
	// The caller MUST call Close() when done to release resources.
	//
	// Example:
	//   audio, err := client.Speech(ctx, &warp.SpeechRequest{
	//       Model: "openai/tts-1",
	//       Input: "Hello, world!",
	//       Voice: "alloy",
	//   })
	//   if err != nil {
	//       log.Fatal(err)
	//   }
	//   defer audio.Close()
	//
	//   // Write to file
	//   out, _ := os.Create("output.mp3")
	//   defer out.Close()
	//   io.Copy(out, audio)
	Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error)

	// Moderation checks content for policy violations.
	//
	// The input can be a single string or an array of strings.
	// Returns moderation results indicating whether content is flagged.
	//
	// Example:
	//   resp, err := client.Moderation(ctx, &warp.ModerationRequest{
	//       Input: "Text to check",
	//   })
	//   if err != nil {
	//       log.Fatal(err)
	//   }
	//   if resp.Results[0].Flagged {
	//       fmt.Println("Content flagged")
	//   }
	Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error)

	// Rerank ranks documents by relevance to a query.
	//
	// Used for RAG (Retrieval-Augmented Generation) applications to rank
	// retrieved documents before feeding them to an LLM.
	//
	// Example:
	//   resp, err := client.Rerank(ctx, &warp.RerankRequest{
	//       Model: "cohere/rerank-english-v3.0",
	//       Query: "What is the capital of France?",
	//       Documents: []string{
	//           "Paris is the capital of France",
	//           "London is the capital of England",
	//       },
	//       TopN: warp.IntPtr(1),
	//   })
	//   if err != nil {
	//       log.Fatal(err)
	//   }
	//   for _, result := range resp.Results {
	//       fmt.Printf("Document %d: score=%.3f\n", result.Index, result.RelevanceScore)
	//   }
	Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error)

	// CompletionCost calculates the cost of a completion
	//
	// Returns 0 if pricing information is not available.
	CompletionCost(resp *CompletionResponse) (float64, error)

	// Close closes the client and releases resources
	Close() error

	// RegisterProvider registers a provider with the client
	RegisterProvider(p Provider) error
}

// client implements the Client interface
type client struct {
	config           *ClientConfig
	providers        map[string]Provider
	providerRegistry providerRegistry // Internal registry for cost calculator
	costCalc         *cost.Calculator
	budget           *cost.BudgetManager
	cache            cache.Cache
	callbacks        *callback.Registry
	mu               sync.RWMutex
	randMu           sync.Mutex
	randSrc          *rand.Rand
}

// providerRegistry wraps the client's provider map to implement cost.ProviderGetter interface.
// This avoids duplication while providing the interface expected by cost.Calculator.
type providerRegistry struct {
	client *client
}

// Get retrieves a provider by name.
func (r providerRegistry) Get(name string) (interface{}, error) {
	return r.client.getProvider(name)
}

// NewClient creates a new Warp client.
//
// Options can be provided to configure the client.
//
// Example:
//
//	client, err := warp.NewClient(
//	    warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
//	    warp.WithAPIKey("anthropic", os.Getenv("ANTHROPIC_API_KEY")),
//	    warp.WithTimeout(30 * time.Second),
//	    warp.WithRetries(3, time.Second, 2.0),
//	)
func NewClient(opts ...ClientOption) (Client, error) {
	// Create default config
	config := defaultConfig()

	// Apply options
	for _, opt := range opts {
		if err := opt(config); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}

	// Validate config
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	// Create client
	c := &client{
		config:    config,
		providers: make(map[string]Provider),
		randSrc:   rand.New(rand.NewSource(time.Now().UnixNano())),
		cache:     config.Cache,
		callbacks: config.Callbacks,
	}

	// Create provider registry wrapper
	c.providerRegistry = providerRegistry{client: c}

	// Initialize cost tracking if enabled
	if config.TrackCost {
		c.costCalc = cost.NewCalculator(c.providerRegistry)
		if config.MaxBudget > 0 {
			c.budget = cost.NewBudgetManager(config.MaxBudget)
		}
	}

	// Note: Providers must be registered externally using RegisterProvider method.
	// This avoids import cycles between the warp and provider packages.
	//
	// Example:
	//   client, _ := warp.NewClient()
	//   provider, _ := openai.NewProvider(openai.WithAPIKey("sk-..."))
	//   client.RegisterProvider(provider)

	return c, nil
}

// parseModel parses a model string into provider and model name.
//
// Format: "provider/model-name"
// Examples: "openai/gpt-4", "anthropic/claude-3-sonnet"
//
// Returns an error if the format is invalid.
func parseModel(model string) (provider, modelName string, err error) {
	parts := strings.SplitN(model, "/", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid model format: %q (expected format: provider/model-name)", model)
	}

	provider = parts[0]
	modelName = parts[1]

	if provider == "" {
		return "", "", fmt.Errorf("provider name is empty in model: %q", model)
	}
	if modelName == "" {
		return "", "", fmt.Errorf("model name is empty in model: %q", model)
	}

	return provider, modelName, nil
}

// withRetry executes a function with retry logic
func (c *client) withRetry(ctx context.Context, fn func() error) error {
	var lastErr error

	maxRetries := c.config.MaxRetries
	if maxRetries < 0 {
		maxRetries = 0
	}

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Check context cancellation before each attempt
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Execute function
		err := fn()
		if err == nil {
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !isRetryable(err) {
			return err
		}

		// Check if we've exhausted retries
		if attempt == maxRetries {
			return fmt.Errorf("max retries (%d) exceeded: %w", maxRetries, err)
		}

		// Calculate delay
		delay := c.calculateDelay(attempt)

		// Wait with context cancellation support
		select {
		case <-time.After(delay):
			// Continue to next attempt
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	return lastErr
}

// calculateDelay calculates the retry delay with exponential backoff
func (c *client) calculateDelay(attempt int) time.Duration {
	delay := float64(c.config.RetryDelay) * math.Pow(c.config.RetryMultiplier, float64(attempt))

	// Add jitter (±10%) with thread-safe random source
	c.randMu.Lock()
	jitter := c.randSrc.Float64()*0.2 - 0.1
	c.randMu.Unlock()
	delay = delay * (1.0 + jitter)

	// Cap at 60 seconds
	if delay > 60*float64(time.Second) {
		delay = 60 * float64(time.Second)
	}

	return time.Duration(delay)
}

// isRetryable checks if an error is retryable
func isRetryable(err error) bool {
	// Check for specific retryable error types
	var rateLimitErr *RateLimitError
	if errors.As(err, &rateLimitErr) {
		return true
	}

	var timeoutErr *TimeoutError
	if errors.As(err, &timeoutErr) {
		return true
	}

	var serviceErr *ServiceUnavailableError
	if errors.As(err, &serviceErr) {
		return true
	}

	// Check generic WarpError with IsRetryable method
	// This handles any custom error types that implement the interface
	type retryable interface {
		IsRetryable() bool
	}

	var retryableErr retryable
	if errors.As(err, &retryableErr) {
		return retryableErr.IsRetryable()
	}

	return false
}

// CompletionCost calculates the cost of a completion.
//
// Returns the cost in USD, or error if pricing not available.
//
// Example:
//
//	cost, err := client.CompletionCost(resp)
//	fmt.Printf("Cost: $%.4f\n", cost)
func (c *client) CompletionCost(resp *CompletionResponse) (float64, error) {
	if resp == nil {
		return 0, fmt.Errorf("response cannot be nil")
	}
	if resp.Usage == nil {
		return 0, fmt.Errorf("usage information not available in response")
	}

	// Use cost calculator
	if c.costCalc == nil {
		c.costCalc = cost.NewCalculator(c.providerRegistry)
	}

	return c.costCalc.CalculateCompletion(resp)
}

// Close closes the client and releases resources.
//
// After calling Close, the client should not be used.
func (c *client) Close() error {
	// Close cache if present
	if c.cache != nil {
		if err := c.cache.Close(); err != nil {
			return fmt.Errorf("failed to close cache: %w", err)
		}
	}
	return nil
}

// RegisterProvider registers a provider with the client.
//
// This allows external packages to register providers without creating
// import cycles.
//
// Example:
//
//	client, err := warp.NewClient()
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	openaiProvider, err := openai.NewProvider(openai.WithAPIKey("sk-..."))
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	if err := client.RegisterProvider(openaiProvider); err != nil {
//	    log.Fatal(err)
//	}
func (c *client) RegisterProvider(p Provider) error {
	if p == nil {
		return fmt.Errorf("provider cannot be nil")
	}

	name := p.Name()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.providers[name]; exists {
		return fmt.Errorf("provider %q already registered", name)
	}

	c.providers[name] = p
	return nil
}

// getProvider retrieves a provider by name.
func (c *client) getProvider(name string) (Provider, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	p, exists := c.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider %q not found", name)
	}

	return p, nil
}
