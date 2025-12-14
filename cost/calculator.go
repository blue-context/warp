package cost

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/blue-context/warp/types"
)

// ProviderGetter is an interface for retrieving providers.
// This allows both provider.Registry and client internal registry to be used.
type ProviderGetter interface {
	Get(name string) (interface{}, error)
}

// CompletionResponse represents a completion response with usage info.
type CompletionResponse interface {
	GetModel() string
	GetUsageInfo() interface{}
}

// EmbeddingResponse represents an embedding response with usage info.
type EmbeddingResponse interface {
	GetModel() string
	GetUsageInfo() interface{}
}

// usageInfo is a helper interface for extracting token counts from usage.
type usageInfo interface {
	GetPromptTokens() int
	GetCompletionTokens() int
	GetTotalTokens() int
}

// Calculator calculates costs for LLM requests and responses.
//
// The calculator queries providers for model pricing information and caches
// the results to minimize overhead. Users can override pricing via AddPricingOverride.
//
// Thread Safety: Calculator is safe for concurrent use.
type Calculator struct {
	providerGetter ProviderGetter
	cache          map[string]*cachedModelInfo
	overrides      map[string]*types.ModelInfo
	cacheTTL       time.Duration
	mu             sync.RWMutex
}

// cachedModelInfo holds cached model info with expiration.
type cachedModelInfo struct {
	Info      *types.ModelInfo
	ExpiresAt time.Time
}

// Pricing contains pricing information for a model (deprecated - use provider.ModelInfo).
//
// Deprecated: This type is maintained for backward compatibility.
// Use provider.ModelInfo instead.
type Pricing struct {
	Provider    string
	Model       string
	InputPer1M  float64 // Cost per 1M input tokens (USD)
	OutputPer1M float64 // Cost per 1M output tokens (USD)
	InputPer1K  float64 // Cost per 1K input tokens (USD) - computed
	OutputPer1K float64 // Cost per 1K output tokens (USD) - computed
}

// NewCalculator creates a new cost calculator that queries providers for pricing.
//
// The calculator caches pricing information for 1 hour to minimize overhead.
// Overrides can be added via AddPricingOverride for custom pricing.
//
// The registry parameter can be either provider.Registry or any type implementing
// ProviderGetter interface (e.g., client's internal registry).
func NewCalculator(registry ProviderGetter) *Calculator {
	return &Calculator{
		providerGetter: registry,
		cache:          make(map[string]*cachedModelInfo),
		overrides:      make(map[string]*types.ModelInfo),
		cacheTTL:       1 * time.Hour,
	}
}

// AddPricing adds or updates pricing for a model (deprecated).
//
// Deprecated: Use AddPricingOverride instead which uses provider.ModelInfo.
func (c *Calculator) AddPricing(p *Pricing) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Convert old Pricing to ModelInfo format
	info := &types.ModelInfo{
		Name:            p.Model,
		Provider:        p.Provider,
		InputCostPer1M:  p.InputPer1M,
		OutputCostPer1M: p.OutputPer1M,
	}

	key := p.Provider + "/" + p.Model
	c.overrides[key] = info
}

// AddPricingOverride adds or updates pricing override for a model.
//
// Overrides take precedence over provider-reported pricing.
// Useful for custom contracts, testing, or self-hosted deployments.
//
// Thread Safety: Safe for concurrent use.
func (c *Calculator) AddPricingOverride(providerName, model string, info *types.ModelInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := providerName + "/" + model
	c.overrides[key] = info
}

// GetPricing retrieves pricing for a model (deprecated).
//
// Deprecated: Use GetModelInfo instead which returns provider.ModelInfo.
func (c *Calculator) GetPricing(providerName, model string) (*Pricing, error) {
	info, err := c.GetModelInfo(providerName, model)
	if err != nil {
		return nil, err
	}

	// Convert ModelInfo to old Pricing format
	return &Pricing{
		Provider:    info.Provider,
		Model:       info.Name,
		InputPer1M:  info.InputCostPer1M,
		OutputPer1M: info.OutputCostPer1M,
		InputPer1K:  info.InputCostPer1M / 1000.0,
		OutputPer1K: info.OutputCostPer1M / 1000.0,
	}, nil
}

// GetModelInfo retrieves model information from provider or cache.
//
// Checks in this order:
// 1. User-defined overrides
// 2. Cache (if not expired)
// 3. Provider registry
//
// Thread Safety: Safe for concurrent use.
func (c *Calculator) GetModelInfo(providerName, model string) (*types.ModelInfo, error) {
	cacheKey := providerName + "/" + model

	// Check override first
	c.mu.RLock()
	if override, exists := c.overrides[cacheKey]; exists {
		c.mu.RUnlock()
		return override, nil
	}

	// Check cache
	if cached, exists := c.cache[cacheKey]; exists {
		if time.Now().Before(cached.ExpiresAt) {
			c.mu.RUnlock()
			return cached.Info, nil
		}
	}
	c.mu.RUnlock()

	// Query provider
	pInterface, err := c.providerGetter.Get(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %s not found: %w", providerName, err)
	}

	// Type assert to provider with GetModelInfo method
	type modelInfoProvider interface {
		GetModelInfo(model string) *types.ModelInfo
	}

	p, ok := pInterface.(modelInfoProvider)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support model info", providerName)
	}

	info := p.GetModelInfo(model)
	if info == nil {
		return nil, fmt.Errorf("model %s not found for provider %s", model, providerName)
	}

	// Cache the result
	c.mu.Lock()
	c.cache[cacheKey] = &cachedModelInfo{
		Info:      info,
		ExpiresAt: time.Now().Add(c.cacheTTL),
	}
	c.mu.Unlock()

	return info, nil
}

// ClearCache clears the pricing cache.
//
// This forces the calculator to re-query providers on the next request.
// Useful after provider configuration changes or for testing.
func (c *Calculator) ClearCache() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[string]*cachedModelInfo)
}

// CalculateCompletion calculates cost for a completion response.
//
// Queries the provider for model pricing and calculates the total cost
// based on token usage. Caches pricing information for performance.
func (c *Calculator) CalculateCompletion(resp CompletionResponse) (float64, error) {
	if resp == nil {
		return 0, fmt.Errorf("response cannot be nil")
	}

	usageRaw := resp.GetUsageInfo()
	if usageRaw == nil {
		return 0, fmt.Errorf("usage information not available")
	}

	// Type assert to usageInfo interface
	usage, ok := usageRaw.(usageInfo)
	if !ok {
		return 0, fmt.Errorf("usage does not implement required interface")
	}

	// Parse provider/model from response
	providerName, model := parseModel(resp.GetModel())

	// Get pricing from provider
	info, err := c.GetModelInfo(providerName, model)
	if err != nil {
		return 0, err
	}

	// Calculate cost using per-1M pricing
	inputCost := float64(usage.GetPromptTokens()) / 1_000_000.0 * info.InputCostPer1M
	outputCost := float64(usage.GetCompletionTokens()) / 1_000_000.0 * info.OutputCostPer1M

	return inputCost + outputCost, nil
}

// CalculateEmbedding calculates cost for an embedding response.
//
// Embeddings only use input pricing (no output tokens).
func (c *Calculator) CalculateEmbedding(resp EmbeddingResponse) (float64, error) {
	if resp == nil {
		return 0, fmt.Errorf("response cannot be nil")
	}

	usageRaw := resp.GetUsageInfo()
	if usageRaw == nil {
		return 0, fmt.Errorf("usage information not available")
	}

	// Type assert to usageInfo interface
	usage, ok := usageRaw.(usageInfo)
	if !ok {
		return 0, fmt.Errorf("usage does not implement required interface")
	}

	// Parse provider/model
	providerName, model := parseModel(resp.GetModel())

	// Get pricing from provider
	info, err := c.GetModelInfo(providerName, model)
	if err != nil {
		return 0, err
	}

	// Embeddings use input pricing only (per-1M)
	cost := float64(usage.GetTotalTokens()) / 1_000_000.0 * info.InputCostPer1M

	return cost, nil
}

// EstimateCost estimates cost before sending request.
//
// Useful for budget management and displaying cost estimates to users.
func (c *Calculator) EstimateCost(providerName, model string, inputTokens, outputTokens int) (float64, error) {
	info, err := c.GetModelInfo(providerName, model)
	if err != nil {
		return 0, err
	}

	inputCost := float64(inputTokens) / 1_000_000.0 * info.InputCostPer1M
	outputCost := float64(outputTokens) / 1_000_000.0 * info.OutputCostPer1M

	return inputCost + outputCost, nil
}

// parseModel parses a model string into provider and model name.
// Format: "provider/model-name" or just "model-name" (defaults to openai).
func parseModel(modelStr string) (provider, model string) {
	parts := strings.SplitN(modelStr, "/", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "openai", modelStr // Default to OpenAI
}
