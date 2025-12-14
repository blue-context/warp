package cost

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/blue-context/warp/types"
)

// mockUsage implements UsageInfo for testing
type mockUsage struct {
	promptTokens     int
	completionTokens int
	totalTokens      int
}

func (m *mockUsage) GetPromptTokens() int     { return m.promptTokens }
func (m *mockUsage) GetCompletionTokens() int { return m.completionTokens }
func (m *mockUsage) GetTotalTokens() int      { return m.totalTokens }

// mockCompletionResponse implements CompletionResponse for testing
type mockCompletionResponse struct {
	model string
	usage *mockUsage
}

func (m *mockCompletionResponse) GetModel() string { return m.model }
func (m *mockCompletionResponse) GetUsageInfo() interface{} {
	if m.usage == nil {
		return nil
	}
	return m.usage
}

// mockEmbeddingResponse implements EmbeddingResponse for testing
type mockEmbeddingResponse struct {
	model string
	usage *mockUsage
}

func (m *mockEmbeddingResponse) GetModel() string { return m.model }
func (m *mockEmbeddingResponse) GetUsageInfo() interface{} {
	if m.usage == nil {
		return nil
	}
	return m.usage
}

// mockProvider implements the provider interface with model info.
type mockProvider struct {
	models map[string]*types.ModelInfo
}

func (m *mockProvider) GetModelInfo(model string) *types.ModelInfo {
	return m.models[model]
}

func (m *mockProvider) ListModels() []*types.ModelInfo {
	models := make([]*types.ModelInfo, 0, len(m.models))
	for _, info := range m.models {
		models = append(models, info)
	}
	return models
}

// mockProviderRegistry implements ProviderGetter for testing.
type mockProviderRegistry struct {
	providers map[string]*mockProvider
}

func (m *mockProviderRegistry) Get(name string) (interface{}, error) {
	if provider, exists := m.providers[name]; exists {
		return provider, nil
	}
	return nil, fmt.Errorf("provider %s not found", name)
}

// newMockRegistry creates a mock registry with common test models.
func newMockRegistry() *mockProviderRegistry {
	return &mockProviderRegistry{
		providers: map[string]*mockProvider{
			"openai": {
				models: map[string]*types.ModelInfo{
					"gpt-4": {
						Name:            "gpt-4",
						Provider:        "openai",
						ContextWindow:   8192,
						InputCostPer1M:  30.00,
						OutputCostPer1M: 60.00,
						Capabilities: types.Capabilities{
							Completion: true,
							Streaming:  true,
						},
					},
					"gpt-3.5-turbo": {
						Name:            "gpt-3.5-turbo",
						Provider:        "openai",
						ContextWindow:   16385,
						InputCostPer1M:  0.50,
						OutputCostPer1M: 1.50,
						Capabilities: types.Capabilities{
							Completion: true,
							Streaming:  true,
						},
					},
					"text-embedding-ada-002": {
						Name:            "text-embedding-ada-002",
						Provider:        "openai",
						ContextWindow:   8191,
						InputCostPer1M:  0.10,
						OutputCostPer1M: 0.00,
						Capabilities: types.Capabilities{
							Embedding: true,
						},
					},
					"text-embedding-3-small": {
						Name:            "text-embedding-3-small",
						Provider:        "openai",
						ContextWindow:   8191,
						InputCostPer1M:  0.02,
						OutputCostPer1M: 0.00,
						Capabilities: types.Capabilities{
							Embedding: true,
						},
					},
				},
			},
			"anthropic": {
				models: map[string]*types.ModelInfo{
					"claude-3-opus-20240229": {
						Name:            "claude-3-opus-20240229",
						Provider:        "anthropic",
						ContextWindow:   200000,
						InputCostPer1M:  15.00,
						OutputCostPer1M: 75.00,
						Capabilities: types.Capabilities{
							Completion: true,
							Streaming:  true,
						},
					},
					"claude-3-sonnet-20240229": {
						Name:            "claude-3-sonnet-20240229",
						Provider:        "anthropic",
						ContextWindow:   200000,
						InputCostPer1M:  3.00,
						OutputCostPer1M: 15.00,
						Capabilities: types.Capabilities{
							Completion: true,
							Streaming:  true,
						},
					},
				},
			},
			"azure": {
				models: map[string]*types.ModelInfo{
					"gpt-4": {
						Name:            "gpt-4",
						Provider:        "azure",
						ContextWindow:   8192,
						InputCostPer1M:  30.00,
						OutputCostPer1M: 60.00,
						Capabilities: types.Capabilities{
							Completion: true,
							Streaming:  true,
						},
					},
				},
			},
		},
	}
}

func TestNewCalculator(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)
	if calc == nil {
		t.Fatal("NewCalculator returned nil")
	}

	if calc.providerGetter == nil {
		t.Error("Expected provider getter to be set")
	}

	// Verify some key models exist
	tests := []struct {
		provider string
		model    string
	}{
		{"openai", "gpt-4"},
		{"openai", "gpt-3.5-turbo"},
		{"openai", "text-embedding-ada-002"},
		{"anthropic", "claude-3-opus-20240229"},
		{"anthropic", "claude-3-sonnet-20240229"},
		{"azure", "gpt-4"},
	}

	for _, tt := range tests {
		t.Run(tt.provider+"/"+tt.model, func(t *testing.T) {
			info, err := calc.GetModelInfo(tt.provider, tt.model)
			if err != nil {
				t.Errorf("Expected model info for %s/%s, got error: %v", tt.provider, tt.model, err)
				return
			}
			if info == nil {
				t.Errorf("Expected model info for %s/%s, got nil", tt.provider, tt.model)
			}
		})
	}
}

func TestAddPricing(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	tests := []struct {
		name    string
		pricing *Pricing
		wantKey string
	}{
		{
			name: "add new pricing",
			pricing: &Pricing{
				Provider:    "test",
				Model:       "test-model",
				InputPer1M:  10.0,
				OutputPer1M: 20.0,
			},
			wantKey: "test/test-model",
		},
		{
			name: "update existing pricing",
			pricing: &Pricing{
				Provider:    "openai",
				Model:       "gpt-4",
				InputPer1M:  100.0,
				OutputPer1M: 200.0,
			},
			wantKey: "openai/gpt-4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calc.AddPricing(tt.pricing)

			// Verify override was added
			calc.mu.RLock()
			override, exists := calc.overrides[tt.wantKey]
			calc.mu.RUnlock()

			if !exists {
				t.Fatalf("Override not found for key %s", tt.wantKey)
			}

			// Verify values match
			if override.InputCostPer1M != tt.pricing.InputPer1M {
				t.Errorf("InputCostPer1M = %f, want %f", override.InputCostPer1M, tt.pricing.InputPer1M)
			}
			if override.OutputCostPer1M != tt.pricing.OutputPer1M {
				t.Errorf("OutputCostPer1M = %f, want %f", override.OutputCostPer1M, tt.pricing.OutputPer1M)
			}
		})
	}
}

func TestGetPricing(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	tests := []struct {
		name      string
		provider  string
		model     string
		wantErr   bool
		checkCost bool
	}{
		{
			name:      "existing pricing",
			provider:  "openai",
			model:     "gpt-4",
			wantErr:   false,
			checkCost: true,
		},
		{
			name:     "non-existent pricing",
			provider: "unknown",
			model:    "unknown-model",
			wantErr:  true,
		},
		{
			name:     "non-existent model in existing provider",
			provider: "openai",
			model:    "unknown-model",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pricing, err := calc.GetPricing(tt.provider, tt.model)

			if (err != nil) != tt.wantErr {
				t.Errorf("GetPricing() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && pricing == nil {
				t.Error("Expected pricing, got nil")
			}

			if tt.checkCost && pricing != nil {
				if pricing.InputPer1K <= 0 {
					t.Error("Expected positive InputPer1K")
				}
				// Verify per-1K conversion
				expectedInput := pricing.InputPer1M / 1000.0
				if pricing.InputPer1K != expectedInput {
					t.Errorf("InputPer1K = %f, want %f", pricing.InputPer1K, expectedInput)
				}
			}
		})
	}
}

func TestCalculateCompletion(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	tests := []struct {
		name     string
		response CompletionResponse
		wantCost float64
		wantErr  bool
	}{
		{
			name: "gpt-4 completion",
			response: &mockCompletionResponse{
				model: "openai/gpt-4",
				usage: &mockUsage{
					promptTokens:     1000,
					completionTokens: 500,
					totalTokens:      1500,
				},
			},
			// (1000 / 1,000,000) * 30.00 + (500 / 1,000,000) * 60.00 = 0.03 + 0.03 = 0.06
			wantCost: 0.06,
			wantErr:  false,
		},
		{
			name: "gpt-3.5-turbo completion",
			response: &mockCompletionResponse{
				model: "openai/gpt-3.5-turbo",
				usage: &mockUsage{
					promptTokens:     2000,
					completionTokens: 1000,
					totalTokens:      3000,
				},
			},
			// (2000 / 1,000,000) * 0.50 + (1000 / 1,000,000) * 1.50 = 0.001 + 0.0015 = 0.0025
			wantCost: 0.0025,
			wantErr:  false,
		},
		{
			name: "model without provider prefix defaults to openai",
			response: &mockCompletionResponse{
				model: "gpt-4",
				usage: &mockUsage{
					promptTokens:     1000,
					completionTokens: 500,
					totalTokens:      1500,
				},
			},
			wantCost: 0.06,
			wantErr:  false,
		},
		{
			name:     "nil response",
			response: nil,
			wantErr:  true,
		},
		{
			name: "missing usage",
			response: &mockCompletionResponse{
				model: "openai/gpt-4",
				usage: nil,
			},
			wantErr: true,
		},
		{
			name: "unknown model",
			response: &mockCompletionResponse{
				model: "unknown/model",
				usage: &mockUsage{
					promptTokens:     1000,
					completionTokens: 500,
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cost, err := calc.CalculateCompletion(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("CalculateCompletion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Use tolerance for floating point comparison
				tolerance := 0.0000001
				if cost < tt.wantCost-tolerance || cost > tt.wantCost+tolerance {
					t.Errorf("CalculateCompletion() cost = %f, want %f", cost, tt.wantCost)
				}
			}
		})
	}
}

func TestCalculateEmbedding(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	tests := []struct {
		name     string
		response EmbeddingResponse
		wantCost float64
		wantErr  bool
	}{
		{
			name: "ada-002 embedding",
			response: &mockEmbeddingResponse{
				model: "openai/text-embedding-ada-002",
				usage: &mockUsage{
					promptTokens: 1000,
					totalTokens:  1000,
				},
			},
			// (1000 / 1,000,000) * 0.10 = 0.0001
			wantCost: 0.0001,
			wantErr:  false,
		},
		{
			name: "embedding-3-small",
			response: &mockEmbeddingResponse{
				model: "openai/text-embedding-3-small",
				usage: &mockUsage{
					promptTokens: 5000,
					totalTokens:  5000,
				},
			},
			// (5000 / 1,000,000) * 0.02 = 0.0001
			wantCost: 0.0001,
			wantErr:  false,
		},
		{
			name:     "nil response",
			response: nil,
			wantErr:  true,
		},
		{
			name: "missing usage",
			response: &mockEmbeddingResponse{
				model: "openai/text-embedding-ada-002",
				usage: nil,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cost, err := calc.CalculateEmbedding(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("CalculateEmbedding() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Use tolerance for floating point comparison
				tolerance := 0.0000001
				if cost < tt.wantCost-tolerance || cost > tt.wantCost+tolerance {
					t.Errorf("CalculateEmbedding() cost = %f, want %f", cost, tt.wantCost)
				}
			}
		})
	}
}

func TestEstimateCost(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	tests := []struct {
		name         string
		provider     string
		model        string
		inputTokens  int
		outputTokens int
		wantCost     float64
		wantErr      bool
	}{
		{
			name:         "gpt-4 estimate",
			provider:     "openai",
			model:        "gpt-4",
			inputTokens:  1000,
			outputTokens: 500,
			wantCost:     0.06,
			wantErr:      false,
		},
		{
			name:         "zero tokens",
			provider:     "openai",
			model:        "gpt-4",
			inputTokens:  0,
			outputTokens: 0,
			wantCost:     0.0,
			wantErr:      false,
		},
		{
			name:         "unknown model",
			provider:     "unknown",
			model:        "model",
			inputTokens:  1000,
			outputTokens: 500,
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cost, err := calc.EstimateCost(tt.provider, tt.model, tt.inputTokens, tt.outputTokens)

			if (err != nil) != tt.wantErr {
				t.Errorf("EstimateCost() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				// Use tolerance for floating point comparison
				tolerance := 0.0000001
				if cost < tt.wantCost-tolerance || cost > tt.wantCost+tolerance {
					t.Errorf("EstimateCost() cost = %f, want %f", cost, tt.wantCost)
				}
			}
		})
	}
}

func TestParseModel(t *testing.T) {
	tests := []struct {
		name         string
		modelStr     string
		wantProvider string
		wantModel    string
	}{
		{
			name:         "provider/model format",
			modelStr:     "openai/gpt-4",
			wantProvider: "openai",
			wantModel:    "gpt-4",
		},
		{
			name:         "anthropic model",
			modelStr:     "anthropic/claude-3-opus-20240229",
			wantProvider: "anthropic",
			wantModel:    "claude-3-opus-20240229",
		},
		{
			name:         "model without provider",
			modelStr:     "gpt-4",
			wantProvider: "openai",
			wantModel:    "gpt-4",
		},
		{
			name:         "model with multiple slashes",
			modelStr:     "provider/model/version",
			wantProvider: "provider",
			wantModel:    "model/version",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, model := parseModel(tt.modelStr)

			if provider != tt.wantProvider {
				t.Errorf("parseModel() provider = %s, want %s", provider, tt.wantProvider)
			}
			if model != tt.wantModel {
				t.Errorf("parseModel() model = %s, want %s", model, tt.wantModel)
			}
		})
	}
}

func TestCalculatorConcurrency(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	// Test concurrent AddPricing and GetPricing
	var wg sync.WaitGroup
	iterations := 100

	// Concurrent adds (using AddPricing for backward compat test)
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			calc.AddPricing(&Pricing{
				Provider:    "test",
				Model:       fmt.Sprintf("model-%d", idx),
				InputPer1M:  float64(idx),
				OutputPer1M: float64(idx * 2),
			})
		}(i)
	}

	// Concurrent gets
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = calc.GetModelInfo("openai", "gpt-4")
		}()
	}

	// Concurrent calculations
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp := &mockCompletionResponse{
				model: "openai/gpt-4",
				usage: &mockUsage{
					promptTokens:     1000,
					completionTokens: 500,
				},
			}
			_, _ = calc.CalculateCompletion(resp)
		}()
	}

	wg.Wait()

	// Verify calculator still works after concurrent operations
	info, err := calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Errorf("Calculator corrupted after concurrent operations: %v", err)
	}
	if info == nil {
		t.Error("Expected model info, got nil")
	}
}

func TestCalculatorCacheExpiration(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	// Set short TTL for testing
	calc.cacheTTL = 1 * time.Millisecond

	// Get model info (should cache it)
	info1, err := calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Fatalf("GetModelInfo failed: %v", err)
	}
	if info1 == nil {
		t.Fatal("Expected model info, got nil")
	}

	// Verify it's in cache
	calc.mu.RLock()
	cached, exists := calc.cache["openai/gpt-4"]
	calc.mu.RUnlock()
	if !exists {
		t.Error("Expected model info to be cached")
	}

	// Wait for cache to expire
	time.Sleep(10 * time.Millisecond)

	// Get again (should query provider again)
	info2, err := calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Fatalf("GetModelInfo failed after cache expiry: %v", err)
	}
	if info2 == nil {
		t.Fatal("Expected model info after cache expiry, got nil")
	}

	// Verify cache was updated
	calc.mu.RLock()
	cached2, exists := calc.cache["openai/gpt-4"]
	calc.mu.RUnlock()
	if !exists {
		t.Error("Expected model info to be re-cached")
	}
	if cached2.ExpiresAt.Before(cached.ExpiresAt) {
		t.Error("Expected new cache entry to have later expiration")
	}
}

func TestCalculatorClearCache(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	// Get model info (should cache it)
	_, err := calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Fatalf("GetModelInfo failed: %v", err)
	}

	// Verify it's in cache
	calc.mu.RLock()
	cacheCount := len(calc.cache)
	calc.mu.RUnlock()
	if cacheCount == 0 {
		t.Error("Expected cache to have entries")
	}

	// Clear cache
	calc.ClearCache()

	// Verify cache is empty
	calc.mu.RLock()
	cacheCount = len(calc.cache)
	calc.mu.RUnlock()
	if cacheCount != 0 {
		t.Errorf("Expected empty cache, got %d entries", cacheCount)
	}

	// Verify we can still get model info
	info, err := calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Fatalf("GetModelInfo failed after cache clear: %v", err)
	}
	if info == nil {
		t.Fatal("Expected model info after cache clear, got nil")
	}
}

// TestGetModelInfo tests the GetModelInfo method with overrides.
func TestGetModelInfo(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	tests := []struct {
		name     string
		provider string
		model    string
		wantErr  bool
	}{
		{
			name:     "existing model",
			provider: "openai",
			model:    "gpt-4",
			wantErr:  false,
		},
		{
			name:     "unknown provider",
			provider: "unknown",
			model:    "model",
			wantErr:  true,
		},
		{
			name:     "unknown model in existing provider",
			provider: "openai",
			model:    "unknown-model",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, err := calc.GetModelInfo(tt.provider, tt.model)

			if (err != nil) != tt.wantErr {
				t.Errorf("GetModelInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if info == nil {
					t.Error("Expected model info, got nil")
					return
				}
				if info.Name != tt.model {
					t.Errorf("Expected model name %s, got %s", tt.model, info.Name)
				}
				if info.Provider != tt.provider {
					t.Errorf("Expected provider %s, got %s", tt.provider, info.Provider)
				}
			}
		})
	}
}

// TestAddPricingOverride tests the AddPricingOverride method.
func TestAddPricingOverride(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	// Add override
	override := &types.ModelInfo{
		Name:            "custom-model",
		Provider:        "custom",
		InputCostPer1M:  100.0,
		OutputCostPer1M: 200.0,
	}
	calc.AddPricingOverride("custom", "custom-model", override)

	// Get override
	info, err := calc.GetModelInfo("custom", "custom-model")
	if err != nil {
		t.Fatalf("GetModelInfo failed: %v", err)
	}

	if info.InputCostPer1M != 100.0 {
		t.Errorf("InputCostPer1M = %f, want 100.0", info.InputCostPer1M)
	}
	if info.OutputCostPer1M != 200.0 {
		t.Errorf("OutputCostPer1M = %f, want 200.0", info.OutputCostPer1M)
	}

	// Verify override takes precedence
	existingOverride := &types.ModelInfo{
		Name:            "gpt-4",
		Provider:        "openai",
		InputCostPer1M:  999.0,
		OutputCostPer1M: 999.0,
	}
	calc.AddPricingOverride("openai", "gpt-4", existingOverride)

	info, err = calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Fatalf("GetModelInfo failed: %v", err)
	}

	if info.InputCostPer1M != 999.0 {
		t.Errorf("Override not applied: InputCostPer1M = %f, want 999.0", info.InputCostPer1M)
	}
}

// TestCalculatorConcurrentCacheAccess tests concurrent access to the cache.
func TestCalculatorConcurrentCacheAccess(t *testing.T) {
	registry := newMockRegistry()
	calc := NewCalculator(registry)

	var wg sync.WaitGroup
	iterations := 100

	// Concurrent reads (should populate cache)
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = calc.GetModelInfo("openai", "gpt-4")
		}()
	}

	// Concurrent writes (overrides)
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			calc.AddPricingOverride("test", fmt.Sprintf("model-%d", idx), &types.ModelInfo{
				Name:            fmt.Sprintf("model-%d", idx),
				Provider:        "test",
				InputCostPer1M:  float64(idx),
				OutputCostPer1M: float64(idx * 2),
			})
		}(i)
	}

	// Concurrent cache clears
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			calc.ClearCache()
		}()
	}

	wg.Wait()

	// Verify calculator still works
	info, err := calc.GetModelInfo("openai", "gpt-4")
	if err != nil {
		t.Errorf("Calculator corrupted after concurrent operations: %v", err)
	}
	if info == nil {
		t.Error("Expected model info, got nil")
	}
}
