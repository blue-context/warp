package openrouter

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains popular OpenRouter model metadata.
//
// OpenRouter provides access to 300+ models from multiple providers.
// This registry contains metadata for the most popular models including
// pricing, capabilities, and context windows.
//
// Note: Model names in OpenRouter use the format "provider/model-name".
var modelRegistry = map[string]*types.ModelInfo{
	// OpenAI Models
	"openai/gpt-4o": {
		Name:              "openai/gpt-4o",
		Provider:          "openrouter",
		ContextWindow:     128000,
		MaxOutputTokens:   16384,
		InputCostPer1M:    5.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},
	"openai/gpt-4-turbo": {
		Name:              "openai/gpt-4-turbo",
		Provider:          "openrouter",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    10.00,
		OutputCostPer1M:   30.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},
	"openai/gpt-3.5-turbo": {
		Name:              "openai/gpt-3.5-turbo",
		Provider:          "openrouter",
		ContextWindow:     16385,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.50,
		OutputCostPer1M:   1.50,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},

	// Anthropic Models
	"anthropic/claude-opus-4": {
		Name:              "anthropic/claude-opus-4",
		Provider:          "openrouter",
		ContextWindow:     200000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    15.00,
		OutputCostPer1M:   75.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},
	"anthropic/claude-sonnet-4": {
		Name:              "anthropic/claude-sonnet-4",
		Provider:          "openrouter",
		ContextWindow:     200000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},
	"anthropic/claude-3.5-sonnet": {
		Name:              "anthropic/claude-3.5-sonnet",
		Provider:          "openrouter",
		ContextWindow:     200000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},

	// Google Models
	"google/gemini-2.0-flash-exp": {
		Name:              "google/gemini-2.0-flash-exp",
		Provider:          "openrouter",
		ContextWindow:     1000000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.00, // Free tier
		OutputCostPer1M:   0.00, // Free tier
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},
	"google/gemini-pro": {
		Name:              "google/gemini-pro",
		Provider:          "openrouter",
		ContextWindow:     32768,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.125,
		OutputCostPer1M:   0.375,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},

	// Meta Models
	"meta-llama/llama-3-70b-instruct": {
		Name:              "meta-llama/llama-3-70b-instruct",
		Provider:          "openrouter",
		ContextWindow:     8192,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.59,
		OutputCostPer1M:   0.79,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},
	"meta-llama/llama-3.1-8b-instruct": {
		Name:              "meta-llama/llama-3.1-8b-instruct",
		Provider:          "openrouter",
		ContextWindow:     131072,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.06,
		OutputCostPer1M:   0.06,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},

	// Mistral Models
	"mistralai/mistral-nemo": {
		Name:              "mistralai/mistral-nemo",
		Provider:          "openrouter",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.15,
		OutputCostPer1M:   0.15,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},
	"mistralai/mistral-large": {
		Name:              "mistralai/mistral-large",
		Provider:          "openrouter",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   9.00,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},

	// DeepSeek Models
	"deepseek/deepseek-r1": {
		Name:              "deepseek/deepseek-r1",
		Provider:          "openrouter",
		ContextWindow:     64000,
		MaxOutputTokens:   8000,
		InputCostPer1M:    2.19,
		OutputCostPer1M:   8.19,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
			JSON:       true,
		},
	},
	"deepseek/deepseek-chat": {
		Name:              "deepseek/deepseek-chat",
		Provider:          "openrouter",
		ContextWindow:     64000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.14,
		OutputCostPer1M:   0.28,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},

	// xAI Models
	"x-ai/grok-4.1-fast": {
		Name:              "x-ai/grok-4.1-fast",
		Provider:          "openrouter",
		ContextWindow:     131072,
		MaxOutputTokens:   4096,
		InputCostPer1M:    5.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},

	// Cohere Models
	"cohere/command-r-plus": {
		Name:              "cohere/command-r-plus",
		Provider:          "openrouter",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    2.50,
		OutputCostPer1M:   10.00,
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			JSON:            true,
		},
	},

	// Perplexity Models
	"perplexity/llama-3.1-sonar-large-128k-online": {
		Name:              "perplexity/llama-3.1-sonar-large-128k-online",
		Provider:          "openrouter",
		ContextWindow:     127072,
		MaxOutputTokens:   4096,
		InputCostPer1M:    1.00,
		OutputCostPer1M:   1.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
			JSON:       true,
		},
	},

	// OpenRouter Auto Router
	"openrouter/auto": {
		Name:              "openrouter/auto",
		Provider:          "openrouter",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00, // Dynamic pricing
		OutputCostPer1M:   0.00, // Dynamic pricing
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
			JSON:            true,
		},
	},

	// Embedding Models - OpenAI
	"openai/text-embedding-ada-002": {
		Name:              "openai/text-embedding-ada-002",
		Provider:          "openrouter",
		ContextWindow:     8191,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.10,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},
	"openai/text-embedding-3-small": {
		Name:              "openai/text-embedding-3-small",
		Provider:          "openrouter",
		ContextWindow:     8191,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.02,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},
	"openai/text-embedding-3-large": {
		Name:              "openai/text-embedding-3-large",
		Provider:          "openrouter",
		ContextWindow:     8191,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.13,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},

	// Embedding Models - Sentence Transformers
	"sentence-transformers/all-mpnet-base-v2": {
		Name:              "sentence-transformers/all-mpnet-base-v2",
		Provider:          "openrouter",
		ContextWindow:     512,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.01,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},
	"sentence-transformers/all-minilm-l6-v2": {
		Name:              "sentence-transformers/all-minilm-l6-v2",
		Provider:          "openrouter",
		ContextWindow:     512,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.01,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},

	// Embedding Models - Cohere
	"cohere/embed-english-v3.0": {
		Name:              "cohere/embed-english-v3.0",
		Provider:          "openrouter",
		ContextWindow:     512,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.10,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown to this provider.
// This method is used by the cost calculator to retrieve pricing and capability information.
//
// Thread Safety: Safe for concurrent use.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	info, exists := modelRegistry[model]
	if exists {
		return info
	}

	// Unknown model
	return nil
}

// ListModels returns all supported models.
//
// Returns a slice of ModelInfo sorted alphabetically by model name.
// This includes chat models, embedding models, and the auto router.
//
// Thread Safety: Safe for concurrent use.
func (p *Provider) ListModels() []*types.ModelInfo {
	models := make([]*types.ModelInfo, 0, len(modelRegistry))
	for _, info := range modelRegistry {
		models = append(models, info)
	}

	// Sort by name for consistent output
	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	return models
}
