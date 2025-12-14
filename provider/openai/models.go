package openai

import (
	"sort"
	"strings"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains all OpenAI model metadata.
// This is the single source of truth for OpenAI models.
var modelRegistry = map[string]*types.ModelInfo{
	// GPT-4 Models
	"gpt-4": {
		Name:              "gpt-4",
		Provider:          "openai",
		ContextWindow:     8192,
		MaxOutputTokens:   4096,
		InputCostPer1M:    30.00,
		OutputCostPer1M:   60.00,
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
	"gpt-4-turbo": {
		Name:              "gpt-4-turbo",
		Provider:          "openai",
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
	"gpt-4-turbo-preview": {
		Name:              "gpt-4-turbo-preview",
		Provider:          "openai",
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
	"gpt-4-0125-preview": {
		Name:              "gpt-4-0125-preview",
		Provider:          "openai",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    10.00,
		OutputCostPer1M:   30.00,
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
	"gpt-4-1106-preview": {
		Name:              "gpt-4-1106-preview",
		Provider:          "openai",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    10.00,
		OutputCostPer1M:   30.00,
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

	// GPT-3.5 Models
	"gpt-3.5-turbo": {
		Name:              "gpt-3.5-turbo",
		Provider:          "openai",
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
	"gpt-3.5-turbo-0125": {
		Name:              "gpt-3.5-turbo-0125",
		Provider:          "openai",
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
	"gpt-3.5-turbo-1106": {
		Name:              "gpt-3.5-turbo-1106",
		Provider:          "openai",
		ContextWindow:     16385,
		MaxOutputTokens:   4096,
		InputCostPer1M:    1.00,
		OutputCostPer1M:   2.00,
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
	"gpt-3.5-turbo-16k": {
		Name:              "gpt-3.5-turbo-16k",
		Provider:          "openai",
		ContextWindow:     16385,
		MaxOutputTokens:   4096,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   4.00,
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

	// Embedding Models
	"text-embedding-ada-002": {
		Name:              "text-embedding-ada-002",
		Provider:          "openai",
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
	"text-embedding-3-small": {
		Name:              "text-embedding-3-small",
		Provider:          "openai",
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
	"text-embedding-3-large": {
		Name:              "text-embedding-3-large",
		Provider:          "openai",
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
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown. Supports fine-tuned models by inferring
// from the base model (e.g., ft:gpt-3.5-turbo:suffix).
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	// Check direct registry lookup first
	info, exists := modelRegistry[model]
	if exists {
		return info
	}

	// Try to infer for fine-tuned models
	return inferModelInfo(model)
}

// ListModels returns all supported models.
//
// Returns a slice of ModelInfo sorted alphabetically by model name.
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

// inferModelInfo infers model info for dynamic models like fine-tuned models.
//
// Fine-tuned models follow the format: ft:base-model:org:name:id
// We infer pricing from the base model.
func inferModelInfo(model string) *types.ModelInfo {
	// Handle fine-tuned models: ft:gpt-3.5-turbo:suffix
	if strings.HasPrefix(model, "ft:") {
		parts := strings.SplitN(model, ":", 3)
		if len(parts) >= 2 {
			baseModel := parts[1]
			if baseInfo, exists := modelRegistry[baseModel]; exists {
				// Create a copy with the actual model name
				info := *baseInfo
				info.Name = model
				return &info
			}
		}
	}

	// Unknown model
	return nil
}
