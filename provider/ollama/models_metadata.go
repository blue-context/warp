package ollama

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains Ollama model metadata.
// Ollama is self-hosted, so pricing is $0. Users can run any model locally.
// This is the single source of truth for common Ollama models.
var modelRegistry = map[string]*types.ModelInfo{
	// Llama Models
	"llama3.1": {
		Name:              "llama3.1",
		Provider:          "ollama",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00, // Self-hosted, no cost
		OutputCostPer1M:   0.00,
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
	"llama3": {
		Name:              "llama3",
		Provider:          "ollama",
		ContextWindow:     8192,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
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
	"llama2": {
		Name:              "llama2",
		Provider:          "ollama",
		ContextWindow:     4096,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
		},
	},

	// Mistral Models
	"mistral": {
		Name:              "mistral",
		Provider:          "ollama",
		ContextWindow:     8192,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
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

	// Code Models
	"codellama": {
		Name:              "codellama",
		Provider:          "ollama",
		ContextWindow:     16384,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
		},
	},
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown. For Ollama, since it's self-hosted,
// users can run any model. This registry only contains common models.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	info, exists := modelRegistry[model]
	if exists {
		return info
	}

	// For unknown Ollama models, return a default with $0 cost
	// Users can customize this via overrides if needed
	return &types.ModelInfo{
		Name:              model,
		Provider:          "ollama",
		ContextWindow:     4096, // Conservative default
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
		},
	}
}

// ListModels returns all supported Ollama models.
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
