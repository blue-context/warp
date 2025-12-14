package cohere

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains Cohere model metadata.
// This is the single source of truth for Cohere models.
var modelRegistry = map[string]*types.ModelInfo{
	// Command Models
	"command-r-plus": {
		Name:              "command-r-plus",
		Provider:          "cohere",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   15.00,
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
	"command-r": {
		Name:              "command-r",
		Provider:          "cohere",
		ContextWindow:     128000,
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
	"command": {
		Name:              "command",
		Provider:          "cohere",
		ContextWindow:     4096,
		MaxOutputTokens:   4096,
		InputCostPer1M:    1.00,
		OutputCostPer1M:   2.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
		},
	},
	"command-light": {
		Name:              "command-light",
		Provider:          "cohere",
		ContextWindow:     4096,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.30,
		OutputCostPer1M:   0.60,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
		},
	},

	// Embedding Models
	"embed-english-v3.0": {
		Name:              "embed-english-v3.0",
		Provider:          "cohere",
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
	"embed-multilingual-v3.0": {
		Name:              "embed-multilingual-v3.0",
		Provider:          "cohere",
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
// Returns nil if the model is unknown to Cohere.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	return modelRegistry[model]
}

// ListModels returns all supported Cohere models.
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
