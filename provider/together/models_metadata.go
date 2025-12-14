package together

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains Together AI model metadata.
// This is the single source of truth for Together models.
var modelRegistry = map[string]*types.ModelInfo{
	// Llama Models
	"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
		Name:              "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
		Provider:          "together",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.88,
		OutputCostPer1M:   0.88,
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
	"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {
		Name:              "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
		Provider:          "together",
		ContextWindow:     128000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.18,
		OutputCostPer1M:   0.18,
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

	// Mixtral Models
	"mistralai/Mixtral-8x7B-Instruct-v0.1": {
		Name:              "mistralai/Mixtral-8x7B-Instruct-v0.1",
		Provider:          "together",
		ContextWindow:     32768,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.60,
		OutputCostPer1M:   0.60,
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
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown to Together AI.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	return modelRegistry[model]
}

// ListModels returns all supported Together AI models.
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
