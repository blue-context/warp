package anthropic

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains all Anthropic Claude model metadata.
// This is the single source of truth for Anthropic models.
var modelRegistry = map[string]*types.ModelInfo{
	// Claude 3.5 Models
	"claude-3-5-sonnet-20241022": {
		Name:              "claude-3-5-sonnet-20241022",
		Provider:          "anthropic",
		ContextWindow:     200000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
		},
	},
	"claude-3-5-sonnet-20240620": {
		Name:              "claude-3-5-sonnet-20240620",
		Provider:          "anthropic",
		ContextWindow:     200000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
		},
	},

	// Claude 3 Models
	"claude-3-opus-20240229": {
		Name:              "claude-3-opus-20240229",
		Provider:          "anthropic",
		ContextWindow:     200000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    15.00,
		OutputCostPer1M:   75.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
		},
	},
	"claude-3-sonnet-20240229": {
		Name:              "claude-3-sonnet-20240229",
		Provider:          "anthropic",
		ContextWindow:     200000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    3.00,
		OutputCostPer1M:   15.00,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
		},
	},
	"claude-3-haiku-20240307": {
		Name:              "claude-3-haiku-20240307",
		Provider:          "anthropic",
		ContextWindow:     200000,
		MaxOutputTokens:   4096,
		InputCostPer1M:    0.25,
		OutputCostPer1M:   1.25,
		SupportsVision:    true,
		SupportsFunctions: true,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
		},
	},
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown to Anthropic.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	return modelRegistry[model]
}

// ListModels returns all supported Anthropic models.
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
