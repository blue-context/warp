package azure

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains all Azure OpenAI model metadata.
// Azure uses the same pricing as OpenAI but with different model naming.
// This is the single source of truth for Azure models.
var modelRegistry = map[string]*types.ModelInfo{
	// GPT-4 Models
	"gpt-4": {
		Name:              "gpt-4",
		Provider:          "azure",
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
		Provider:          "azure",
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

	// GPT-3.5 Models (Azure uses gpt-35-turbo naming)
	"gpt-35-turbo": {
		Name:              "gpt-35-turbo",
		Provider:          "azure",
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
	"gpt-35-turbo-16k": {
		Name:              "gpt-35-turbo-16k",
		Provider:          "azure",
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
		Provider:          "azure",
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
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown to Azure.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	return modelRegistry[model]
}

// ListModels returns all supported Azure models.
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
