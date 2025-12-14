package vertex

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains Google Vertex AI model metadata.
// This is the single source of truth for Vertex models.
var modelRegistry = map[string]*types.ModelInfo{
	// Gemini 1.5 Models
	"gemini-1.5-pro": {
		Name:              "gemini-1.5-pro",
		Provider:          "vertex",
		ContextWindow:     1048576, // 1M tokens
		MaxOutputTokens:   8192,
		InputCostPer1M:    1.25,
		OutputCostPer1M:   5.00,
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
	"gemini-1.5-flash": {
		Name:              "gemini-1.5-flash",
		Provider:          "vertex",
		ContextWindow:     1048576,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.075,
		OutputCostPer1M:   0.30,
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

	// Gemini 1.0 Models
	"gemini-pro": {
		Name:              "gemini-pro",
		Provider:          "vertex",
		ContextWindow:     32760,
		MaxOutputTokens:   8192,
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
	"gemini-pro-vision": {
		Name:              "gemini-pro-vision",
		Provider:          "vertex",
		ContextWindow:     16384,
		MaxOutputTokens:   2048,
		InputCostPer1M:    0.50,
		OutputCostPer1M:   1.50,
		SupportsVision:    true,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: true,
		Capabilities: types.Capabilities{
			Completion: true,
			Streaming:  true,
			Vision:     true,
		},
	},
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown to Vertex AI.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	return modelRegistry[model]
}

// ListModels returns all supported Vertex AI models.
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
