package groq

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains Groq model metadata.
// This is the single source of truth for Groq models.
var modelRegistry = map[string]*types.ModelInfo{
	// Llama 3 Models
	"llama-3.1-70b-versatile": {
		Name:              "llama-3.1-70b-versatile",
		Provider:          "groq",
		ContextWindow:     128000,
		MaxOutputTokens:   8192,
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
	"llama-3.1-8b-instant": {
		Name:              "llama-3.1-8b-instant",
		Provider:          "groq",
		ContextWindow:     128000,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.05,
		OutputCostPer1M:   0.08,
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
	"llama3-70b-8192": {
		Name:              "llama3-70b-8192",
		Provider:          "groq",
		ContextWindow:     8192,
		MaxOutputTokens:   8192,
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
	"llama3-8b-8192": {
		Name:              "llama3-8b-8192",
		Provider:          "groq",
		ContextWindow:     8192,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.05,
		OutputCostPer1M:   0.08,
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
	"mixtral-8x7b-32768": {
		Name:              "mixtral-8x7b-32768",
		Provider:          "groq",
		ContextWindow:     32768,
		MaxOutputTokens:   32768,
		InputCostPer1M:    0.24,
		OutputCostPer1M:   0.24,
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

	// Gemma Models
	"gemma-7b-it": {
		Name:              "gemma-7b-it",
		Provider:          "groq",
		ContextWindow:     8192,
		MaxOutputTokens:   8192,
		InputCostPer1M:    0.07,
		OutputCostPer1M:   0.07,
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
// Returns nil if the model is unknown to Groq.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	return modelRegistry[model]
}

// ListModels returns all supported Groq models.
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
