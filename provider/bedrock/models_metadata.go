package bedrock

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelMetadataRegistry contains AWS Bedrock model pricing and capability metadata.
// This is the single source of truth for Bedrock model metadata.
var modelMetadataRegistry = map[string]*types.ModelInfo{
	// Anthropic Claude via Bedrock
	"anthropic.claude-3-5-sonnet-20241022-v2:0": {
		Name:              "anthropic.claude-3-5-sonnet-20241022-v2:0",
		Provider:          "bedrock",
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
	"anthropic.claude-3-opus-20240229-v1:0": {
		Name:              "anthropic.claude-3-opus-20240229-v1:0",
		Provider:          "bedrock",
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
	"anthropic.claude-3-sonnet-20240229-v1:0": {
		Name:              "anthropic.claude-3-sonnet-20240229-v1:0",
		Provider:          "bedrock",
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
	"anthropic.claude-3-haiku-20240307-v1:0": {
		Name:              "anthropic.claude-3-haiku-20240307-v1:0",
		Provider:          "bedrock",
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
// Returns nil if the model is unknown to Bedrock.
// For short model names (e.g., "claude-3-opus"), converts to full Bedrock ID first.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	// Try to convert short name to full Bedrock ID
	fullModelID, err := getModelID(model)
	if err != nil {
		return nil
	}

	return modelMetadataRegistry[fullModelID]
}

// ListModels returns all supported Bedrock models.
//
// Returns a slice of ModelInfo sorted alphabetically by model name.
func (p *Provider) ListModels() []*types.ModelInfo {
	models := make([]*types.ModelInfo, 0, len(modelMetadataRegistry))
	for _, info := range modelMetadataRegistry {
		models = append(models, info)
	}

	// Sort by name for consistent output
	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	return models
}
