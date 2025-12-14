package vllmsemanticrouter

import (
	"github.com/blue-context/warp/types"
)

// GetModelInfo returns metadata for a specific model.
//
// vLLM Semantic Router supports the special "auto" model for semantic routing,
// as well as backend models that can be configured in the router deployment.
// Since backend models vary by deployment, we only provide information for
// the "auto" model.
//
// Returns nil if the model is unknown.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	if info, exists := modelMetadata[model]; exists {
		return &info
	}
	return nil
}

// ListModels returns all known models supported by this provider.
//
// Returns the special "auto" model which is the primary feature of
// vLLM Semantic Router. Backend models are deployment-specific and
// not included in this list.
func (p *Provider) ListModels() []*types.ModelInfo {
	models := make([]*types.ModelInfo, 0, len(modelMetadata))
	for _, info := range modelMetadata {
		infoCopy := info
		models = append(models, &infoCopy)
	}
	return models
}

// modelMetadata contains metadata for vLLM Semantic Router models.
//
// The primary model is "auto" which enables semantic routing to backend models.
// Backend-specific models are deployment-dependent and not included here.
var modelMetadata = map[string]types.ModelInfo{
	"auto": {
		Name:            "auto",
		Provider:        "vllm-semantic-router",
		ContextWindow:   0, // Depends on backend model selected by router
		MaxOutputTokens: 0, // Depends on backend model selected by router
		InputCostPer1M:  0, // Depends on backend model selected by router
		OutputCostPer1M: 0, // Depends on backend model selected by router
		Capabilities: types.Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          false,
			JSON:            true,
		},
		SupportsVision:    false,
		SupportsFunctions: true,
		SupportsJSON:      true,
		SupportsStreaming: true,
	},
}
