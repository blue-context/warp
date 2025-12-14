package vllm

import (
	"sort"

	"github.com/blue-context/warp/types"
)

// modelRegistry contains vLLM model metadata.
// vLLM is self-hosted, so pricing is $0. Users can run any model locally.
// This registry contains common models that users typically deploy with vLLM.
var modelRegistry = map[string]*types.ModelInfo{
	// Llama Models
	"meta-llama/Llama-2-7b-hf": {
		Name:              "meta-llama/Llama-2-7b-hf",
		Provider:          "vllm",
		ContextWindow:     4096,
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
	"meta-llama/Llama-2-13b-hf": {
		Name:              "meta-llama/Llama-2-13b-hf",
		Provider:          "vllm",
		ContextWindow:     4096,
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
	"meta-llama/Meta-Llama-3-8B-Instruct": {
		Name:              "meta-llama/Meta-Llama-3-8B-Instruct",
		Provider:          "vllm",
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
	"meta-llama/Meta-Llama-3.1-8B-Instruct": {
		Name:              "meta-llama/Meta-Llama-3.1-8B-Instruct",
		Provider:          "vllm",
		ContextWindow:     128000,
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

	// Mistral Models
	"mistralai/Mistral-7B-v0.1": {
		Name:              "mistralai/Mistral-7B-v0.1",
		Provider:          "vllm",
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
	"mistralai/Mistral-7B-Instruct-v0.2": {
		Name:              "mistralai/Mistral-7B-Instruct-v0.2",
		Provider:          "vllm",
		ContextWindow:     32768,
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

	// NousResearch Models
	"NousResearch/Meta-Llama-3-8B-Instruct": {
		Name:              "NousResearch/Meta-Llama-3-8B-Instruct",
		Provider:          "vllm",
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
	"codellama/CodeLlama-7b-hf": {
		Name:              "codellama/CodeLlama-7b-hf",
		Provider:          "vllm",
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

	// Embedding Models (for /pooling endpoint)
	"intfloat/e5-small": {
		Name:              "intfloat/e5-small",
		Provider:          "vllm",
		ContextWindow:     512,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},
	"intfloat/e5-base": {
		Name:              "intfloat/e5-base",
		Provider:          "vllm",
		ContextWindow:     512,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Embedding: true,
		},
	},

	// Reranking Models (for /rerank endpoint)
	"BAAI/bge-reranker-large": {
		Name:              "BAAI/bge-reranker-large",
		Provider:          "vllm",
		ContextWindow:     512,
		MaxOutputTokens:   0,
		InputCostPer1M:    0.00,
		OutputCostPer1M:   0.00,
		SupportsVision:    false,
		SupportsFunctions: false,
		SupportsJSON:      false,
		SupportsStreaming: false,
		Capabilities: types.Capabilities{
			Rerank: true,
		},
	},
}

// GetModelInfo returns metadata for a specific model.
//
// Returns nil if the model is unknown. For vLLM, since it's self-hosted,
// users can run any model. This registry only contains common models.
func (p *Provider) GetModelInfo(model string) *types.ModelInfo {
	info, exists := modelRegistry[model]
	if exists {
		return info
	}

	// For unknown vLLM models, return a default with $0 cost
	// Users can customize this via overrides if needed
	return &types.ModelInfo{
		Name:              model,
		Provider:          "vllm",
		ContextWindow:     4096, // Conservative default
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
	}
}

// ListModels returns all supported vLLM models.
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
