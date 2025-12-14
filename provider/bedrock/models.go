package bedrock

import (
	"fmt"
	"strings"
)

// modelIDMap maps Warp model names to AWS Bedrock model IDs.
//
// Bedrock uses fully qualified model IDs like "anthropic.claude-3-opus-20240229-v1:0"
// while Warp uses short names like "claude-3-opus".
//
// This map enables users to use familiar model names while automatically
// translating them to Bedrock's format.
var modelIDMap = map[string]string{
	// Anthropic Claude models
	"claude-3-opus":        "anthropic.claude-3-opus-20240229-v1:0",
	"claude-3-sonnet":      "anthropic.claude-3-sonnet-20240229-v1:0",
	"claude-3-haiku":       "anthropic.claude-3-haiku-20240307-v1:0",
	"claude-3-5-sonnet":    "anthropic.claude-3-5-sonnet-20241022-v2:0",
	"claude-3-5-sonnet-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",

	// Meta Llama models
	"llama3-70b":   "meta.llama3-70b-instruct-v1:0",
	"llama3-8b":    "meta.llama3-8b-instruct-v1:0",
	"llama3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
	"llama3-1-8b":  "meta.llama3-1-8b-instruct-v1:0",
	"llama3-2-1b":  "meta.llama3-2-1b-instruct-v1:0",
	"llama3-2-3b":  "meta.llama3-2-3b-instruct-v1:0",
	"llama3-2-11b": "meta.llama3-2-11b-vision-instruct-v1:0",
	"llama3-2-90b": "meta.llama3-2-90b-vision-instruct-v1:0",

	// Amazon Titan models
	"titan-text-express": "amazon.titan-text-express-v1",
	"titan-text-lite":    "amazon.titan-text-lite-v1",
	"titan-embed-text":   "amazon.titan-embed-text-v1",
	"titan-embed-image":  "amazon.titan-embed-image-v1",

	// Cohere models
	"command-text":       "cohere.command-text-v14",
	"command-light-text": "cohere.command-light-text-v14",
	"command-r":          "cohere.command-r-v1:0",
	"command-r-plus":     "cohere.command-r-plus-v1:0",

	// AI21 Labs Jurassic models
	"j2-ultra": "ai21.j2-ultra-v1",
	"j2-mid":   "ai21.j2-mid-v1",

	// Stability AI models
	"stable-diffusion-xl": "stability.stable-diffusion-xl-v1",
}

// modelFamily represents the family of models (claude, llama, titan, etc.).
type modelFamily string

const (
	familyClaude    modelFamily = "claude"
	familyLlama     modelFamily = "llama"
	familyTitan     modelFamily = "titan"
	familyCohere    modelFamily = "cohere"
	familyAI21      modelFamily = "ai21"
	familyStability modelFamily = "stability"
	familyUnknown   modelFamily = "unknown"
)

// getModelID returns the Bedrock model ID for a given model name.
//
// If the model name is in the map, returns the mapped Bedrock ID.
// If not in the map, assumes the input is already a full Bedrock model ID.
//
// Example:
//
//	id, _ := getModelID("claude-3-opus")
//	// Returns: "anthropic.claude-3-opus-20240229-v1:0"
//
//	id, _ := getModelID("anthropic.claude-3-opus-20240229-v1:0")
//	// Returns: "anthropic.claude-3-opus-20240229-v1:0" (unchanged)
func getModelID(model string) (string, error) {
	if model == "" {
		return "", fmt.Errorf("model name cannot be empty")
	}

	// Check if model is in the map
	if id, ok := modelIDMap[model]; ok {
		return id, nil
	}

	// If not in map, assume it's already a full Bedrock model ID
	// Validate it has the expected format (vendor.model-name)
	if strings.Contains(model, ".") {
		return model, nil
	}

	// Model not found and doesn't look like a Bedrock ID
	return "", fmt.Errorf("unknown model: %s (use format 'vendor.model-name' for custom models)", model)
}

// detectModelFamily determines the model family from a Bedrock model ID.
//
// This is used to route requests to the correct transformation and parsing logic,
// as different model families have different API formats.
//
// Example:
//
//	family := detectModelFamily("anthropic.claude-3-opus-20240229-v1:0")
//	// Returns: familyClaude
//
//	family := detectModelFamily("meta.llama3-70b-instruct-v1:0")
//	// Returns: familyLlama
func detectModelFamily(modelID string) modelFamily {
	// Extract vendor prefix (everything before first dot)
	parts := strings.SplitN(modelID, ".", 2)
	if len(parts) < 2 {
		return familyUnknown
	}

	vendor := parts[0]

	switch vendor {
	case "anthropic":
		return familyClaude
	case "meta":
		return familyLlama
	case "amazon":
		// Distinguish between Titan text and embed models
		if strings.Contains(modelID, "titan-text") {
			return familyTitan
		}
		// Embedding models are handled separately
		return familyTitan
	case "cohere":
		return familyCohere
	case "ai21":
		return familyAI21
	case "stability":
		return familyStability
	default:
		return familyUnknown
	}
}

// supportsStreaming returns true if the model family supports streaming.
//
// Currently only Claude models have streaming implemented via Bedrock.
// Other model families may support streaming in the future.
func supportsStreaming(family modelFamily) bool {
	switch family {
	case familyClaude:
		return true
	default:
		// Llama, Titan, Cohere streaming not yet implemented
		return false
	}
}

// supportsTools returns true if the model family supports function calling/tools.
//
// Only Claude models via Bedrock support tools currently.
func supportsTools(family modelFamily) bool {
	switch family {
	case familyClaude:
		return true
	default:
		return false
	}
}

// supportsVision returns true if the model family supports vision/multimodal input.
//
// Claude 3 models and some Llama 3.2 models support vision.
func supportsVision(family modelFamily) bool {
	switch family {
	case familyClaude:
		return true
	case familyLlama:
		// Only specific Llama 3.2 vision models
		return true
	default:
		return false
	}
}
