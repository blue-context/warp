// Package types contains shared type definitions used across packages to avoid import cycles.
package types

// ModelInfo contains information about a specific model.
//
// This is used for model metadata, cost calculation, and capability discovery.
//
// Thread Safety: ModelInfo is safe for concurrent reads after creation.
// It should be treated as immutable once created.
type ModelInfo struct {
	Name            string       // Model name (e.g., "gpt-4", "claude-3-opus")
	Provider        string       // Provider name (e.g., "openai", "anthropic")
	ContextWindow   int          // Maximum context window size
	MaxOutputTokens int          // Maximum output tokens (0 if not specified)
	InputCostPer1M  float64      // Cost per 1M input tokens (USD)
	OutputCostPer1M float64      // Cost per 1M output tokens (USD)
	Capabilities    Capabilities // Supported features for this model

	// Additional metadata
	SupportsVision    bool // Vision/multimodal support (redundant with Capabilities.Vision)
	SupportsFunctions bool // Function calling support (redundant with Capabilities.FunctionCalling)
	SupportsJSON      bool // JSON mode support (redundant with Capabilities.JSON)
	SupportsStreaming bool // Streaming support (redundant with Capabilities.Streaming)

	Deprecated bool   // Model is deprecated
	ReplacedBy string // Replacement model if deprecated
}

// Capabilities defines what operations a provider supports.
//
// Use this to check feature availability before making requests.
// Different providers support different features.
//
// Thread Safety: Capabilities is safe for concurrent reads after creation.
// It should be treated as immutable once returned from Supports().
type Capabilities struct {
	Completion      bool // Chat completion support
	Streaming       bool // Streaming completion support
	Embedding       bool // Embedding support
	ImageGeneration bool // Image generation support (DALL-E, etc.)
	ImageEdit       bool // Image editing support (DALL-E 2, etc.)
	ImageVariation  bool // Image variation support (DALL-E 2, etc.)
	Transcription   bool // Audio transcription support (Whisper, etc.)
	Speech          bool // Text-to-speech support
	Moderation      bool // Content moderation support
	FunctionCalling bool // Function/tool calling support
	Vision          bool // Vision/multimodal support
	JSON            bool // JSON mode support
	Rerank          bool // Document reranking support
}
