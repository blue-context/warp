package token

// ProviderCounter creates a counter optimized for a specific provider.
//
// Currently returns the default approximation counter for all providers.
// In the future, this can be extended to support provider-specific optimizations:
// - OpenAI: Use tiktoken-compatible algorithms
// - Anthropic: Use Claude-specific tokenization patterns
// - Azure: Use OpenAI algorithms with Azure-specific adjustments
//
// Thread Safety: The returned Counter is safe for concurrent use.
//
// Example:
//
//	counter := token.ProviderCounter("openai")
//	tokens := counter.CountText("Hello, world!")
func ProviderCounter(provider string) Counter {
	// Future provider-specific implementations can be added here
	switch provider {
	case "openai", "azure":
		// OpenAI and Azure use similar tokenization
		return NewCounter()

	case "anthropic":
		// Anthropic Claude uses different tokenization
		return NewCounter()

	case "google", "vertex_ai":
		// Google models use SentencePiece tokenization
		return NewCounter()

	case "aws", "bedrock":
		// AWS Bedrock varies by model
		return NewCounter()

	default:
		// Default approximation for unknown providers
		return NewCounter()
	}
}
