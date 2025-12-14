// Package warp provides a unified Go SDK for calling 100+ LLM providers
// with a consistent OpenAI-compatible API format.
//
// The SDK supports OpenAI, Anthropic, Azure, AWS Bedrock, Google Vertex AI,
// and many other providers through a single interface.
//
// Basic usage:
//
//	package main
//
//	import (
//	    "context"
//	    "fmt"
//	    "log"
//	    "os"
//
//	    "github.com/blue-context/warp"
//	)
//
//	func main() {
//	    // Create client
//	    client, err := warp.NewClient(
//	        warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
//	    )
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    defer client.Close()
//
//	    // Send completion request
//	    resp, err := client.Completion(context.Background(), &warp.CompletionRequest{
//	        Model: "openai/gpt-4",
//	        Messages: []warp.Message{
//	            {Role: "user", Content: "Hello, world!"},
//	        },
//	    })
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//
//	    // Print response
//	    fmt.Println(resp.Choices[0].Message.Content)
//	}
package warp

import (
	"io"
	"time"
)

// CompletionRequest represents a chat completion request to any supported LLM provider.
// This request format follows OpenAI's API structure and is automatically translated
// for other providers (Anthropic, Azure, AWS Bedrock, Google Vertex AI, etc.).
//
// Note: There is no Stream field. Use Completion() for non-streaming
// or CompletionStream() for streaming responses. The method you call
// determines the behavior.
//
// Thread Safety: CompletionRequest is safe for concurrent reads after creation.
// The Metadata field should not be modified concurrently without external synchronization.
type CompletionRequest struct {
	// Model specifies the LLM model to use. Format: "provider/model-name"
	// Examples: "openai/gpt-4", "anthropic/claude-3-opus-20240229", "azure/gpt-4"
	Model string `json:"model"`

	// Messages contains the conversation history
	Messages []Message `json:"messages"`

	// Temperature controls randomness in the output (0.0 to 2.0).
	// Lower values make output more focused and deterministic.
	Temperature *float64 `json:"temperature,omitempty"`

	// MaxTokens specifies the maximum number of tokens to generate.
	MaxTokens *int `json:"max_tokens,omitempty"`

	// TopP controls nucleus sampling (0.0 to 1.0).
	// Alternative to temperature, for controlling diversity.
	TopP *float64 `json:"top_p,omitempty"`

	// FrequencyPenalty penalizes frequent tokens (-2.0 to 2.0).
	// Positive values decrease likelihood of repeating the same line.
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`

	// PresencePenalty penalizes tokens based on presence (-2.0 to 2.0).
	// Positive values increase likelihood of talking about new topics.
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`

	// Stop contains up to 4 sequences where the API will stop generating.
	Stop []string `json:"stop,omitempty"`

	// N specifies how many chat completion choices to generate.
	N *int `json:"n,omitempty"`

	// Tools defines available function calling tools for the model.
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice controls which tool (if any) is called by the model.
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// ResponseFormat specifies the format of the model's output.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Fallbacks contains fallback model names to try if the primary model fails.
	// Format: ["provider/model-name", ...]
	Fallbacks []string `json:"fallbacks,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// Message represents a single message in a conversation.
// Supports both simple text content and multimodal content (text + images).
//
// Thread Safety: Message is safe for concurrent reads after creation.
// The Content field (when storing multimodal data) should not be modified
// concurrently without external synchronization.
type Message struct {
	// Role identifies the message sender.
	// Valid values: "system", "user", "assistant", "tool"
	Role string `json:"role"`

	// Content can be either:
	// - string: simple text content
	// - []ContentPart: multimodal content (text and/or images)
	Content any `json:"content"`

	// Name is an optional name for the participant.
	// Used primarily for multi-user conversations.
	Name string `json:"name,omitempty"`

	// ToolCalls contains tool invocations made by the assistant.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// ToolCallID identifies which tool call this message is responding to.
	// Used when Role is "tool".
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ContentPart represents a component of multimodal message content.
// A message can contain multiple parts mixing text and images.
type ContentPart struct {
	// Type specifies the content type.
	// Valid values: "text", "image_url"
	Type string `json:"type"`

	// Text contains the text content (when Type is "text").
	Text string `json:"text,omitempty"`

	// ImageURL contains the image reference (when Type is "image_url").
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image reference in multimodal content.
type ImageURL struct {
	// URL is the image URL or base64-encoded data URI.
	// Formats:
	// - HTTP(S) URL: "https://example.com/image.jpg"
	// - Data URI: "data:image/jpeg;base64,..."
	URL string `json:"url"`

	// Detail specifies the image detail level for vision models.
	// Valid values: "auto" (default), "low", "high"
	Detail string `json:"detail,omitempty"`
}

// CompletionResponse represents a completion response from the LLM provider.
//
// Thread Safety: CompletionResponse is safe for concurrent reads.
// Provider-specific fields (ProviderFields, HiddenParams) should not be modified
// concurrently without external synchronization.
type CompletionResponse struct {
	// ID is a unique identifier for this completion.
	ID string `json:"id"`

	// Object is the object type (e.g., "chat.completion").
	Object string `json:"object"`

	// Created is the Unix timestamp (in seconds) of when the completion was created.
	Created int64 `json:"created"`

	// Model is the model used for this completion.
	Model string `json:"model"`

	// Choices contains the generated completion choices.
	Choices []Choice `json:"choices"`

	// Usage contains token usage information for this request.
	Usage *Usage `json:"usage,omitempty"`

	// SystemFingerprint is a unique identifier for the model configuration.
	// Can be used to understand model behavior changes.
	SystemFingerprint string `json:"system_fingerprint,omitempty"`

	// ProviderFields contains provider-specific response fields.
	ProviderFields map[string]any `json:"provider_specific_fields,omitempty"`

	// HiddenParams contains internal metadata (prefixed with _).
	// Used for debugging and tracking internal state.
	HiddenParams map[string]any `json:"_hidden_params,omitempty"`
}

// GetModel returns the model name.
func (r *CompletionResponse) GetModel() string {
	if r == nil {
		return ""
	}
	return r.Model
}

// GetUsageInfo returns the usage information as a generic interface.
// This satisfies the cost.CompletionResponse interface.
func (r *CompletionResponse) GetUsageInfo() interface{} {
	if r == nil {
		return nil
	}
	return r.Usage
}

// Choice represents a single completion choice in the response.
type Choice struct {
	// Index is the zero-based index of this choice in the Choices array.
	Index int `json:"index"`

	// Message contains the generated message.
	Message Message `json:"message"`

	// FinishReason explains why the model stopped generating.
	// Valid values: "stop", "length", "tool_calls", "content_filter"
	FinishReason string `json:"finish_reason"`

	// Logprobs contains log probability information for generated tokens.
	Logprobs *Logprobs `json:"logprobs,omitempty"`
}

// Logprobs represents log probability information for generated tokens.
// This is useful for understanding model confidence and alternative choices.
type Logprobs struct {
	// Content contains log probability information for each token.
	Content []TokenLogprob `json:"content,omitempty"`
}

// TokenLogprob represents log probability information for a single token.
type TokenLogprob struct {
	// Token is the text representation of the token.
	Token string `json:"token"`

	// Logprob is the log probability of this token.
	Logprob float64 `json:"logprob"`

	// Bytes is the byte representation of the token (if applicable).
	Bytes []byte `json:"bytes,omitempty"`
}

// Usage represents token usage statistics for a request.
type Usage struct {
	// PromptTokens is the number of tokens in the prompt.
	PromptTokens int `json:"prompt_tokens"`

	// CompletionTokens is the number of tokens in the generated completion.
	CompletionTokens int `json:"completion_tokens"`

	// TotalTokens is the total number of tokens (prompt + completion).
	TotalTokens int `json:"total_tokens"`

	// PromptDetails provides detailed breakdown of prompt tokens.
	PromptDetails *PromptTokensDetails `json:"prompt_tokens_details,omitempty"`

	// CompletionDetails provides detailed breakdown of completion tokens.
	CompletionDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// GetPromptTokens returns the number of prompt tokens.
func (u *Usage) GetPromptTokens() int {
	if u == nil {
		return 0
	}
	return u.PromptTokens
}

// GetCompletionTokens returns the number of completion tokens.
func (u *Usage) GetCompletionTokens() int {
	if u == nil {
		return 0
	}
	return u.CompletionTokens
}

// GetTotalTokens returns the total number of tokens.
func (u *Usage) GetTotalTokens() int {
	if u == nil {
		return 0
	}
	return u.TotalTokens
}

// PromptTokensDetails provides detailed breakdown of prompt token usage.
type PromptTokensDetails struct {
	// CachedTokens is the number of cached tokens that didn't need processing.
	CachedTokens int `json:"cached_tokens,omitempty"`

	// AudioTokens is the number of tokens from audio input.
	AudioTokens int `json:"audio_tokens,omitempty"`
}

// CompletionTokensDetails provides detailed breakdown of completion token usage.
type CompletionTokensDetails struct {
	// ReasoningTokens is the number of tokens used for reasoning (e.g., in reasoning models).
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`

	// AudioTokens is the number of tokens from audio output.
	AudioTokens int `json:"audio_tokens,omitempty"`
}

// Tool represents a function tool that the model can call.
type Tool struct {
	// Type is the tool type. Currently only "function" is supported.
	Type string `json:"type"`

	// Function contains the function definition.
	Function Function `json:"function"`
}

// Function represents a function that can be called by the model.
//
// Thread Safety: Function is safe for concurrent reads after creation.
// The Parameters field should not be modified concurrently without external synchronization.
type Function struct {
	// Name is the function name that the model will use.
	Name string `json:"name"`

	// Description explains what the function does.
	// The model uses this to decide when to call the function.
	Description string `json:"description,omitempty"`

	// Parameters is the function's parameter schema in JSON Schema format.
	// Example:
	//   {
	//     "type": "object",
	//     "properties": {
	//       "location": {"type": "string", "description": "City name"},
	//       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
	//     },
	//     "required": ["location"]
	//   }
	Parameters map[string]any `json:"parameters"`
}

// ToolChoice controls which tool (if any) the model should call.
type ToolChoice struct {
	// Type controls tool selection behavior.
	// Valid values:
	// - "auto": model decides whether to call a tool
	// - "none": model will not call any tools
	// - "required": model must call at least one tool
	Type string `json:"type,omitempty"`

	// Function specifies a specific function to call.
	// When set, Type is ignored and this function is always called.
	Function *Function `json:"function,omitempty"`
}

// ToolCall represents a tool invocation made by the model.
type ToolCall struct {
	// ID is a unique identifier for this tool call.
	ID string `json:"id"`

	// Type is the tool type. Currently only "function" is supported.
	Type string `json:"type"`

	// Function contains the function call details.
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function invocation.
type FunctionCall struct {
	// Name is the function name to call.
	Name string `json:"name"`

	// Arguments contains the function arguments as a JSON string.
	// Parse this string to extract the actual argument values.
	Arguments string `json:"arguments"`
}

// ResponseFormat specifies the format of the model's response.
type ResponseFormat struct {
	// Type specifies the response format.
	// Valid values:
	// - "text": plain text response (default)
	// - "json_object": response will be valid JSON
	Type string `json:"type"`
}

// CompletionChunk represents a single chunk in a streaming response.
type CompletionChunk struct {
	// ID is a unique identifier for this completion stream.
	ID string `json:"id"`

	// Object is the object type (e.g., "chat.completion.chunk").
	Object string `json:"object"`

	// Created is the Unix timestamp (in seconds) of when the completion was created.
	Created int64 `json:"created"`

	// Model is the model used for this completion.
	Model string `json:"model"`

	// Choices contains the streaming choices.
	Choices []ChunkChoice `json:"choices"`

	// Usage contains cumulative token usage (only present in final chunk).
	Usage *Usage `json:"usage,omitempty"`
}

// ChunkChoice represents a single choice in a streaming chunk.
type ChunkChoice struct {
	// Index is the zero-based index of this choice.
	Index int `json:"index"`

	// Delta contains the incremental content for this chunk.
	Delta MessageDelta `json:"delta"`

	// FinishReason explains why the model stopped (only present in final chunk).
	// Valid values: "stop", "length", "tool_calls", "content_filter"
	FinishReason *string `json:"finish_reason"`

	// Logprobs contains log probability information for this chunk.
	Logprobs *Logprobs `json:"logprobs,omitempty"`
}

// MessageDelta represents incremental message content in a streaming response.
// Clients should accumulate deltas to build the complete message.
type MessageDelta struct {
	// Role is the message role (only present in the first chunk).
	Role string `json:"role,omitempty"`

	// Content contains the incremental text content.
	Content string `json:"content,omitempty"`

	// ToolCalls contains incremental tool call information.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// EmbeddingRequest represents a request to generate embeddings from text input.
//
// Thread Safety: EmbeddingRequest is safe for concurrent reads after creation.
// The Metadata field should not be modified concurrently without external synchronization.
type EmbeddingRequest struct {
	// Model specifies the embedding model to use. Format: "provider/model-name"
	// Examples: "openai/text-embedding-ada-002", "openai/text-embedding-3-small"
	Model string `json:"model"`

	// Input contains the text(s) to embed.
	// Can be either:
	// - string: single text to embed
	// - []string: multiple texts to embed in a batch
	Input any `json:"input"`

	// EncodingFormat specifies the format of the returned embeddings.
	// Valid values: "float" (default), "base64"
	EncodingFormat string `json:"encoding_format,omitempty"`

	// Dimensions specifies the number of dimensions for the embedding.
	// Only supported by certain models (e.g., text-embedding-3-*).
	Dimensions *int `json:"dimensions,omitempty"`

	// User is a unique identifier representing your end-user.
	// Used for abuse detection and monitoring.
	User string `json:"user,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// EmbeddingResponse represents the response from an embedding request.
//
// Thread Safety: EmbeddingResponse is safe for concurrent reads.
// Provider-specific fields should not be modified concurrently without external synchronization.
type EmbeddingResponse struct {
	// Object is the object type (e.g., "list").
	Object string `json:"object"`

	// Data contains the embedding results.
	Data []Embedding `json:"data"`

	// Model is the model used for this embedding.
	Model string `json:"model"`

	// Usage contains token usage information for this request.
	Usage *EmbeddingUsage `json:"usage,omitempty"`
}

// GetModel returns the model name.
func (r *EmbeddingResponse) GetModel() string {
	if r == nil {
		return ""
	}
	return r.Model
}

// GetUsageInfo returns the usage information as a generic interface.
// This satisfies the cost.EmbeddingResponse interface.
func (r *EmbeddingResponse) GetUsageInfo() interface{} {
	if r == nil {
		return nil
	}
	return r.Usage
}

// Embedding represents a single embedding vector.
type Embedding struct {
	// Object is the object type (e.g., "embedding").
	Object string `json:"object"`

	// Embedding is the vector representation of the input text.
	// Length depends on the model used.
	Embedding []float64 `json:"embedding"`

	// Index is the zero-based index of this embedding in the Data array.
	Index int `json:"index"`
}

// EmbeddingUsage represents token usage for an embedding request.
type EmbeddingUsage struct {
	// PromptTokens is the number of tokens in the input.
	PromptTokens int `json:"prompt_tokens"`

	// TotalTokens is the total number of tokens used.
	TotalTokens int `json:"total_tokens"`
}

// GetPromptTokens returns the number of prompt tokens.
func (u *EmbeddingUsage) GetPromptTokens() int {
	if u == nil {
		return 0
	}
	return u.PromptTokens
}

// GetCompletionTokens returns 0 for embeddings (no completion tokens).
func (u *EmbeddingUsage) GetCompletionTokens() int {
	return 0
}

// GetTotalTokens returns the total number of tokens.
func (u *EmbeddingUsage) GetTotalTokens() int {
	if u == nil {
		return 0
	}
	return u.TotalTokens
}

// Stream represents a streaming response from a provider.
//
// The caller must call Close() when done to release resources.
// Recv() should be called in a loop until it returns io.EOF.
//
// Thread Safety: Stream is NOT safe for concurrent use.
// Only one goroutine should call Recv() at a time.
//
// Example:
//
//	stream, err := client.CompletionStream(ctx, req)
//	if err != nil {
//	    return err
//	}
//	defer stream.Close()
//
//	for {
//	    chunk, err := stream.Recv()
//	    if err == io.EOF {
//	        break
//	    }
//	    if err != nil {
//	        return err
//	    }
//	    // Process chunk...
//	}
type Stream interface {
	// Recv receives the next chunk from the stream.
	//
	// Returns io.EOF when the stream is complete.
	// Returns other errors for failure conditions.
	//
	// After receiving io.EOF or any error, subsequent calls will return the same error.
	Recv() (*CompletionChunk, error)

	// Close closes the stream and releases resources.
	//
	// It is safe to call Close multiple times.
	// Close must be called even if Recv returns an error.
	//
	// Example:
	//   defer stream.Close()
	Close() error
}

// ImageGenerationRequest represents a request to generate images from text prompts.
//
// Thread Safety: ImageGenerationRequest is safe for concurrent reads after creation.
// The Metadata field should not be modified concurrently without external synchronization.
type ImageGenerationRequest struct {
	// Model specifies the image generation model. Format: "provider/model-name"
	// Examples: "openai/dall-e-3", "openai/dall-e-2", "azure/dall-e-3"
	Model string `json:"model"`

	// Prompt is the text description of the desired image(s).
	Prompt string `json:"prompt"`

	// N specifies how many images to generate (1-10).
	// Note: DALL-E 3 only supports N=1.
	N *int `json:"n,omitempty"`

	// Size specifies the image dimensions.
	// DALL-E 2: "256x256", "512x512", "1024x1024"
	// DALL-E 3: "1024x1024", "1024x1792", "1792x1024"
	Size string `json:"size,omitempty"`

	// Quality controls the image generation quality.
	// DALL-E 3: "standard" (default), "hd"
	Quality string `json:"quality,omitempty"`

	// Style controls the style of generated images.
	// DALL-E 3: "vivid" (default), "natural"
	Style string `json:"style,omitempty"`

	// ResponseFormat controls how images are returned.
	// Valid values: "url" (default), "b64_json"
	ResponseFormat string `json:"response_format,omitempty"`

	// User is a unique identifier for the end-user (for tracking/abuse detection).
	User string `json:"user,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Fallbacks contains fallback model names to try if the primary model fails.
	Fallbacks []string `json:"fallbacks,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// ImageGenerationResponse represents the response from an image generation request.
//
// Thread Safety: ImageGenerationResponse is safe for concurrent reads.
type ImageGenerationResponse struct {
	// Created is the Unix timestamp of when the images were created.
	Created int64 `json:"created"`

	// Data contains the generated images.
	Data []ImageData `json:"data"`

	// Provider is the provider that generated the images (internal metadata).
	Provider string `json:"-"`

	// Model is the model used for generation (internal metadata).
	Model string `json:"-"`
}

// ImageData represents a single generated image.
type ImageData struct {
	// URL is the HTTP(S) URL to the generated image (when ResponseFormat="url").
	URL string `json:"url,omitempty"`

	// B64JSON is the base64-encoded image data (when ResponseFormat="b64_json").
	B64JSON string `json:"b64_json,omitempty"`

	// RevisedPrompt is the AI-revised prompt (DALL-E 3 only).
	// Shows how the model interpreted/enhanced the original prompt.
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// ImageEditRequest represents an image editing request.
//
// Allows editing an image using AI based on a text prompt.
// The image must be a PNG file less than 4MB.
// An optional mask can specify which areas to edit.
//
// Thread Safety: ImageEditRequest is safe for concurrent reads after creation.
// The Image and Mask fields should not be read concurrently as io.Reader is not guaranteed to be thread-safe.
// The Metadata field should not be modified concurrently without external synchronization.
type ImageEditRequest struct {
	// Model specifies the image editing model. Format: "provider/model-name"
	// Examples: "openai/dall-e-2"
	// Note: DALL-E 3 does not support editing, only DALL-E 2.
	Model string `json:"model"`

	// Image is the PNG image to edit. Must be less than 4MB.
	// The reader will be fully consumed during the request.
	Image io.Reader `json:"-"`

	// ImageFilename is the name of the image file (including extension).
	// Required for multipart form data encoding.
	// Must end with ".png".
	ImageFilename string `json:"-"`

	// Prompt is the text description of the desired edit.
	// Describes what changes to make to the image.
	Prompt string `json:"prompt"`

	// Mask is an optional PNG image where transparent areas indicate where to edit.
	// Must be the same size as the image being edited.
	// If not provided, the entire image may be edited.
	Mask io.Reader `json:"-"`

	// MaskFilename is the name of the mask file (including extension).
	// Required if Mask is provided.
	// Must end with ".png".
	MaskFilename string `json:"-"`

	// N specifies how many edited images to generate (1-10).
	// Note: DALL-E 2 supports up to 10 images.
	N *int `json:"n,omitempty"`

	// Size specifies the output image dimensions.
	// DALL-E 2: "256x256", "512x512", "1024x1024"
	Size string `json:"size,omitempty"`

	// ResponseFormat controls how images are returned.
	// Valid values: "url" (default), "b64_json"
	ResponseFormat string `json:"response_format,omitempty"`

	// User is a unique identifier for the end-user (for tracking/abuse detection).
	User string `json:"user,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Fallbacks contains fallback model names to try if the primary model fails.
	Fallbacks []string `json:"fallbacks,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// ImageVariationRequest represents an image variation request.
//
// Creates variations of an existing image using AI.
// The image must be a PNG file less than 4MB.
//
// Thread Safety: ImageVariationRequest is safe for concurrent reads after creation.
// The Image field should not be read concurrently as io.Reader is not guaranteed to be thread-safe.
// The Metadata field should not be modified concurrently without external synchronization.
type ImageVariationRequest struct {
	// Model specifies the image variation model. Format: "provider/model-name"
	// Examples: "openai/dall-e-2"
	// Note: DALL-E 3 does not support variations, only DALL-E 2.
	// Defaults to "openai/dall-e-2" if not specified.
	Model string `json:"model,omitempty"`

	// Image is the PNG image to create variations of. Must be less than 4MB.
	// The reader will be fully consumed during the request.
	Image io.Reader `json:"-"`

	// ImageFilename is the name of the image file (including extension).
	// Required for multipart form data encoding.
	// Must end with ".png".
	ImageFilename string `json:"-"`

	// N specifies how many image variations to generate (1-10).
	// Note: DALL-E 2 supports up to 10 images.
	N *int `json:"n,omitempty"`

	// Size specifies the output image dimensions.
	// DALL-E 2: "256x256", "512x512", "1024x1024"
	Size string `json:"size,omitempty"`

	// ResponseFormat controls how images are returned.
	// Valid values: "url" (default), "b64_json"
	ResponseFormat string `json:"response_format,omitempty"`

	// User is a unique identifier for the end-user (for tracking/abuse detection).
	User string `json:"user,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Fallbacks contains fallback model names to try if the primary model fails.
	Fallbacks []string `json:"fallbacks,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// TranscriptionRequest represents an audio transcription request.
//
// Thread Safety: TranscriptionRequest is safe for concurrent reads after creation.
// The File field should not be read concurrently as io.Reader is not guaranteed to be thread-safe.
// The Metadata field should not be modified concurrently without external synchronization.
type TranscriptionRequest struct {
	// Model specifies the transcription model. Format: "provider/model-name"
	// Examples: "openai/whisper-1", "groq/whisper-large-v3"
	Model string `json:"model"`

	// File is the audio data to transcribe.
	// Supported formats: MP3, MP4, MPEG, MPGA, M4A, WAV, WEBM
	// The reader will be fully consumed during the request.
	File io.Reader `json:"-"`

	// Filename is the name of the audio file (including extension).
	// Required for multipart form data encoding.
	Filename string `json:"-"`

	// Language is a hint for the audio language (ISO 639-1 code).
	// Examples: "en", "es", "fr"
	// Improves accuracy and reduces latency.
	Language string `json:"language,omitempty"`

	// Prompt provides optional context to guide transcription.
	// Can include proper nouns, technical terms, etc.
	Prompt string `json:"prompt,omitempty"`

	// ResponseFormat specifies the output format.
	// Valid values: "json" (default), "text", "srt", "vtt", "verbose_json"
	ResponseFormat string `json:"response_format,omitempty"`

	// Temperature controls randomness in transcription (0.0-1.0).
	// 0.0 = most deterministic, higher = more creative interpretations.
	Temperature *float64 `json:"temperature,omitempty"`

	// TimestampGranularities specifies timestamp detail level.
	// Valid values: "word", "segment"
	// Requires ResponseFormat="verbose_json"
	TimestampGranularities []string `json:"timestamp_granularities,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Fallbacks contains fallback model names to try if the primary model fails.
	Fallbacks []string `json:"fallbacks,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// TranscriptionResponse represents a transcription response.
//
// Thread Safety: TranscriptionResponse is safe for concurrent reads.
type TranscriptionResponse struct {
	// Text is the full transcribed text.
	Text string `json:"text"`

	// Language is the detected or specified language.
	Language string `json:"language,omitempty"`

	// Duration is the audio duration in seconds.
	Duration float64 `json:"duration,omitempty"`

	// Words contains word-level timestamps (when TimestampGranularities includes "word").
	Words []Word `json:"words,omitempty"`

	// Segments contains detailed segment information (when ResponseFormat="verbose_json").
	Segments []Segment `json:"segments,omitempty"`

	// Provider is the provider that performed the transcription (internal metadata).
	Provider string `json:"-"`

	// Model is the model used for transcription (internal metadata).
	Model string `json:"-"`
}

// Word represents a transcribed word with timestamp.
type Word struct {
	// Word is the transcribed word.
	Word string `json:"word"`

	// Start is the start time in seconds.
	Start float64 `json:"start"`

	// End is the end time in seconds.
	End float64 `json:"end"`
}

// Segment represents a transcribed segment with detailed information.
type Segment struct {
	// ID is the segment identifier.
	ID int `json:"id"`

	// Seek is the seek position for this segment.
	Seek int `json:"seek"`

	// Start is the start time in seconds.
	Start float64 `json:"start"`

	// End is the end time in seconds.
	End float64 `json:"end"`

	// Text is the transcribed text for this segment.
	Text string `json:"text"`

	// Tokens contains the token IDs for this segment.
	Tokens []int `json:"tokens"`

	// Temperature is the sampling temperature used for this segment.
	Temperature float64 `json:"temperature"`

	// AvgLogprob is the average log probability of tokens in this segment.
	AvgLogprob float64 `json:"avg_logprob"`

	// CompressionRatio is the compression ratio of this segment.
	CompressionRatio float64 `json:"compression_ratio"`

	// NoSpeechProb is the probability of no speech in this segment.
	NoSpeechProb float64 `json:"no_speech_prob"`
}

// SpeechRequest represents a text-to-speech request.
//
// Thread Safety: SpeechRequest is safe for concurrent reads after creation.
// The Metadata field should not be modified concurrently without external synchronization.
type SpeechRequest struct {
	// Model specifies the TTS model. Format: "provider/model-name"
	// Examples: "openai/tts-1", "openai/tts-1-hd"
	Model string `json:"model"`

	// Input is the text to convert to speech (max 4096 characters).
	Input string `json:"input"`

	// Voice specifies the voice to use.
	// OpenAI: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
	Voice string `json:"voice"`

	// ResponseFormat specifies the audio format.
	// Valid values: "mp3" (default), "opus", "aac", "flac", "wav", "pcm"
	ResponseFormat string `json:"response_format,omitempty"`

	// Speed controls the playback speed (0.25 to 4.0).
	// 1.0 = normal speed. Default: 1.0
	Speed *float64 `json:"speed,omitempty"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`

	// NumRetries specifies the number of retry attempts for failed requests.
	NumRetries int `json:"num_retries,omitempty"`

	// Fallbacks contains fallback model names to try if the primary model fails.
	Fallbacks []string `json:"fallbacks,omitempty"`

	// Timeout specifies the maximum duration for this request.
	Timeout time.Duration `json:"timeout,omitempty"`
}

// ModerationRequest represents a content moderation request.
//
// The input can be a single string or an array of strings to check for content policy violations.
//
// Thread Safety: ModerationRequest is safe for concurrent reads after creation.
// The Metadata field should not be modified concurrently without external synchronization.
type ModerationRequest struct {
	// Model specifies the moderation model to use. Format: "provider/model-name"
	// Examples: "openai/text-moderation-latest", "openai/text-moderation-stable"
	// Default: "openai/text-moderation-latest"
	Model string `json:"model,omitempty"`

	// Input contains the text(s) to check for policy violations.
	// Can be either:
	// - string: single text to moderate
	// - []string: multiple texts to moderate in a batch
	Input any `json:"input"`

	// APIKey overrides the provider API key for this request.
	APIKey string `json:"api_key,omitempty"`

	// APIBase overrides the provider API base URL for this request.
	APIBase string `json:"api_base,omitempty"`

	// APIVersion overrides the provider API version for this request.
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary key-value pairs for callbacks and tracking.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ModerationResponse represents a content moderation response.
//
// Thread Safety: ModerationResponse is safe for concurrent reads.
type ModerationResponse struct {
	// ID is a unique identifier for this moderation request.
	ID string `json:"id"`

	// Model is the model used for moderation.
	Model string `json:"model"`

	// Results contains moderation results for each input text.
	// The results array has the same length as the input array.
	Results []ModerationResult `json:"results"`

	// Provider is the provider that performed the moderation (internal metadata).
	Provider string `json:"-"`
}

// ModerationResult represents moderation results for a single input text.
type ModerationResult struct {
	// Flagged indicates whether the content violates any policy.
	// True if any category is flagged.
	Flagged bool `json:"flagged"`

	// Categories contains boolean flags for each content category.
	Categories ModerationCategories `json:"categories"`

	// CategoryScores contains confidence scores (0.0 to 1.0) for each category.
	// Higher scores indicate higher confidence that the content violates the policy.
	CategoryScores ModerationCategoryScores `json:"category_scores"`
}

// ModerationCategories represents boolean flags for content policy categories.
type ModerationCategories struct {
	// Sexual content (not including sexual content involving minors).
	Sexual bool `json:"sexual"`

	// Hate content that expresses, incites, or promotes hate based on identity.
	Hate bool `json:"hate"`

	// Harassment content that promotes harassment or bullying.
	Harassment bool `json:"harassment"`

	// Self-harm content that promotes, encourages, or depicts acts of self-harm.
	SelfHarm bool `json:"self-harm"`

	// Sexual content involving minors.
	SexualMinors bool `json:"sexual/minors"`

	// Hate content that also includes violence or serious harm.
	HateThreatening bool `json:"hate/threatening"`

	// Violent content depicted in graphic detail.
	ViolenceGraphic bool `json:"violence/graphic"`

	// Self-harm content where the intent is to encourage or provoke.
	SelfHarmIntent bool `json:"self-harm/intent"`

	// Self-harm content that provides instructions or advice.
	SelfHarmInstructions bool `json:"self-harm/instructions"`

	// Harassment content that also includes threats or violence.
	HarassmentThreatening bool `json:"harassment/threatening"`

	// Violence content that depicts violence or describes death, violence, or injury.
	Violence bool `json:"violence"`
}

// ModerationCategoryScores represents confidence scores for content policy categories.
//
// Scores range from 0.0 to 1.0, where higher scores indicate higher confidence
// that the content violates the policy for that category.
type ModerationCategoryScores struct {
	// Confidence score for sexual content.
	Sexual float64 `json:"sexual"`

	// Confidence score for hate content.
	Hate float64 `json:"hate"`

	// Confidence score for harassment content.
	Harassment float64 `json:"harassment"`

	// Confidence score for self-harm content.
	SelfHarm float64 `json:"self-harm"`

	// Confidence score for sexual content involving minors.
	SexualMinors float64 `json:"sexual/minors"`

	// Confidence score for threatening hate content.
	HateThreatening float64 `json:"hate/threatening"`

	// Confidence score for graphic violence.
	ViolenceGraphic float64 `json:"violence/graphic"`

	// Confidence score for self-harm with intent to provoke.
	SelfHarmIntent float64 `json:"self-harm/intent"`

	// Confidence score for self-harm instructions.
	SelfHarmInstructions float64 `json:"self-harm/instructions"`

	// Confidence score for threatening harassment.
	HarassmentThreatening float64 `json:"harassment/threatening"`

	// Confidence score for violence content.
	Violence float64 `json:"violence"`
}

// Float64Ptr returns a pointer to the provided float64 value.
// This helper function is useful for setting optional pointer fields in request types.
func Float64Ptr(v float64) *float64 {
	return &v
}

// IntPtr returns a pointer to the provided int value.
// This helper function is useful for setting optional pointer fields in request types.
func IntPtr(v int) *int {
	return &v
}

// BoolPtr returns a pointer to the provided bool value.
// This helper function is useful for setting optional pointer fields in request types.
func BoolPtr(v bool) *bool {
	return &v
}

// RerankRequest represents a document reranking request.
//
// Used for RAG (Retrieval-Augmented Generation) applications to rank
// retrieved documents by relevance to a query.
//
// Thread Safety: RerankRequest is safe for concurrent reads after creation.
// The Metadata field should not be modified concurrently without external synchronization.
type RerankRequest struct {
	// Model specifies the reranking model to use. Format: "provider/model-name"
	// Examples: "cohere/rerank-english-v3.0", "cohere/rerank-multilingual-v3.0"
	Model string `json:"model"`

	// Query is the search query to rank documents against.
	Query string `json:"query"`

	// Documents are the documents to rank by relevance to the query.
	Documents []string `json:"documents"`

	// TopN returns only the top N most relevant documents.
	// If nil or 0, all documents are returned.
	TopN *int `json:"top_n,omitempty"`

	// ReturnDocuments includes the document text in the response.
	// If false, only indices and relevance scores are returned.
	ReturnDocuments *bool `json:"return_documents,omitempty"`

	// MaxChunksPerDoc is a Cohere-specific parameter that controls
	// how many chunks each document is split into for reranking.
	// Only applies to Cohere provider.
	MaxChunksPerDoc *int `json:"max_chunks_per_doc,omitempty"`

	// Provider overrides
	APIKey     string `json:"api_key,omitempty"`
	APIBase    string `json:"api_base,omitempty"`
	APIVersion string `json:"api_version,omitempty"`

	// Metadata contains arbitrary metadata for tracking and logging.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// RerankResponse represents a reranking response.
//
// Contains ranked documents with relevance scores, sorted by relevance
// (most relevant first).
//
// Thread Safety: RerankResponse is safe for concurrent reads after creation.
type RerankResponse struct {
	// ID is a unique identifier for this reranking request.
	ID string `json:"id"`

	// Results contains the ranked documents, sorted by relevance score (highest first).
	Results []RerankResult `json:"results"`

	// Meta contains metadata about the reranking operation.
	Meta *RerankMeta `json:"meta,omitempty"`

	// Internal metadata (not serialized)
	Provider string `json:"-"`
	Model    string `json:"-"`
}

// RerankResult represents a single ranked document.
//
// Contains the document's index in the original input, its relevance score,
// and optionally the document text.
type RerankResult struct {
	// Index is the zero-based index of this document in the original Documents array.
	Index int `json:"index"`

	// RelevanceScore indicates how relevant this document is to the query.
	// Scores typically range from 0.0 to 1.0, with higher scores indicating higher relevance.
	// The exact range may vary by provider.
	RelevanceScore float64 `json:"relevance_score"`

	// Document contains the document text if ReturnDocuments was true in the request.
	// Otherwise this field is empty.
	Document string `json:"document,omitempty"`
}

// RerankMeta contains metadata about the reranking operation.
//
// Includes API version information and billing details.
type RerankMeta struct {
	// APIVersion contains information about the API version used.
	APIVersion *APIVersion `json:"api_version,omitempty"`

	// BilledUnits contains billing information for the request.
	BilledUnits *BilledUnits `json:"billed_units,omitempty"`
}

// APIVersion represents API version information.
type APIVersion struct {
	// Version is the API version string.
	Version string `json:"version"`
}

// BilledUnits represents billing information for a reranking request.
type BilledUnits struct {
	// SearchUnits is the number of search units consumed by this request.
	// Used by Cohere for billing rerank requests.
	SearchUnits int `json:"search_units,omitempty"`
}
