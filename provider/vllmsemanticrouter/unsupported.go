package vllmsemanticrouter

import (
	"context"
	"fmt"
	"io"

	"github.com/blue-context/warp"
)

// Embedding sends an embedding request.
//
// vLLM Semantic Router focuses on intelligent routing for completions
// and does not support embeddings.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	return nil, fmt.Errorf("embeddings not supported by vllm-semantic-router provider")
}

// ImageGeneration generates images from text prompts.
//
// vLLM Semantic Router does not support image generation.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image generation not supported by vllm-semantic-router provider")
}

// ImageEdit edits an image using AI based on a text prompt.
//
// vLLM Semantic Router does not support image editing.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image editing not supported by vllm-semantic-router provider")
}

// ImageVariation creates variations of an existing image.
//
// vLLM Semantic Router does not support image variation.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image variation not supported by vllm-semantic-router provider")
}

// Transcription transcribes audio to text.
//
// vLLM Semantic Router does not support audio transcription.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, fmt.Errorf("transcription not supported by vllm-semantic-router provider")
}

// Speech converts text to speech.
//
// vLLM Semantic Router does not support text-to-speech.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("speech synthesis not supported by vllm-semantic-router provider")
}

// Moderation checks content for policy violations.
//
// vLLM Semantic Router does not support content moderation through
// the main API, though classification endpoints are available separately.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, fmt.Errorf("content moderation not supported by vllm-semantic-router provider")
}
