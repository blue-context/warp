package openrouter

import (
	"context"
	"fmt"
	"io"

	"github.com/blue-context/warp"
)

// ImageGeneration generates images from text prompts.
//
// OpenRouter supports image generation through specific models like DALL-E,
// but this method is not yet implemented in this provider.
//
// Returns an error indicating the feature is not yet implemented.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image generation not yet implemented for OpenRouter provider")
}

// Transcription transcribes audio to text.
//
// OpenRouter does not support audio transcription through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, fmt.Errorf("transcription not supported by OpenRouter provider")
}

// Speech converts text to speech.
//
// OpenRouter does not support text-to-speech through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("speech synthesis not supported by OpenRouter provider")
}

// ImageEdit edits an image using AI based on a text prompt.
//
// OpenRouter does not support image editing through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image editing not supported by OpenRouter provider")
}

// ImageVariation creates variations of an existing image.
//
// OpenRouter does not support image variation through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, fmt.Errorf("image variation not supported by OpenRouter provider")
}

// Moderation checks content for policy violations.
//
// OpenRouter does not support content moderation through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, fmt.Errorf("content moderation not supported by OpenRouter provider")
}

// Rerank ranks documents by relevance to a query.
//
// OpenRouter does not support document reranking through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, fmt.Errorf("document reranking not supported by OpenRouter provider")
}
