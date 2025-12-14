package openrouter

import (
	"context"
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
	return nil, &warp.WarpError{
		Message:  "image generation is not yet implemented for OpenRouter",
		Provider: "openrouter",
	}
}

// Transcription transcribes audio to text.
//
// OpenRouter does not support audio transcription through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by OpenRouter",
		Provider: "openrouter",
	}
}

// Speech converts text to speech.
//
// OpenRouter does not support text-to-speech through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by OpenRouter",
		Provider: "openrouter",
	}
}

// ImageEdit edits an image using AI based on a text prompt.
//
// OpenRouter does not support image editing through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by OpenRouter",
		Provider: "openrouter",
	}
}

// ImageVariation creates variations of an existing image.
//
// OpenRouter does not support image variation through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by OpenRouter",
		Provider: "openrouter",
	}
}

// Moderation checks content for policy violations.
//
// OpenRouter does not support content moderation through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "content moderation is not supported by OpenRouter",
		Provider: "openrouter",
	}
}

// Rerank ranks documents by relevance to a query.
//
// OpenRouter does not support document reranking through its API.
//
// Returns an error indicating the feature is not supported.
func (p *Provider) Rerank(ctx context.Context, req *warp.RerankRequest) (*warp.RerankResponse, error) {
	return nil, &warp.WarpError{
		Message:  "document reranking is not supported by OpenRouter",
		Provider: "openrouter",
	}
}
