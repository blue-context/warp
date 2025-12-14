package vertex

import (
	"context"
	"io"

	"github.com/blue-context/warp"
)

// Moderation checks content for policy violations.
//
// Vertex does not support content moderation.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "moderation is not supported by Vertex",
		Provider: "vertex",
	}
}

// Transcription transcribes audio to text.
//
// Vertex does not support audio transcription.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by Vertex",
		Provider: "vertex",
	}
}

// Speech converts text to speech.
//
// Vertex does not support text-to-speech.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by Vertex",
		Provider: "vertex",
	}
}

// ImageGeneration generates images from text prompts.
//
// Vertex does not support image generation.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image generation is not supported by Vertex",
		Provider: "vertex",
	}
}

// ImageEdit edits an image using AI based on a text prompt.
//
// Vertex does not support image editing.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by Vertex",
		Provider: "vertex",
	}
}

// ImageVariation creates variations of an existing image.
//
// Vertex does not support image variation.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by Vertex",
		Provider: "vertex",
	}
}
