package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to Cohere.
//
// This method handles the complete request/response cycle including:
// - Request transformation to Cohere format
// - HTTP request/response handling
// - Error parsing and classification
// - Response transformation to Warp format
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "command-r",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	// Transform request to Cohere format
	cohereReq := transformToCohereRequest(req)

	// Marshal to JSON
	body, err := json.Marshal(cohereReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("cohere", httpResp.StatusCode, body, nil)
	}

	// Parse response
	var cohereResp cohereResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&cohereResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Transform to Warp format
	resp := transformFromCohereResponse(&cohereResp)

	// Set the model from request since Cohere doesn't echo it
	resp.Model = req.Model

	return resp, nil
}

// CompletionStream is not implemented for Cohere.
//
// Cohere uses a different streaming format that requires separate implementation.
// Use Completion() for non-streaming requests.
func (p *Provider) CompletionStream(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error) {
	return nil, &warp.WarpError{
		Message:  "streaming is not supported by this Cohere provider implementation",
		Provider: "cohere",
	}
}

// Embedding is not supported by this Cohere provider implementation.
//
// Cohere has a separate /embed endpoint with different request/response format.
// Use the Cohere embeddings API directly if you need embedding support.
func (p *Provider) Embedding(ctx context.Context, req *warp.EmbeddingRequest) (*warp.EmbeddingResponse, error) {
	return nil, &warp.WarpError{
		Message:  "embeddings are not supported by this Cohere provider implementation",
		Provider: "cohere",
	}
}

// Transcription transcribes audio to text.
//
// Cohere does not support audio transcription.
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	return nil, &warp.WarpError{
		Message:  "transcription is not supported by Cohere",
		Provider: "cohere",
	}
}

// Speech converts text to speech.
//
// Cohere does not support text-to-speech.
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	return nil, &warp.WarpError{
		Message:  "speech synthesis is not supported by Cohere",
		Provider: "cohere",
	}
}

// Moderation checks content for policy violations.
//
// Cohere does not support content moderation.
func (p *Provider) Moderation(ctx context.Context, req *warp.ModerationRequest) (*warp.ModerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "moderation is not supported by Cohere",
		Provider: "cohere",
	}
}

// ImageGeneration generates images from text prompts.
//
// Cohere does not support image generation.
func (p *Provider) ImageGeneration(ctx context.Context, req *warp.ImageGenerationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image generation is not supported by Cohere",
		Provider: "cohere",
	}
}

// ImageEdit edits an image using AI.
//
// Cohere does not support image editing.
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image editing is not supported by Cohere",
		Provider: "cohere",
	}
}

// ImageVariation creates variations of an image.
//
// Cohere does not support image variations.
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	return nil, &warp.WarpError{
		Message:  "image variation is not supported by Cohere",
		Provider: "cohere",
	}
}
