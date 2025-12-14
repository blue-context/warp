package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Speech converts text to speech using OpenAI TTS.
//
// Returns an io.ReadCloser containing the audio data.
// The caller MUST close the reader when done.
//
// Supported models: "tts-1", "tts-1-hd"
// Supported voices: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
// Supported formats: "mp3", "opus", "aac", "flac", "wav", "pcm"
//
// Thread Safety: This method is safe for concurrent use.
//
// Example:
//
//	audio, err := provider.Speech(ctx, &warp.SpeechRequest{
//	    Model: "tts-1",
//	    Input: "Hello, world!",
//	    Voice: "alloy",
//	})
//	if err != nil {
//	    return err
//	}
//	defer audio.Close()
//
//	f, _ := os.Create("output.mp3")
//	defer f.Close()
//	io.Copy(f, audio)
func (p *Provider) Speech(ctx context.Context, req *warp.SpeechRequest) (io.ReadCloser, error) {
	// Build request body
	reqBody := map[string]interface{}{
		"model": req.Model,
		"input": req.Input,
		"voice": req.Voice,
	}

	// Add optional parameters
	if req.ResponseFormat != "" {
		reqBody["response_format"] = req.ResponseFormat
	}
	if req.Speed != nil {
		reqBody["speed"] = *req.Speed
	}

	// Marshal request
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Use per-request overrides if provided
	apiKey := p.apiKey
	if req.APIKey != "" {
		apiKey = req.APIKey
	}

	apiBase := p.apiBase
	if req.APIBase != "" {
		apiBase = req.APIBase
	}

	// Create HTTP request
	url := apiBase + "/audio/speech"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check status
	if httpResp.StatusCode != http.StatusOK {
		// Read error response
		defer httpResp.Body.Close()
		errBody, _ := io.ReadAll(httpResp.Body)

		// Parse error
		return nil, warp.ParseProviderError("openai", httpResp.StatusCode, errBody, nil)
	}

	// Return audio stream (caller must close!)
	// Important: We do NOT close the response body here.
	// The caller is responsible for closing the returned io.ReadCloser.
	return httpResp.Body, nil
}
