package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/internal/multipart"
)

// Transcription transcribes audio to text using OpenAI's Whisper model.
//
// Supports all Whisper API parameters including language hints, prompts,
// response formats (json, text, srt, vtt, verbose_json), and timestamp
// granularities (word, segment).
//
// Thread Safety: This method is safe for concurrent use.
//
// Example:
//
//	f, err := os.Open("audio.mp3")
//	if err != nil {
//	    return err
//	}
//	defer f.Close()
//
//	resp, err := provider.Transcription(ctx, &warp.TranscriptionRequest{
//	    Model:    "whisper-1",
//	    File:     f,
//	    Filename: "audio.mp3",
//	    Language: "en",
//	    ResponseFormat: "verbose_json",
//	    TimestampGranularities: []string{"word", "segment"},
//	})
func (p *Provider) Transcription(ctx context.Context, req *warp.TranscriptionRequest) (*warp.TranscriptionResponse, error) {
	if req == nil {
		return nil, &warp.WarpError{
			Message:  "request cannot be nil",
			Provider: "openai",
		}
	}
	if req.File == nil {
		return nil, &warp.WarpError{
			Message:  "file is required",
			Provider: "openai",
		}
	}
	if req.Filename == "" {
		return nil, &warp.WarpError{
			Message:  "filename is required",
			Provider: "openai",
		}
	}

	// Use request-specific API key/base if provided
	apiKey := p.apiKey
	if req.APIKey != "" {
		apiKey = req.APIKey
	}

	apiBase := p.apiBase
	if req.APIBase != "" {
		apiBase = req.APIBase
	}

	// Build form fields
	fields := map[string]string{
		"model": req.Model,
	}

	if req.Language != "" {
		fields["language"] = req.Language
	}

	if req.Prompt != "" {
		fields["prompt"] = req.Prompt
	}

	if req.ResponseFormat != "" {
		fields["response_format"] = req.ResponseFormat
	}

	if req.Temperature != nil {
		fields["temperature"] = strconv.FormatFloat(*req.Temperature, 'f', -1, 64)
	}

	if len(req.TimestampGranularities) > 0 {
		// OpenAI expects timestamp_granularities[] as multiple form fields
		for _, granularity := range req.TimestampGranularities {
			// Note: For multiple values, we'll handle this specially
			// For now, join with comma (will be split in multipart encoding if needed)
			if _, exists := fields["timestamp_granularities[]"]; exists {
				fields["timestamp_granularities[]"] += "," + granularity
			} else {
				fields["timestamp_granularities[]"] = granularity
			}
		}
	}

	// Create multipart form data
	body, contentType, err := multipart.CreateFormFile("file", req.Filename, req.File, fields)
	if err != nil {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("failed to create multipart form: %v", err),
			Provider: "openai",
		}
	}

	// Create HTTP request
	url := apiBase + "/audio/transcriptions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("failed to create HTTP request: %v", err),
			Provider: "openai",
		}
	}

	// Set headers
	httpReq.Header.Set("Content-Type", contentType)
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("HTTP request failed: %v", err),
			Provider: "openai",
		}
	}
	defer httpResp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("failed to read response: %v", err),
			Provider: "openai",
		}
	}

	// Check for errors
	if httpResp.StatusCode != http.StatusOK {
		return nil, warp.ParseProviderError("openai", httpResp.StatusCode, respBody, nil)
	}

	// Parse response based on format
	responseFormat := req.ResponseFormat
	if responseFormat == "" {
		responseFormat = "json" // Default format
	}

	var resp warp.TranscriptionResponse

	switch responseFormat {
	case "text":
		// Plain text response
		resp.Text = string(respBody)

	case "srt", "vtt":
		// Subtitle formats return plain text
		resp.Text = string(respBody)

	case "json", "verbose_json":
		// JSON response
		if err := json.Unmarshal(respBody, &resp); err != nil {
			return nil, &warp.WarpError{
				Message:  fmt.Sprintf("failed to decode JSON response: %v", err),
				Provider: "openai",
			}
		}

	default:
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("unsupported response format: %s", responseFormat),
			Provider: "openai",
		}
	}

	return &resp, nil
}

// openaiTranscriptionResponse represents the OpenAI transcription API response.
// This matches the OpenAI API response structure for verbose_json format.
type openaiTranscriptionResponse struct {
	Text     string          `json:"text"`
	Task     string          `json:"task,omitempty"`
	Language string          `json:"language,omitempty"`
	Duration float64         `json:"duration,omitempty"`
	Words    []openaiWord    `json:"words,omitempty"`
	Segments []openaiSegment `json:"segments,omitempty"`
}

type openaiWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type openaiSegment struct {
	ID               int     `json:"id"`
	Seek             int     `json:"seek"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int   `json:"tokens"`
	Temperature      float64 `json:"temperature"`
	AvgLogprob       float64 `json:"avg_logprob"`
	CompressionRatio float64 `json:"compression_ratio"`
	NoSpeechProb     float64 `json:"no_speech_prob"`
}
