package openai

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/blue-context/warp"
)

// mockHTTPClient is a mock HTTP client for testing
type mockTranscriptionHTTPClient struct {
	doFunc func(req *http.Request) (*http.Response, error)
}

func (m *mockTranscriptionHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return m.doFunc(req)
}

func TestProviderTranscription(t *testing.T) {
	tests := []struct {
		name         string
		req          *warp.TranscriptionRequest
		mockResponse string
		mockStatus   int
		wantErr      bool
		wantErrMsg   string
		checkResp    func(*testing.T, *warp.TranscriptionResponse)
	}{
		{
			name: "successful transcription (json)",
			req: &warp.TranscriptionRequest{
				Model:          "whisper-1",
				File:           strings.NewReader("fake audio data"),
				Filename:       "test.mp3",
				Language:       "en",
				ResponseFormat: "json",
			},
			mockResponse: `{"text": "Hello, world!"}`,
			mockStatus:   http.StatusOK,
			wantErr:      false,
			checkResp: func(t *testing.T, resp *warp.TranscriptionResponse) {
				if resp.Text != "Hello, world!" {
					t.Errorf("Text = %q, want %q", resp.Text, "Hello, world!")
				}
			},
		},
		{
			name: "successful transcription (verbose_json)",
			req: &warp.TranscriptionRequest{
				Model:                  "whisper-1",
				File:                   strings.NewReader("fake audio data"),
				Filename:               "test.mp3",
				Language:               "en",
				ResponseFormat:         "verbose_json",
				TimestampGranularities: []string{"word", "segment"},
			},
			mockResponse: `{
				"text": "The quick brown fox",
				"language": "en",
				"duration": 2.5,
				"words": [
					{"word": "The", "start": 0.0, "end": 0.5},
					{"word": "quick", "start": 0.5, "end": 1.0}
				],
				"segments": [
					{
						"id": 0,
						"seek": 0,
						"start": 0.0,
						"end": 2.0,
						"text": "The quick brown fox",
						"tokens": [123, 456, 789],
						"temperature": 0.0,
						"avg_logprob": -0.5,
						"compression_ratio": 1.2,
						"no_speech_prob": 0.01
					}
				]
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
			checkResp: func(t *testing.T, resp *warp.TranscriptionResponse) {
				if resp.Text != "The quick brown fox" {
					t.Errorf("Text = %q, want %q", resp.Text, "The quick brown fox")
				}
				if resp.Language != "en" {
					t.Errorf("Language = %q, want %q", resp.Language, "en")
				}
				if resp.Duration != 2.5 {
					t.Errorf("Duration = %f, want %f", resp.Duration, 2.5)
				}
				if len(resp.Words) != 2 {
					t.Errorf("len(Words) = %d, want 2", len(resp.Words))
				}
				if len(resp.Segments) != 1 {
					t.Errorf("len(Segments) = %d, want 1", len(resp.Segments))
				}
			},
		},
		{
			name: "text response format",
			req: &warp.TranscriptionRequest{
				Model:          "whisper-1",
				File:           strings.NewReader("audio data"),
				Filename:       "test.mp3",
				ResponseFormat: "text",
			},
			mockResponse: "Plain text transcription",
			mockStatus:   http.StatusOK,
			wantErr:      false,
			checkResp: func(t *testing.T, resp *warp.TranscriptionResponse) {
				if resp.Text != "Plain text transcription" {
					t.Errorf("Text = %q, want %q", resp.Text, "Plain text transcription")
				}
			},
		},
		{
			name: "srt response format",
			req: &warp.TranscriptionRequest{
				Model:          "whisper-1",
				File:           strings.NewReader("audio data"),
				Filename:       "test.mp3",
				ResponseFormat: "srt",
			},
			mockResponse: "1\n00:00:00,000 --> 00:00:02,000\nHello, world!",
			mockStatus:   http.StatusOK,
			wantErr:      false,
			checkResp: func(t *testing.T, resp *warp.TranscriptionResponse) {
				if !strings.Contains(resp.Text, "Hello, world!") {
					t.Errorf("Text should contain subtitle: %q", resp.Text)
				}
			},
		},
		{
			name: "with temperature",
			req: &warp.TranscriptionRequest{
				Model:          "whisper-1",
				File:           strings.NewReader("audio data"),
				Filename:       "test.mp3",
				Temperature:    floatPtr(0.3),
				ResponseFormat: "json",
			},
			mockResponse: `{"text": "Transcription with temperature"}`,
			mockStatus:   http.StatusOK,
			wantErr:      false,
		},
		{
			name: "with prompt",
			req: &warp.TranscriptionRequest{
				Model:          "whisper-1",
				File:           strings.NewReader("audio data"),
				Filename:       "test.mp3",
				Prompt:         "This audio is about AI and machine learning",
				ResponseFormat: "json",
			},
			mockResponse: `{"text": "Transcription with prompt context"}`,
			mockStatus:   http.StatusOK,
			wantErr:      false,
		},
		{
			name:         "nil request",
			req:          nil,
			mockResponse: `{"error": "bad request"}`,
			mockStatus:   http.StatusBadRequest,
			wantErr:      true,
			wantErrMsg:   "request cannot be nil",
		},
		{
			name: "nil file",
			req: &warp.TranscriptionRequest{
				Model:    "whisper-1",
				Filename: "test.mp3",
			},
			mockResponse: `{"error": "bad request"}`,
			mockStatus:   http.StatusBadRequest,
			wantErr:      true,
			wantErrMsg:   "file is required",
		},
		{
			name: "empty filename",
			req: &warp.TranscriptionRequest{
				Model: "whisper-1",
				File:  strings.NewReader("data"),
			},
			mockResponse: `{"error": "bad request"}`,
			mockStatus:   http.StatusBadRequest,
			wantErr:      true,
			wantErrMsg:   "filename is required",
		},
		{
			name: "API error",
			req: &warp.TranscriptionRequest{
				Model:    "whisper-1",
				File:     strings.NewReader("audio data"),
				Filename: "test.mp3",
			},
			mockResponse: `{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}`,
			mockStatus:   http.StatusTooManyRequests,
			wantErr:      true,
			wantErrMsg:   "Rate limit",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock HTTP client
			mockClient := &mockTranscriptionHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					// Verify request
					if req.Method != "POST" {
						t.Errorf("Method = %s, want POST", req.Method)
					}
					if !strings.HasSuffix(req.URL.Path, "/audio/transcriptions") {
						t.Errorf("Path = %s, want /audio/transcriptions", req.URL.Path)
					}

					// Check headers
					if !strings.HasPrefix(req.Header.Get("Content-Type"), "multipart/form-data") {
						t.Errorf("Content-Type should be multipart/form-data")
					}
					if !strings.HasPrefix(req.Header.Get("Authorization"), "Bearer ") {
						t.Errorf("Authorization header should start with 'Bearer '")
					}

					// Return mock response
					return &http.Response{
						StatusCode: tt.mockStatus,
						Body:       io.NopCloser(bytes.NewBufferString(tt.mockResponse)),
						Header:     make(http.Header),
					}, nil
				},
			}

			// Create provider
			provider := &Provider{
				apiKey:     "test-key",
				apiBase:    "https://api.openai.com/v1",
				httpClient: mockClient,
			}

			// Call Transcription
			ctx := context.Background()
			resp, err := provider.Transcription(ctx, tt.req)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Transcription() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				if tt.wantErrMsg != "" && !strings.Contains(err.Error(), tt.wantErrMsg) {
					t.Errorf("error message = %q, want to contain %q", err.Error(), tt.wantErrMsg)
				}
				return
			}

			// Check response
			if resp == nil {
				t.Fatal("response is nil")
			}

			if tt.checkResp != nil {
				tt.checkResp(t, resp)
			}
		})
	}
}

func TestProviderTranscriptionWithAPIOverrides(t *testing.T) {
	customAPIKey := "custom-key"
	customAPIBase := "https://custom.api.com/v1"

	mockClient := &mockTranscriptionHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			// Verify custom API key is used
			auth := req.Header.Get("Authorization")
			if auth != "Bearer "+customAPIKey {
				t.Errorf("Authorization = %s, want Bearer %s", auth, customAPIKey)
			}

			// Verify custom API base is used
			if !strings.HasPrefix(req.URL.String(), customAPIBase) {
				t.Errorf("URL = %s, should start with %s", req.URL.String(), customAPIBase)
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(`{"text": "Custom API response"}`)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider := &Provider{
		apiKey:     "default-key",
		apiBase:    "https://api.openai.com/v1",
		httpClient: mockClient,
	}

	req := &warp.TranscriptionRequest{
		Model:    "whisper-1",
		File:     strings.NewReader("audio data"),
		Filename: "test.mp3",
		APIKey:   customAPIKey,
		APIBase:  customAPIBase,
	}

	ctx := context.Background()
	resp, err := provider.Transcription(ctx, req)
	if err != nil {
		t.Fatalf("Transcription() error = %v", err)
	}

	if resp.Text != "Custom API response" {
		t.Errorf("Text = %q, want %q", resp.Text, "Custom API response")
	}
}

func TestProviderTranscriptionMultipartEncoding(t *testing.T) {
	// Test that multipart form is correctly encoded
	mockClient := &mockTranscriptionHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			// Read and verify multipart form
			contentType := req.Header.Get("Content-Type")
			if !strings.HasPrefix(contentType, "multipart/form-data; boundary=") {
				t.Errorf("Invalid Content-Type: %s", contentType)
			}

			// Read body
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("Failed to read body: %v", err)
			}

			bodyStr := string(body)

			// Verify fields are present
			if !strings.Contains(bodyStr, "whisper-1") {
				t.Error("model field not found in multipart body")
			}
			if !strings.Contains(bodyStr, "en") {
				t.Error("language field not found in multipart body")
			}
			if !strings.Contains(bodyStr, "test prompt") {
				t.Error("prompt field not found in multipart body")
			}
			if !strings.Contains(bodyStr, "fake audio data") {
				t.Error("file content not found in multipart body")
			}

			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewBufferString(`{"text": "Transcription"}`)),
				Header:     make(http.Header),
			}, nil
		},
	}

	provider := &Provider{
		apiKey:     "test-key",
		apiBase:    "https://api.openai.com/v1",
		httpClient: mockClient,
	}

	req := &warp.TranscriptionRequest{
		Model:    "whisper-1",
		File:     strings.NewReader("fake audio data"),
		Filename: "test.mp3",
		Language: "en",
		Prompt:   "test prompt",
	}

	ctx := context.Background()
	_, err := provider.Transcription(ctx, req)
	if err != nil {
		t.Fatalf("Transcription() error = %v", err)
	}
}

// Helper function
func floatPtr(f float64) *float64 {
	return &f
}
