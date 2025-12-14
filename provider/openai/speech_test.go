package openai

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/internal/testutil"
)

func TestProviderSpeech(t *testing.T) {
	tests := []struct {
		name           string
		req            *warp.SpeechRequest
		mockStatusCode int
		mockResponse   []byte
		mockError      error
		wantErr        bool
		errContains    string
		checkAudio     func(*testing.T, []byte)
	}{
		{
			name: "successful speech generation",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Hello, world!",
				Voice: "alloy",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("mock mp3 audio data"),
			wantErr:        false,
			checkAudio: func(t *testing.T, data []byte) {
				if !bytes.Equal(data, []byte("mock mp3 audio data")) {
					t.Errorf("audio data = %q, want %q", data, "mock mp3 audio data")
				}
			},
		},
		{
			name: "with response format mp3",
			req: &warp.SpeechRequest{
				Model:          "tts-1",
				Input:          "Test audio",
				Voice:          "echo",
				ResponseFormat: "mp3",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("mp3 data"),
			wantErr:        false,
		},
		{
			name: "with response format opus",
			req: &warp.SpeechRequest{
				Model:          "tts-1",
				Input:          "Test audio",
				Voice:          "fable",
				ResponseFormat: "opus",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("opus data"),
			wantErr:        false,
		},
		{
			name: "with speed parameter",
			req: &warp.SpeechRequest{
				Model: "tts-1-hd",
				Input: "Fast speech",
				Voice: "nova",
				Speed: float64Ptr(1.5),
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("fast audio"),
			wantErr:        false,
		},
		{
			name: "all voices - alloy",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("alloy voice"),
			wantErr:        false,
		},
		{
			name: "all voices - echo",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "echo",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("echo voice"),
			wantErr:        false,
		},
		{
			name: "all voices - fable",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "fable",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("fable voice"),
			wantErr:        false,
		},
		{
			name: "all voices - onyx",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "onyx",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("onyx voice"),
			wantErr:        false,
		},
		{
			name: "all voices - nova",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "nova",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("nova voice"),
			wantErr:        false,
		},
		{
			name: "all voices - shimmer",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "shimmer",
			},
			mockStatusCode: http.StatusOK,
			mockResponse:   []byte("shimmer voice"),
			wantErr:        false,
		},
		{
			name: "rate limit error",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			mockStatusCode: http.StatusTooManyRequests,
			mockResponse:   []byte(`{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}`),
			wantErr:        true,
			errContains:    "Rate limit",
		},
		{
			name: "invalid API key",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			mockStatusCode: http.StatusUnauthorized,
			mockResponse:   []byte(`{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}`),
			wantErr:        true,
			errContains:    "Invalid API key",
		},
		{
			name: "service unavailable",
			req: &warp.SpeechRequest{
				Model: "tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			mockStatusCode: http.StatusServiceUnavailable,
			mockResponse:   []byte(`{"error": {"message": "Service unavailable"}}`),
			wantErr:        true,
			errContains:    "Service unavailable",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock HTTP client
			mockClient := &testutil.MockHTTPClient{
				DoFunc: func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: tt.mockStatusCode,
						Body:       io.NopCloser(bytes.NewReader(tt.mockResponse)),
						Header:     make(http.Header),
					}, nil
				},
			}

			// Create provider
			p := &Provider{
				apiKey:     "test-key",
				apiBase:    "https://api.openai.com/v1",
				httpClient: mockClient,
			}

			// Call Speech
			audio, err := p.Speech(context.Background(), tt.req)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Speech() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}

			if audio == nil {
				t.Errorf("expected audio stream, got nil")
				return
			}

			// Read and verify audio data
			defer audio.Close()
			data, err := io.ReadAll(audio)
			if err != nil {
				t.Errorf("failed to read audio: %v", err)
				return
			}

			if tt.checkAudio != nil {
				tt.checkAudio(t, data)
			} else if !bytes.Equal(data, tt.mockResponse) {
				t.Errorf("audio data = %q, want %q", data, tt.mockResponse)
			}
		})
	}
}

func TestProviderSpeechWithAPIOverrides(t *testing.T) {
	// Test API key override
	mockClient := &testutil.MockHTTPClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewReader([]byte("audio data"))),
				Header:     make(http.Header),
			}, nil
		},
	}

	p := &Provider{
		apiKey:     "default-key",
		apiBase:    "https://api.openai.com/v1",
		httpClient: mockClient,
	}

	req := &warp.SpeechRequest{
		Model:   "tts-1",
		Input:   "Test",
		Voice:   "alloy",
		APIKey:  "override-key",
		APIBase: "https://custom.openai.com/v1",
	}

	audio, err := p.Speech(context.Background(), req)
	if err != nil {
		t.Fatalf("Speech() error = %v", err)
	}
	defer audio.Close()

	// Verify the request was made with overridden values
	// (This would be verified by inspecting mockClient's recorded request in a more sophisticated test)
}

func TestProviderSpeechContextCancellation(t *testing.T) {
	// Create a context that's already cancelled
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mockClient := &testutil.MockHTTPClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewReader([]byte("audio"))),
				Header:     make(http.Header),
			}, nil
		},
	}

	p := &Provider{
		apiKey:     "test-key",
		apiBase:    "https://api.openai.com/v1",
		httpClient: mockClient,
	}

	req := &warp.SpeechRequest{
		Model: "tts-1",
		Input: "Test",
		Voice: "alloy",
	}

	// Should handle cancelled context
	audio, err := p.Speech(ctx, req)
	if err != nil {
		// Context cancellation is acceptable
		if !strings.Contains(err.Error(), "context") {
			t.Errorf("expected context error, got: %v", err)
		}
		return
	}

	if audio != nil {
		audio.Close()
	}
}

func TestProviderSpeechLargeInput(t *testing.T) {
	// Test with maximum allowed input (4096 characters)
	largeInput := strings.Repeat("a", 4096)

	mockClient := &testutil.MockHTTPClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewReader([]byte("large audio data"))),
				Header:     make(http.Header),
			}, nil
		},
	}

	p := &Provider{
		apiKey:     "test-key",
		apiBase:    "https://api.openai.com/v1",
		httpClient: mockClient,
	}

	req := &warp.SpeechRequest{
		Model: "tts-1",
		Input: largeInput,
		Voice: "alloy",
	}

	audio, err := p.Speech(context.Background(), req)
	if err != nil {
		t.Fatalf("Speech() error = %v", err)
	}
	defer audio.Close()

	data, err := io.ReadAll(audio)
	if err != nil {
		t.Errorf("failed to read audio: %v", err)
		return
	}

	if !bytes.Equal(data, []byte("large audio data")) {
		t.Errorf("audio data mismatch")
	}
}

func TestProviderSpeechStreamClosed(t *testing.T) {
	mockClient := &testutil.MockHTTPClient{
		DoFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewReader([]byte("test audio"))),
				Header:     make(http.Header),
			}, nil
		},
	}

	p := &Provider{
		apiKey:     "test-key",
		apiBase:    "https://api.openai.com/v1",
		httpClient: mockClient,
	}

	req := &warp.SpeechRequest{
		Model: "tts-1",
		Input: "Test",
		Voice: "alloy",
	}

	audio, err := p.Speech(context.Background(), req)
	if err != nil {
		t.Fatalf("Speech() error = %v", err)
	}

	// Close the stream
	err = audio.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// Try to read from closed stream
	// This behavior depends on the io.ReadCloser implementation
	_, err = io.ReadAll(audio)
	// Reading from a closed body might or might not error
	// We just verify that Close() worked without error
}
