package warp

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

// mockTranscriptionProvider implements Provider interface for testing
type mockTranscriptionProvider struct {
	name                  string
	transcriptionResp     *TranscriptionResponse
	transcriptionError    error
	supportsTranscription bool
}

func (m *mockTranscriptionProvider) Name() string {
	return m.name
}

func (m *mockTranscriptionProvider) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockTranscriptionProvider) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	return nil, errors.New("not implemented")
}

func (m *mockTranscriptionProvider) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockTranscriptionProvider) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	if m.transcriptionError != nil {
		return nil, m.transcriptionError
	}
	return m.transcriptionResp, nil
}

func (m *mockTranscriptionProvider) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	return nil, errors.New("not implemented")
}

func (m *mockTranscriptionProvider) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockTranscriptionProvider) Supports() interface{} {
	// Return a struct that matches the type assertion in audio.go
	return struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
	}{
		Transcription: m.supportsTranscription,
		Speech:        false,
	}
}

func TestClientTranscription(t *testing.T) {
	tests := []struct {
		name                  string
		req                   *TranscriptionRequest
		mockResp              *TranscriptionResponse
		mockError             error
		supportsTranscription bool
		wantErr               bool
		wantErrMsg            string
		checkResponse         func(*testing.T, *TranscriptionResponse)
	}{
		{
			name: "successful transcription",
			req: &TranscriptionRequest{
				Model:    "openai/whisper-1",
				File:     strings.NewReader("fake audio data"),
				Filename: "test.mp3",
				Language: "en",
			},
			mockResp: &TranscriptionResponse{
				Text:     "Hello, world!",
				Language: "en",
				Duration: 2.5,
			},
			supportsTranscription: true,
			wantErr:               false,
			checkResponse: func(t *testing.T, resp *TranscriptionResponse) {
				if resp.Text != "Hello, world!" {
					t.Errorf("Text = %q, want %q", resp.Text, "Hello, world!")
				}
				if resp.Language != "en" {
					t.Errorf("Language = %q, want %q", resp.Language, "en")
				}
				if resp.Duration != 2.5 {
					t.Errorf("Duration = %f, want %f", resp.Duration, 2.5)
				}
				if resp.Provider != "openai" {
					t.Errorf("Provider = %q, want %q", resp.Provider, "openai")
				}
				if resp.Model != "whisper-1" {
					t.Errorf("Model = %q, want %q", resp.Model, "whisper-1")
				}
			},
		},
		{
			name:       "nil request",
			req:        nil,
			wantErr:    true,
			wantErrMsg: "request cannot be nil",
		},
		{
			name: "missing model",
			req: &TranscriptionRequest{
				File:     strings.NewReader("data"),
				Filename: "test.mp3",
			},
			wantErr:    true,
			wantErrMsg: "model is required",
		},
		{
			name: "missing file",
			req: &TranscriptionRequest{
				Model:    "openai/whisper-1",
				Filename: "test.mp3",
			},
			wantErr:    true,
			wantErrMsg: "file is required",
		},
		{
			name: "missing filename",
			req: &TranscriptionRequest{
				Model: "openai/whisper-1",
				File:  strings.NewReader("data"),
			},
			wantErr:    true,
			wantErrMsg: "filename is required",
		},
		{
			name: "invalid model format",
			req: &TranscriptionRequest{
				Model:    "invalid-model",
				File:     strings.NewReader("data"),
				Filename: "test.mp3",
			},
			wantErr:    true,
			wantErrMsg: "invalid model format",
		},
		{
			name: "provider not found",
			req: &TranscriptionRequest{
				Model:    "unknown/model",
				File:     strings.NewReader("data"),
				Filename: "test.mp3",
			},
			wantErr:    true,
			wantErrMsg: "not found",
		},
		{
			name: "provider doesn't support transcription",
			req: &TranscriptionRequest{
				Model:    "openai/whisper-1",
				File:     strings.NewReader("data"),
				Filename: "test.mp3",
			},
			supportsTranscription: false,
			wantErr:               true,
			wantErrMsg:            "does not support transcription",
		},
		{
			name: "provider returns error",
			req: &TranscriptionRequest{
				Model:    "openai/whisper-1",
				File:     strings.NewReader("data"),
				Filename: "test.mp3",
			},
			mockError:             errors.New("provider error"),
			supportsTranscription: true,
			wantErr:               true,
			wantErrMsg:            "provider error",
		},
		{
			name: "with verbose_json response format",
			req: &TranscriptionRequest{
				Model:                  "openai/whisper-1",
				File:                   strings.NewReader("audio data"),
				Filename:               "test.mp3",
				ResponseFormat:         "verbose_json",
				TimestampGranularities: []string{"word", "segment"},
			},
			mockResp: &TranscriptionResponse{
				Text:     "The quick brown fox",
				Language: "en",
				Duration: 5.0,
				Words: []Word{
					{Word: "The", Start: 0.0, End: 0.5},
					{Word: "quick", Start: 0.5, End: 1.0},
				},
				Segments: []Segment{
					{
						ID:    0,
						Start: 0.0,
						End:   2.0,
						Text:  "The quick brown fox",
					},
				},
			},
			supportsTranscription: true,
			wantErr:               false,
			checkResponse: func(t *testing.T, resp *TranscriptionResponse) {
				if len(resp.Words) != 2 {
					t.Errorf("len(Words) = %d, want 2", len(resp.Words))
				}
				if len(resp.Segments) != 1 {
					t.Errorf("len(Segments) = %d, want 1", len(resp.Segments))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client
			client := &client{
				config: &ClientConfig{
					DefaultTimeout:  30 * time.Second,
					MaxRetries:      3,
					RetryDelay:      time.Second,
					RetryMultiplier: 2.0,
				},
				providers: make(map[string]Provider),
			}

			// Register mock provider if needed
			if tt.req != nil && tt.req.Model != "" {
				providerName, _, err := parseModel(tt.req.Model)
				if err == nil && providerName != "unknown" {
					mockProvider := &mockTranscriptionProvider{
						name:                  providerName,
						transcriptionResp:     tt.mockResp,
						transcriptionError:    tt.mockError,
						supportsTranscription: tt.supportsTranscription,
					}
					client.RegisterProvider(mockProvider)
				}
			}

			// Call Transcription
			ctx := context.Background()
			resp, err := client.Transcription(ctx, tt.req)

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

			if tt.checkResponse != nil {
				tt.checkResponse(t, resp)
			}
		})
	}
}

func TestClientTranscriptionWithTimeout(t *testing.T) {
	t.Skip("Skipping timeout test for now - requires more complex mock setup")
	// TODO: Implement this test with proper timeout handling
}

func TestClientTranscriptionWithLargeFile(t *testing.T) {
	// Create a large mock audio file (1MB)
	largeData := bytes.Repeat([]byte("A"), 1024*1024)
	reader := bytes.NewReader(largeData)

	client := &client{
		config: &ClientConfig{
			DefaultTimeout:  30 * time.Second,
			MaxRetries:      0,
			RetryDelay:      time.Second,
			RetryMultiplier: 2.0,
		},
		providers: make(map[string]Provider),
	}

	// Create a variable to track if file was read
	fileReadSize := 0

	mockProvider := &mockTranscriptionProvider{
		name: "openai",
		transcriptionResp: &TranscriptionResponse{
			Text:     "Transcribed large audio file",
			Duration: 60.0,
		},
		transcriptionError:    nil,
		supportsTranscription: true,
	}

	// Override with custom function that verifies file read
	mockProvider.transcriptionError = nil
	mockProvider.transcriptionResp = &TranscriptionResponse{
		Text:     "Transcribed large audio file",
		Duration: 60.0,
	}

	client.RegisterProvider(mockProvider)

	ctx := context.Background()
	req := &TranscriptionRequest{
		Model:    "openai/whisper-1",
		File:     reader,
		Filename: "large_audio.mp3",
	}

	resp, err := client.Transcription(ctx, req)
	if err != nil {
		t.Fatalf("Transcription() error = %v", err)
	}

	if resp.Text != "Transcribed large audio file" {
		t.Errorf("Text = %q, want %q", resp.Text, "Transcribed large audio file")
	}

	// Note: We can't verify file was read in this test because the mock
	// doesn't actually consume the reader. That's tested in integration tests.
	_ = fileReadSize
}

// mockSpeechProvider implements Provider interface for testing speech
type mockSpeechProvider struct {
	name           string
	audioData      []byte
	speechError    error
	supportsSpeech bool
}

func (m *mockSpeechProvider) Name() string {
	return m.name
}

func (m *mockSpeechProvider) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockSpeechProvider) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	return nil, errors.New("not implemented")
}

func (m *mockSpeechProvider) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockSpeechProvider) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockSpeechProvider) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	if m.speechError != nil {
		return nil, m.speechError
	}
	return io.NopCloser(bytes.NewReader(m.audioData)), nil
}

func (m *mockSpeechProvider) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockSpeechProvider) Supports() interface{} {
	return struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
	}{
		Speech: m.supportsSpeech,
	}
}

func TestClientSpeech(t *testing.T) {
	tests := []struct {
		name           string
		req            *SpeechRequest
		audioData      []byte
		speechError    error
		supportsSpeech bool
		wantErr        bool
		wantErrMsg     string
		checkAudio     func(*testing.T, []byte)
	}{
		{
			name: "successful speech generation",
			req: &SpeechRequest{
				Model: "openai/tts-1",
				Input: "Hello, world!",
				Voice: "alloy",
			},
			audioData:      []byte("mock audio data"),
			supportsSpeech: true,
			wantErr:        false,
			checkAudio: func(t *testing.T, data []byte) {
				if !bytes.Equal(data, []byte("mock audio data")) {
					t.Errorf("audio data = %q, want %q", data, "mock audio data")
				}
			},
		},
		{
			name: "with response format and speed",
			req: &SpeechRequest{
				Model:          "openai/tts-1-hd",
				Input:          "Fast speech test",
				Voice:          "nova",
				ResponseFormat: "mp3",
				Speed:          float64Ptr(1.5),
			},
			audioData:      []byte("fast mp3 audio"),
			supportsSpeech: true,
			wantErr:        false,
			checkAudio: func(t *testing.T, data []byte) {
				if !bytes.Equal(data, []byte("fast mp3 audio")) {
					t.Errorf("audio data = %q, want %q", data, "fast mp3 audio")
				}
			},
		},
		{
			name:       "nil request",
			req:        nil,
			wantErr:    true,
			wantErrMsg: "request cannot be nil",
		},
		{
			name: "missing model",
			req: &SpeechRequest{
				Input: "Test",
				Voice: "alloy",
			},
			wantErr:    true,
			wantErrMsg: "model is required",
		},
		{
			name: "missing input",
			req: &SpeechRequest{
				Model: "openai/tts-1",
				Voice: "alloy",
			},
			wantErr:    true,
			wantErrMsg: "input text is required",
		},
		{
			name: "missing voice",
			req: &SpeechRequest{
				Model: "openai/tts-1",
				Input: "Test",
			},
			wantErr:    true,
			wantErrMsg: "voice is required",
		},
		{
			name: "invalid model format",
			req: &SpeechRequest{
				Model: "invalid-model",
				Input: "Test",
				Voice: "alloy",
			},
			wantErr:    true,
			wantErrMsg: "invalid model format",
		},
		{
			name: "provider not found",
			req: &SpeechRequest{
				Model: "unknown/tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			wantErr:    true,
			wantErrMsg: "not found",
		},
		{
			name: "provider doesn't support speech",
			req: &SpeechRequest{
				Model: "openai/tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			supportsSpeech: false,
			wantErr:        true,
			wantErrMsg:     "does not support text-to-speech",
		},
		{
			name: "provider returns error",
			req: &SpeechRequest{
				Model: "openai/tts-1",
				Input: "Test",
				Voice: "alloy",
			},
			speechError:    errors.New("TTS service error"),
			supportsSpeech: true,
			wantErr:        true,
			wantErrMsg:     "TTS service error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client
			client := &client{
				config: &ClientConfig{
					DefaultTimeout:  30 * time.Second,
					MaxRetries:      3,
					RetryDelay:      time.Second,
					RetryMultiplier: 2.0,
				},
				providers: make(map[string]Provider),
			}

			// Register mock provider if needed
			if tt.req != nil && tt.req.Model != "" {
				providerName, _, err := parseModel(tt.req.Model)
				if err == nil && providerName != "unknown" {
					mockProvider := &mockSpeechProvider{
						name:           providerName,
						audioData:      tt.audioData,
						speechError:    tt.speechError,
						supportsSpeech: tt.supportsSpeech,
					}
					client.RegisterProvider(mockProvider)
				}
			}

			// Call Speech
			ctx := context.Background()
			audio, err := client.Speech(ctx, tt.req)

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Speech() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				if tt.wantErrMsg != "" && !strings.Contains(err.Error(), tt.wantErrMsg) {
					t.Errorf("error message = %q, want to contain %q", err.Error(), tt.wantErrMsg)
				}
				return
			}

			// Check audio stream
			if audio == nil {
				t.Fatal("audio stream is nil")
			}
			defer audio.Close()

			// Read audio data
			data, err := io.ReadAll(audio)
			if err != nil {
				t.Errorf("failed to read audio: %v", err)
				return
			}

			if tt.checkAudio != nil {
				tt.checkAudio(t, data)
			}
		})
	}
}

func TestClientSpeechWithContextCancellation(t *testing.T) {
	client := &client{
		config: &ClientConfig{
			DefaultTimeout:  30 * time.Second,
			MaxRetries:      0,
			RetryDelay:      time.Second,
			RetryMultiplier: 2.0,
		},
		providers: make(map[string]Provider),
	}

	mockProvider := &mockSpeechProvider{
		name:           "openai",
		audioData:      []byte("test audio"),
		supportsSpeech: true,
	}
	client.RegisterProvider(mockProvider)

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := &SpeechRequest{
		Model: "openai/tts-1",
		Input: "Test",
		Voice: "alloy",
	}

	// Speech should handle context cancellation
	audio, err := client.Speech(ctx, req)
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

func float64Ptr(f float64) *float64 {
	return &f
}
