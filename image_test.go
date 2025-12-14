package warp

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mockImageProvider is a mock provider that implements ImageGeneration, ImageEdit, and ImageVariation
type mockImageProvider struct {
	name              string
	imageResp         *ImageGenerationResponse
	imageErr          error
	imageEditResp     *ImageGenerationResponse
	imageEditErr      error
	imageVariationResp *ImageGenerationResponse
	imageVariationErr  error
}

func (m *mockImageProvider) Name() string {
	return m.name
}

func (m *mockImageProvider) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockImageProvider) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockImageProvider) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockImageProvider) ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	if m.imageErr != nil {
		return nil, m.imageErr
	}
	return m.imageResp, nil
}

func (m *mockImageProvider) ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageGenerationResponse, error) {
	if m.imageEditErr != nil {
		return nil, m.imageEditErr
	}
	return m.imageEditResp, nil
}

func (m *mockImageProvider) ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageGenerationResponse, error) {
	if m.imageVariationErr != nil {
		return nil, m.imageVariationErr
	}
	return m.imageVariationResp, nil
}

func (m *mockImageProvider) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockImageProvider) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockImageProvider) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockImageProvider) Supports() interface{} {
	return struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		ImageEdit       bool
		ImageVariation  bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
	}{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: true,
		ImageEdit:       true,
		ImageVariation:  true,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true,
		Vision:          false,
		JSON:            true,
	}
}

// mockNoImageProvider is a mock provider that does NOT implement ImageGeneration
type mockNoImageProvider struct {
	name string
}

func (m *mockNoImageProvider) Name() string {
	return m.name
}

func (m *mockNoImageProvider) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockNoImageProvider) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockNoImageProvider) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockNoImageProvider) ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, fmt.Errorf("provider %q does not support image generation", m.name)
}

func (m *mockNoImageProvider) ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageGenerationResponse, error) {
	return nil, fmt.Errorf("provider %q does not support image editing", m.name)
}

func (m *mockNoImageProvider) ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageGenerationResponse, error) {
	return nil, fmt.Errorf("provider %q does not support image variation", m.name)
}

func (m *mockNoImageProvider) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockNoImageProvider) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockNoImageProvider) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

func (m *mockNoImageProvider) Supports() interface{} {
	return struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		ImageEdit       bool
		ImageVariation  bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
	}{
		Completion:      true,
		Streaming:       true,
		Embedding:       true,
		ImageGeneration: false,
		ImageEdit:       false,
		ImageVariation:  false,
		Transcription:   false,
		Speech:          false,
		Moderation:      false,
		FunctionCalling: true,
		Vision:          false,
		JSON:            true,
	}
}

func TestImageGeneration(t *testing.T) {
	tests := []struct {
		name      string
		req       *ImageGenerationRequest
		provider  Provider
		wantErr   bool
		errString string
	}{
		{
			name: "successful image generation",
			req: &ImageGenerationRequest{
				Model:  "test/dall-e-3",
				Prompt: "A cute baby sea otter",
			},
			provider: &mockImageProvider{
				name: "test",
				imageResp: &ImageGenerationResponse{
					Created: 1234567890,
					Data: []ImageData{
						{
							URL:           "https://example.com/image.png",
							RevisedPrompt: "A detailed image of a cute baby sea otter",
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "nil request",
			req:  nil,
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "request cannot be nil",
		},
		{
			name: "missing model",
			req: &ImageGenerationRequest{
				Prompt: "test",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "model is required",
		},
		{
			name: "missing prompt",
			req: &ImageGenerationRequest{
				Model: "test/dall-e-3",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "prompt is required",
		},
		{
			name: "invalid model format",
			req: &ImageGenerationRequest{
				Model:  "invalid-model",
				Prompt: "test",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "invalid model format",
		},
		{
			name: "provider not found",
			req: &ImageGenerationRequest{
				Model:  "nonexistent/dall-e-3",
				Prompt: "test",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "provider \"nonexistent\" not found",
		},
		{
			name: "provider does not support image generation",
			req: &ImageGenerationRequest{
				Model:  "test/dall-e-3",
				Prompt: "test",
			},
			provider: &mockNoImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "does not support image generation",
		},
		{
			name: "provider returns error",
			req: &ImageGenerationRequest{
				Model:  "test/dall-e-3",
				Prompt: "test",
			},
			provider: &mockImageProvider{
				name:     "test",
				imageErr: fmt.Errorf("provider error"),
			},
			wantErr:   true,
			errString: "provider error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client
			c, err := NewClient()
			if err != nil {
				t.Fatalf("failed to create client: %v", err)
			}
			defer c.Close()

			// Register provider
			if tt.provider != nil {
				if err := c.RegisterProvider(tt.provider); err != nil {
					t.Fatalf("failed to register provider: %v", err)
				}
			}

			// Call ImageGeneration
			resp, err := c.ImageGeneration(context.Background(), tt.req)

			// Check error
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errString)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Validate response
			if resp == nil {
				t.Fatal("response is nil")
			}

			if len(resp.Data) == 0 {
				t.Error("response has no images")
			}

			if resp.Provider != tt.provider.Name() {
				t.Errorf("provider = %q, want %q", resp.Provider, tt.provider.Name())
			}
		})
	}
}

func TestImageData_SaveToFile(t *testing.T) {
	// Create temporary directory for test files
	tempDir := t.TempDir()

	// Create a test HTTP server for URL downloads
	testImageData := []byte("fake image data")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write(testImageData)
	}))
	defer server.Close()

	// Create base64 encoded test data
	base64Data := base64.StdEncoding.EncodeToString(testImageData)

	tests := []struct {
		name      string
		imageData ImageData
		wantErr   bool
		errString string
	}{
		{
			name: "save from URL",
			imageData: ImageData{
				URL: server.URL + "/image.png",
			},
			wantErr: false,
		},
		{
			name: "save from base64",
			imageData: ImageData{
				B64JSON: base64Data,
			},
			wantErr: false,
		},
		{
			name:      "no image data",
			imageData: ImageData{},
			wantErr:   true,
			errString: "no image data available",
		},
		{
			name: "invalid base64",
			imageData: ImageData{
				B64JSON: "not-valid-base64!@#$",
			},
			wantErr:   true,
			errString: "failed to decode base64 image",
		},
		{
			name: "invalid URL",
			imageData: ImageData{
				URL: "http://invalid-domain-that-does-not-exist.test/image.png",
			},
			wantErr:   true,
			errString: "failed to download image",
		},
		{
			name: "HTTP error",
			imageData: ImageData{
				URL: server.URL + "/notfound",
			},
			wantErr: false, // Server returns 200 for all requests in this test
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(tempDir, tt.name+".png")

			err := tt.imageData.SaveToFile(context.Background(), path)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errString)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify file was created
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("failed to read saved file: %v", err)
			}

			if len(data) == 0 {
				t.Error("saved file is empty")
			}
		})
	}
}

func TestImageData_SaveToFile_ContextCancellation(t *testing.T) {
	// Create a server that delays response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	img := ImageData{
		URL: server.URL + "/image.png",
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	tempDir := t.TempDir()
	path := filepath.Join(tempDir, "cancelled.png")

	err := img.SaveToFile(ctx, path)
	if err == nil {
		t.Error("expected error for cancelled context")
	}
}

func TestDownloadImage(t *testing.T) {
	testData := []byte("test image data")

	tests := []struct {
		name       string
		statusCode int
		data       []byte
		wantErr    bool
		errString  string
	}{
		{
			name:       "successful download",
			statusCode: http.StatusOK,
			data:       testData,
			wantErr:    false,
		},
		{
			name:       "not found",
			statusCode: http.StatusNotFound,
			data:       []byte("not found"),
			wantErr:    true,
			errString:  "HTTP 404",
		},
		{
			name:       "server error",
			statusCode: http.StatusInternalServerError,
			data:       []byte("error"),
			wantErr:    true,
			errString:  "HTTP 500",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Write(tt.data)
			}))
			defer server.Close()

			data, err := downloadImage(context.Background(), server.URL)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errString)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if string(data) != string(tt.data) {
				t.Errorf("data = %q, want %q", string(data), string(tt.data))
			}
		})
	}
}

func TestImageEdit(t *testing.T) {
	tests := []struct {
		name      string
		req       *ImageEditRequest
		provider  Provider
		wantErr   bool
		errString string
	}{
		{
			name: "successful image edit without mask",
			req: &ImageEditRequest{
				Model:         "test/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
				imageEditResp: &ImageGenerationResponse{
					Created: 1234567890,
					Data: []ImageData{
						{
							URL: "https://example.com/edited-image.png",
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "successful image edit with mask",
			req: &ImageEditRequest{
				Model:         "test/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Mask:          strings.NewReader("fake mask data"),
				MaskFilename:  "mask.png",
				Prompt:        "Change the background",
			},
			provider: &mockImageProvider{
				name: "test",
				imageEditResp: &ImageGenerationResponse{
					Created: 1234567890,
					Data: []ImageData{
						{
							URL: "https://example.com/edited-image.png",
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "nil request",
			req:  nil,
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "request cannot be nil",
		},
		{
			name: "missing model",
			req: &ImageEditRequest{
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "model is required",
		},
		{
			name: "missing image",
			req: &ImageEditRequest{
				Model:         "test/dall-e-2",
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "image is required",
		},
		{
			name: "missing image filename",
			req: &ImageEditRequest{
				Model:  "test/dall-e-2",
				Image:  strings.NewReader("fake image data"),
				Prompt: "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "image filename is required",
		},
		{
			name: "missing prompt",
			req: &ImageEditRequest{
				Model:         "test/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "prompt is required",
		},
		{
			name: "mask without filename",
			req: &ImageEditRequest{
				Model:         "test/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Mask:          strings.NewReader("fake mask data"),
				Prompt:        "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "mask filename is required when mask is provided",
		},
		{
			name: "invalid model format",
			req: &ImageEditRequest{
				Model:         "invalid-model",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "invalid model format",
		},
		{
			name: "provider not found",
			req: &ImageEditRequest{
				Model:         "unknown/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "provider \"unknown\" not found",
		},
		{
			name: "provider does not support image editing",
			req: &ImageEditRequest{
				Model:         "noimage/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			provider: &mockNoImageProvider{
				name: "noimage",
			},
			wantErr:   true,
			errString: "provider \"noimage\" does not support image editing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				config:    defaultConfig(),
				providers: make(map[string]Provider),
			}

			if tt.provider != nil {
				c.providers[tt.provider.Name()] = tt.provider
			}

			resp, err := c.ImageEdit(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errString)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if resp == nil {
				t.Fatal("response is nil")
			}

			if len(resp.Data) == 0 {
				t.Error("expected at least one image in response")
			}

			if resp.Provider != "test" {
				t.Errorf("provider = %q, want %q", resp.Provider, "test")
			}

			if resp.Model != "dall-e-2" {
				t.Errorf("model = %q, want %q", resp.Model, "dall-e-2")
			}
		})
	}
}

func TestImageVariation(t *testing.T) {
	tests := []struct {
		name      string
		req       *ImageVariationRequest
		provider  Provider
		wantErr   bool
		errString string
	}{
		{
			name: "successful image variation with default model",
			req: &ImageVariationRequest{
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			provider: &mockImageProvider{
				name: "openai",
				imageVariationResp: &ImageGenerationResponse{
					Created: 1234567890,
					Data: []ImageData{
						{
							URL: "https://example.com/variation1.png",
						},
						{
							URL: "https://example.com/variation2.png",
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "successful image variation with explicit model",
			req: &ImageVariationRequest{
				Model:         "test/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				N:             IntPtr(3),
				Size:          "512x512",
			},
			provider: &mockImageProvider{
				name: "test",
				imageVariationResp: &ImageGenerationResponse{
					Created: 1234567890,
					Data: []ImageData{
						{
							URL: "https://example.com/variation1.png",
						},
						{
							URL: "https://example.com/variation2.png",
						},
						{
							URL: "https://example.com/variation3.png",
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "nil request",
			req:  nil,
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "request cannot be nil",
		},
		{
			name: "missing image",
			req: &ImageVariationRequest{
				Model:         "test/dall-e-2",
				ImageFilename: "original.png",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "image is required",
		},
		{
			name: "missing image filename",
			req: &ImageVariationRequest{
				Model: "test/dall-e-2",
				Image: strings.NewReader("fake image data"),
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "image filename is required",
		},
		{
			name: "invalid model format",
			req: &ImageVariationRequest{
				Model:         "invalid-model",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "invalid model format",
		},
		{
			name: "provider not found",
			req: &ImageVariationRequest{
				Model:         "unknown/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			provider: &mockImageProvider{
				name: "test",
			},
			wantErr:   true,
			errString: "provider \"unknown\" not found",
		},
		{
			name: "provider does not support image variation",
			req: &ImageVariationRequest{
				Model:         "noimage/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			provider: &mockNoImageProvider{
				name: "noimage",
			},
			wantErr:   true,
			errString: "provider \"noimage\" does not support image variation",
		},
		{
			name: "provider returns error",
			req: &ImageVariationRequest{
				Model:         "test/dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			provider: &mockImageProvider{
				name:              "test",
				imageVariationErr: fmt.Errorf("provider error"),
			},
			wantErr:   true,
			errString: "provider error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &client{
				config:    defaultConfig(),
				providers: make(map[string]Provider),
			}

			if tt.provider != nil {
				c.providers[tt.provider.Name()] = tt.provider
			}

			resp, err := c.ImageVariation(context.Background(), tt.req)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if tt.errString != "" && !strings.Contains(err.Error(), tt.errString) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errString)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if resp == nil {
				t.Fatal("response is nil")
			}

			if len(resp.Data) == 0 {
				t.Error("expected at least one image in response")
			}

			if resp.Provider != tt.provider.Name() {
				t.Errorf("provider = %q, want %q", resp.Provider, tt.provider.Name())
			}

			// For the default model case, verify the model was set to dall-e-2
			if tt.req != nil && tt.req.Model == "" {
				if resp.Model != "dall-e-2" {
					t.Errorf("model = %q, want %q (default)", resp.Model, "dall-e-2")
				}
			} else if tt.req != nil && strings.Contains(tt.req.Model, "/") {
				expectedModel := strings.Split(tt.req.Model, "/")[1]
				if resp.Model != expectedModel {
					t.Errorf("model = %q, want %q", resp.Model, expectedModel)
				}
			}
		})
	}
}
