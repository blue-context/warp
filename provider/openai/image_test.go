package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/blue-context/warp"
)

func TestProvider_ImageGeneration(t *testing.T) {
	tests := []struct {
		name           string
		req            *warp.ImageGenerationRequest
		mockStatusCode int
		mockResponse   *warp.ImageGenerationResponse
		mockError      map[string]any
		wantErr        bool
		errString      string
	}{
		{
			name: "successful DALL-E 3 generation",
			req: &warp.ImageGenerationRequest{
				Model:  "dall-e-3",
				Prompt: "A cute baby sea otter",
				Size:   "1024x1024",
			},
			mockStatusCode: http.StatusOK,
			mockResponse: &warp.ImageGenerationResponse{
				Created: 1234567890,
				Data: []warp.ImageData{
					{
						URL:           "https://example.com/image.png",
						RevisedPrompt: "A detailed image of a cute baby sea otter floating on its back in the ocean",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "successful DALL-E 2 with base64",
			req: &warp.ImageGenerationRequest{
				Model:          "dall-e-2",
				Prompt:         "Abstract art",
				Size:           "512x512",
				ResponseFormat: "b64_json",
			},
			mockStatusCode: http.StatusOK,
			mockResponse: &warp.ImageGenerationResponse{
				Created: 1234567890,
				Data: []warp.ImageData{
					{
						B64JSON: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "with quality and style",
			req: &warp.ImageGenerationRequest{
				Model:   "dall-e-3",
				Prompt:  "A futuristic city",
				Size:    "1792x1024",
				Quality: "hd",
				Style:   "vivid",
			},
			mockStatusCode: http.StatusOK,
			mockResponse: &warp.ImageGenerationResponse{
				Created: 1234567890,
				Data: []warp.ImageData{
					{
						URL: "https://example.com/image.png",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "with user parameter",
			req: &warp.ImageGenerationRequest{
				Model:  "dall-e-3",
				Prompt: "test",
				User:   "user-123",
			},
			mockStatusCode: http.StatusOK,
			mockResponse: &warp.ImageGenerationResponse{
				Created: 1234567890,
				Data: []warp.ImageData{
					{
						URL: "https://example.com/image.png",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple images DALL-E 2",
			req: &warp.ImageGenerationRequest{
				Model:  "dall-e-2",
				Prompt: "test",
				N:      intPtr(4),
			},
			mockStatusCode: http.StatusOK,
			mockResponse: &warp.ImageGenerationResponse{
				Created: 1234567890,
				Data: []warp.ImageData{
					{URL: "https://example.com/image1.png"},
					{URL: "https://example.com/image2.png"},
					{URL: "https://example.com/image3.png"},
					{URL: "https://example.com/image4.png"},
				},
			},
			wantErr: false,
		},
		{
			name:           "nil request",
			req:            nil,
			mockStatusCode: http.StatusOK,
			wantErr:        true,
			errString:      "request cannot be nil",
		},
		{
			name: "missing model",
			req: &warp.ImageGenerationRequest{
				Prompt: "test",
			},
			mockStatusCode: http.StatusOK,
			wantErr:        true,
			errString:      "model is required",
		},
		{
			name: "missing prompt",
			req: &warp.ImageGenerationRequest{
				Model: "dall-e-3",
			},
			mockStatusCode: http.StatusOK,
			wantErr:        true,
			errString:      "prompt is required",
		},
		{
			name: "rate limit error",
			req: &warp.ImageGenerationRequest{
				Model:  "dall-e-3",
				Prompt: "test",
			},
			mockStatusCode: http.StatusTooManyRequests,
			mockError: map[string]any{
				"error": map[string]any{
					"message": "Rate limit exceeded",
					"type":    "rate_limit_error",
				},
			},
			wantErr:   true,
			errString: "Rate limit",
		},
		{
			name: "invalid request error",
			req: &warp.ImageGenerationRequest{
				Model:  "dall-e-3",
				Prompt: "test",
			},
			mockStatusCode: http.StatusBadRequest,
			mockError: map[string]any{
				"error": map[string]any{
					"message": "Invalid size for DALL-E 3",
					"type":    "invalid_request_error",
				},
			},
			wantErr:   true,
			errString: "Invalid",
		},
		{
			name: "content policy violation",
			req: &warp.ImageGenerationRequest{
				Model:  "dall-e-3",
				Prompt: "inappropriate content",
			},
			mockStatusCode: http.StatusBadRequest,
			mockError: map[string]any{
				"error": map[string]any{
					"message": "Your request was rejected as a result of our safety system",
					"type":    "invalid_request_error",
				},
			},
			wantErr:   true,
			errString: "safety system",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request method and path
				if r.Method != "POST" {
					t.Errorf("method = %q, want POST", r.Method)
				}
				if !strings.HasSuffix(r.URL.Path, "/images/generations") {
					t.Errorf("path = %q, want /images/generations", r.URL.Path)
				}

				// Verify headers
				if r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("Content-Type = %q, want application/json", r.Header.Get("Content-Type"))
				}
				if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
					t.Error("Authorization header missing Bearer prefix")
				}

				// Read and validate request body
				if tt.req != nil {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						t.Fatalf("failed to read request body: %v", err)
					}

					var reqBody map[string]any
					if err := json.Unmarshal(body, &reqBody); err != nil {
						t.Fatalf("failed to unmarshal request body: %v", err)
					}

					// Validate required fields
					if tt.req.Model != "" {
						if reqBody["model"] != tt.req.Model {
							t.Errorf("model = %v, want %v", reqBody["model"], tt.req.Model)
						}
					}
					if tt.req.Prompt != "" {
						if reqBody["prompt"] != tt.req.Prompt {
							t.Errorf("prompt = %v, want %v", reqBody["prompt"], tt.req.Prompt)
						}
					}

					// Validate optional fields
					if tt.req.N != nil {
						if reqBody["n"] == nil {
							t.Error("n is missing from request")
						}
					}
					if tt.req.Size != "" {
						if reqBody["size"] != tt.req.Size {
							t.Errorf("size = %v, want %v", reqBody["size"], tt.req.Size)
						}
					}
					if tt.req.Quality != "" {
						if reqBody["quality"] != tt.req.Quality {
							t.Errorf("quality = %v, want %v", reqBody["quality"], tt.req.Quality)
						}
					}
					if tt.req.Style != "" {
						if reqBody["style"] != tt.req.Style {
							t.Errorf("style = %v, want %v", reqBody["style"], tt.req.Style)
						}
					}
					if tt.req.ResponseFormat != "" {
						if reqBody["response_format"] != tt.req.ResponseFormat {
							t.Errorf("response_format = %v, want %v", reqBody["response_format"], tt.req.ResponseFormat)
						}
					}
					if tt.req.User != "" {
						if reqBody["user"] != tt.req.User {
							t.Errorf("user = %v, want %v", reqBody["user"], tt.req.User)
						}
					}
				}

				// Send response
				w.WriteHeader(tt.mockStatusCode)

				var respData []byte
				var err error

				if tt.mockError != nil {
					respData, err = json.Marshal(tt.mockError)
				} else if tt.mockResponse != nil {
					respData, err = json.Marshal(tt.mockResponse)
				}

				if err != nil {
					t.Fatalf("failed to marshal response: %v", err)
				}

				w.Write(respData)
			}))
			defer server.Close()

			// Create provider
			provider, err := NewProvider(
				WithAPIKey("sk-test"),
				WithAPIBase(server.URL),
			)
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}

			// Call ImageGeneration
			resp, err := provider.ImageGeneration(context.Background(), tt.req)

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

			if resp.Created != tt.mockResponse.Created {
				t.Errorf("created = %d, want %d", resp.Created, tt.mockResponse.Created)
			}

			if len(resp.Data) != len(tt.mockResponse.Data) {
				t.Errorf("data length = %d, want %d", len(resp.Data), len(tt.mockResponse.Data))
			}

			for i, img := range resp.Data {
				if img.URL != tt.mockResponse.Data[i].URL {
					t.Errorf("data[%d].URL = %q, want %q", i, img.URL, tt.mockResponse.Data[i].URL)
				}
				if img.B64JSON != tt.mockResponse.Data[i].B64JSON {
					t.Errorf("data[%d].B64JSON = %q, want %q", i, img.B64JSON, tt.mockResponse.Data[i].B64JSON)
				}
				if img.RevisedPrompt != tt.mockResponse.Data[i].RevisedPrompt {
					t.Errorf("data[%d].RevisedPrompt = %q, want %q", i, img.RevisedPrompt, tt.mockResponse.Data[i].RevisedPrompt)
				}
			}
		})
	}
}

func TestProvider_ImageGeneration_Integration(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	provider, err := NewProvider(WithAPIKey(apiKey))
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	resp, err := provider.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:  "dall-e-3",
		Prompt: "A cute baby sea otter",
		Size:   "1024x1024",
	})

	if err != nil {
		t.Fatalf("image generation failed: %v", err)
	}

	if resp == nil {
		t.Fatal("response is nil")
	}

	if len(resp.Data) == 0 {
		t.Fatal("response has no images")
	}

	if resp.Data[0].URL == "" {
		t.Error("image URL is empty")
	}

	t.Logf("Generated image URL: %s", resp.Data[0].URL)
	if resp.Data[0].RevisedPrompt != "" {
		t.Logf("Revised prompt: %s", resp.Data[0].RevisedPrompt)
	}
}

func TestProvider_ImageGeneration_RequestOverrides(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify custom API key
		auth := r.Header.Get("Authorization")
		if auth != "Bearer sk-custom" {
			t.Errorf("Authorization = %q, want Bearer sk-custom", auth)
		}

		w.WriteHeader(http.StatusOK)
		resp := &warp.ImageGenerationResponse{
			Created: 1234567890,
			Data: []warp.ImageData{
				{URL: "https://example.com/image.png"},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider, err := NewProvider(
		WithAPIKey("sk-default"),
		WithAPIBase(server.URL),
	)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	// Test request-level API key override
	_, err = provider.ImageGeneration(context.Background(), &warp.ImageGenerationRequest{
		Model:   "dall-e-3",
		Prompt:  "test",
		APIKey:  "sk-custom",
		APIBase: server.URL,
	})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestProvider_ImageGeneration_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Wait for context cancellation
		<-r.Context().Done()
	}))
	defer server.Close()

	provider, err := NewProvider(
		WithAPIKey("sk-test"),
		WithAPIBase(server.URL),
	)
	if err != nil {
		t.Fatalf("failed to create provider: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = provider.ImageGeneration(ctx, &warp.ImageGenerationRequest{
		Model:  "dall-e-3",
		Prompt: "test",
	})

	if err == nil {
		t.Error("expected error for cancelled context")
	}
}
