package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/blue-context/warp"
)

func TestProvider_ImageVariation(t *testing.T) {
	tests := []struct {
		name         string
		req          *warp.ImageVariationRequest
		mockResponse string
		mockStatus   int
		wantErr      bool
		errString    string
	}{
		{
			name: "successful variation",
			req: &warp.ImageVariationRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			mockResponse: `{
				"created": 1234567890,
				"data": [
					{
						"url": "https://example.com/variation1.png"
					},
					{
						"url": "https://example.com/variation2.png"
					}
				]
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "with optional parameters",
			req: &warp.ImageVariationRequest{
				Model:          "dall-e-2",
				Image:          strings.NewReader("fake image data"),
				ImageFilename:  "original.png",
				N:              intPtr(3),
				Size:           "512x512",
				ResponseFormat: "b64_json",
				User:           "test-user",
			},
			mockResponse: `{
				"created": 1234567890,
				"data": [
					{
						"b64_json": "base64data1"
					},
					{
						"b64_json": "base64data2"
					},
					{
						"b64_json": "base64data3"
					}
				]
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "nil request",
			req:  nil,
			mockResponse: `{
				"created": 1234567890,
				"data": []
			}`,
			mockStatus: http.StatusOK,
			wantErr:    true,
			errString:  "request cannot be nil",
		},
		{
			name: "missing image",
			req: &warp.ImageVariationRequest{
				Model:         "dall-e-2",
				ImageFilename: "original.png",
			},
			mockResponse: `{
				"created": 1234567890,
				"data": []
			}`,
			mockStatus: http.StatusOK,
			wantErr:    true,
			errString:  "image is required",
		},
		{
			name: "missing image filename",
			req: &warp.ImageVariationRequest{
				Model: "dall-e-2",
				Image: strings.NewReader("fake image data"),
			},
			mockResponse: `{
				"created": 1234567890,
				"data": []
			}`,
			mockStatus: http.StatusOK,
			wantErr:    true,
			errString:  "image filename is required",
		},
		{
			name: "api error",
			req: &warp.ImageVariationRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			mockResponse: `{
				"error": {
					"message": "Invalid image format",
					"type": "invalid_request_error"
				}
			}`,
			mockStatus: http.StatusBadRequest,
			wantErr:    true,
			errString:  "Invalid image format",
		},
		{
			name: "rate limit error",
			req: &warp.ImageVariationRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			mockResponse: `{
				"error": {
					"message": "Rate limit exceeded",
					"type": "rate_limit_error"
				}
			}`,
			mockStatus: http.StatusTooManyRequests,
			wantErr:    true,
			errString:  "Rate limit exceeded",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request method and path
				if r.Method != "POST" {
					t.Errorf("expected POST request, got %s", r.Method)
				}
				if r.URL.Path != "/images/variations" {
					t.Errorf("expected path /images/variations, got %s", r.URL.Path)
				}

				// Verify content type is multipart/form-data
				contentType := r.Header.Get("Content-Type")
				if !strings.HasPrefix(contentType, "multipart/form-data") {
					t.Errorf("expected multipart/form-data content type, got %s", contentType)
				}

				// Verify Authorization header
				auth := r.Header.Get("Authorization")
				if !strings.HasPrefix(auth, "Bearer ") {
					t.Errorf("expected Authorization header with Bearer token")
				}

				w.WriteHeader(tt.mockStatus)
				w.Write([]byte(tt.mockResponse))
			}))
			defer server.Close()

			// Create provider
			provider, err := NewProvider(
				WithAPIKey("test-key"),
				WithAPIBase(server.URL),
			)
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}

			// Call ImageVariation
			resp, err := provider.ImageVariation(context.Background(), tt.req)

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
				t.Error("expected at least one image in response")
			}

			// Verify the correct number of images if N was specified
			if tt.req != nil && tt.req.N != nil {
				if len(resp.Data) != *tt.req.N {
					t.Errorf("expected %d images, got %d", *tt.req.N, len(resp.Data))
				}
			}

			// Verify response format
			if tt.req != nil && tt.req.ResponseFormat == "b64_json" {
				for i, img := range resp.Data {
					if img.B64JSON == "" {
						t.Errorf("image %d missing b64_json data", i)
					}
					if img.URL != "" {
						t.Errorf("image %d should not have URL when using b64_json format", i)
					}
				}
			} else {
				for i, img := range resp.Data {
					if img.URL == "" {
						t.Errorf("image %d missing URL", i)
					}
				}
			}
		})
	}
}

func TestCreateImageVariationForm(t *testing.T) {
	tests := []struct {
		name       string
		req        *warp.ImageVariationRequest
		fields     map[string]string
		wantErr    bool
		errString  string
		checkField func(t *testing.T, req *http.Request)
	}{
		{
			name: "basic form",
			req: &warp.ImageVariationRequest{
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			fields:  map[string]string{},
			wantErr: false,
			checkField: func(t *testing.T, req *http.Request) {
				if err := req.ParseMultipartForm(32 << 20); err != nil {
					t.Fatalf("failed to parse multipart form: %v", err)
				}
				if req.MultipartForm.File["image"] == nil {
					t.Error("image field not found in multipart form")
				}
			},
		},
		{
			name: "with all fields",
			req: &warp.ImageVariationRequest{
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			fields: map[string]string{
				"model":           "dall-e-2",
				"n":               "3",
				"size":            "512x512",
				"response_format": "b64_json",
				"user":            "test-user",
			},
			wantErr: false,
			checkField: func(t *testing.T, req *http.Request) {
				if err := req.ParseMultipartForm(32 << 20); err != nil {
					t.Fatalf("failed to parse multipart form: %v", err)
				}

				expectedFields := map[string]string{
					"model":           "dall-e-2",
					"n":               "3",
					"size":            "512x512",
					"response_format": "b64_json",
					"user":            "test-user",
				}

				for field, expectedValue := range expectedFields {
					actualValue := req.FormValue(field)
					if actualValue != expectedValue {
						t.Errorf("field %s = %q, want %q", field, actualValue, expectedValue)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, contentType, err := createImageVariationForm(tt.req, tt.fields)

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

			if len(body) == 0 {
				t.Error("body is empty")
			}

			if !strings.HasPrefix(contentType, "multipart/form-data") {
				t.Errorf("expected multipart/form-data content type, got %s", contentType)
			}

			if tt.checkField != nil {
				// Create a mock HTTP request to test the form
				req := httptest.NewRequest("POST", "/test", strings.NewReader(string(body)))
				req.Header.Set("Content-Type", contentType)
				tt.checkField(t, req)
			}
		})
	}
}
