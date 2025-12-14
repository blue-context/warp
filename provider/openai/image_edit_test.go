package openai

import (
	"bytes"
	"context"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/blue-context/warp"
)

func TestProvider_ImageEdit(t *testing.T) {
	tests := []struct {
		name         string
		req          *warp.ImageEditRequest
		mockResponse string
		mockStatus   int
		wantErr      bool
		errString    string
	}{
		{
			name: "successful edit without mask",
			req: &warp.ImageEditRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			mockResponse: `{
				"created": 1234567890,
				"data": [
					{
						"url": "https://example.com/edited-image.png"
					}
				]
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "successful edit with mask",
			req: &warp.ImageEditRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Mask:          strings.NewReader("fake mask data"),
				MaskFilename:  "mask.png",
				Prompt:        "Change the background",
			},
			mockResponse: `{
				"created": 1234567890,
				"data": [
					{
						"url": "https://example.com/edited-image.png"
					}
				]
			}`,
			mockStatus: http.StatusOK,
			wantErr:    false,
		},
		{
			name: "with optional parameters",
			req: &warp.ImageEditRequest{
				Model:          "dall-e-2",
				Image:          strings.NewReader("fake image data"),
				ImageFilename:  "original.png",
				Prompt:         "Add a party hat",
				N:              intPtr(2),
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
			req: &warp.ImageEditRequest{
				Model:         "dall-e-2",
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			mockResponse: `{}`,
			mockStatus:   http.StatusOK,
			wantErr:      true,
			errString:    "image is required",
		},
		{
			name: "missing image filename",
			req: &warp.ImageEditRequest{
				Model:  "dall-e-2",
				Image:  strings.NewReader("fake image data"),
				Prompt: "Add a party hat",
			},
			mockResponse: `{}`,
			mockStatus:   http.StatusOK,
			wantErr:      true,
			errString:    "image filename is required",
		},
		{
			name: "missing prompt",
			req: &warp.ImageEditRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
			},
			mockResponse: `{}`,
			mockStatus:   http.StatusOK,
			wantErr:      true,
			errString:    "prompt is required",
		},
		{
			name: "mask without filename",
			req: &warp.ImageEditRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Mask:          strings.NewReader("fake mask data"),
				Prompt:        "Add a party hat",
			},
			mockResponse: `{}`,
			mockStatus:   http.StatusOK,
			wantErr:      true,
			errString:    "mask filename is required when mask is provided",
		},
		{
			name: "API error",
			req: &warp.ImageEditRequest{
				Model:         "dall-e-2",
				Image:         strings.NewReader("fake image data"),
				ImageFilename: "original.png",
				Prompt:        "Add a party hat",
			},
			mockResponse: `{
				"error": {
					"message": "Invalid image format",
					"type": "invalid_request_error",
					"code": "invalid_image"
				}
			}`,
			mockStatus: http.StatusBadRequest,
			wantErr:    true,
			errString:  "Invalid image format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request
				if r.Method != http.MethodPost {
					t.Errorf("expected POST, got %s", r.Method)
				}

				if !strings.HasSuffix(r.URL.Path, "/images/edits") {
					t.Errorf("expected path to end with /images/edits, got %s", r.URL.Path)
				}

				// Verify Authorization header
				authHeader := r.Header.Get("Authorization")
				if !strings.HasPrefix(authHeader, "Bearer ") {
					t.Errorf("expected Authorization header to start with 'Bearer ', got %s", authHeader)
				}

				// Verify Content-Type is multipart
				contentType := r.Header.Get("Content-Type")
				if !strings.HasPrefix(contentType, "multipart/form-data") {
					t.Errorf("expected Content-Type to start with 'multipart/form-data', got %s", contentType)
				}

				// For successful requests, verify multipart form fields
				if tt.mockStatus == http.StatusOK && tt.req != nil && tt.req.Image != nil {
					// Parse multipart form
					mr, err := r.MultipartReader()
					if err != nil {
						t.Fatalf("failed to create multipart reader: %v", err)
					}

					hasImage := false
					hasMask := false
					hasPrompt := false

					for {
						part, err := mr.NextPart()
						if err == io.EOF {
							break
						}
						if err != nil {
							t.Fatalf("failed to read part: %v", err)
						}

						switch part.FormName() {
						case "image":
							hasImage = true
							if part.FileName() != tt.req.ImageFilename {
								t.Errorf("image filename = %q, want %q", part.FileName(), tt.req.ImageFilename)
							}
						case "mask":
							hasMask = true
							if part.FileName() != tt.req.MaskFilename {
								t.Errorf("mask filename = %q, want %q", part.FileName(), tt.req.MaskFilename)
							}
						case "prompt":
							hasPrompt = true
							data, _ := io.ReadAll(part)
							if string(data) != tt.req.Prompt {
								t.Errorf("prompt = %q, want %q", string(data), tt.req.Prompt)
							}
						}
					}

					if !hasImage {
						t.Error("multipart form missing 'image' field")
					}
					if !hasPrompt {
						t.Error("multipart form missing 'prompt' field")
					}
					if tt.req.Mask != nil && !hasMask {
						t.Error("multipart form missing 'mask' field when mask provided")
					}
					if tt.req.Mask == nil && hasMask {
						t.Error("multipart form has 'mask' field when mask not provided")
					}
				}

				// Send response
				w.WriteHeader(tt.mockStatus)
				w.Write([]byte(tt.mockResponse))
			}))
			defer server.Close()

			// Create provider with test server
			provider, err := NewProvider(
				WithAPIKey("test-key"),
				WithAPIBase(server.URL),
			)
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}

			// Execute test
			resp, err := provider.ImageEdit(context.Background(), tt.req)

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

			// Verify response
			if resp == nil {
				t.Fatal("response is nil")
			}

			if resp.Created == 0 {
				t.Error("expected non-zero created timestamp")
			}

			if len(resp.Data) == 0 {
				t.Error("expected at least one image in response")
			}

			// Verify response data based on request
			if tt.req.ResponseFormat == "b64_json" {
				for i, img := range resp.Data {
					if img.B64JSON == "" {
						t.Errorf("image[%d].B64JSON is empty", i)
					}
					if img.URL != "" {
						t.Errorf("image[%d].URL should be empty for b64_json format", i)
					}
				}
			} else {
				for i, img := range resp.Data {
					if img.URL == "" {
						t.Errorf("image[%d].URL is empty", i)
					}
				}
			}

			// Verify number of images matches request
			if tt.req.N != nil && len(resp.Data) != *tt.req.N {
				t.Errorf("got %d images, want %d", len(resp.Data), *tt.req.N)
			}
		})
	}
}

func TestCreateImageEditForm(t *testing.T) {
	tests := []struct {
		name            string
		req             *warp.ImageEditRequest
		fields          map[string]string
		wantErr         bool
		errString       string
		checkImageFile  bool
		checkMaskFile   bool
		checkFields     []string
	}{
		{
			name: "form with image only",
			req: &warp.ImageEditRequest{
				Image:         strings.NewReader("image data"),
				ImageFilename: "test.png",
			},
			fields: map[string]string{
				"prompt": "test prompt",
			},
			checkImageFile: true,
			checkMaskFile:  false,
			checkFields:    []string{"prompt"},
		},
		{
			name: "form with image and mask",
			req: &warp.ImageEditRequest{
				Image:         strings.NewReader("image data"),
				ImageFilename: "test.png",
				Mask:          strings.NewReader("mask data"),
				MaskFilename:  "mask.png",
			},
			fields: map[string]string{
				"prompt": "test prompt",
			},
			checkImageFile: true,
			checkMaskFile:  true,
			checkFields:    []string{"prompt"},
		},
		{
			name: "form with all fields",
			req: &warp.ImageEditRequest{
				Image:         strings.NewReader("image data"),
				ImageFilename: "test.png",
			},
			fields: map[string]string{
				"prompt": "test prompt",
				"n":      "2",
				"size":   "512x512",
			},
			checkImageFile: true,
			checkMaskFile:  false,
			checkFields:    []string{"prompt", "n", "size"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, contentType, err := createImageEditForm(tt.req, tt.fields)

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
				t.Fatal("body is empty")
			}

			if !strings.HasPrefix(contentType, "multipart/form-data; boundary=") {
				t.Errorf("invalid content type: %s", contentType)
			}

			// Parse the multipart form to verify contents
			mr := multipart.NewReader(bytes.NewReader(body), contentType[30:]) // Skip "multipart/form-data; boundary="

			foundImage := false
			foundMask := false
			foundFields := make(map[string]bool)

			for {
				part, err := mr.NextPart()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatalf("failed to read part: %v", err)
				}

				switch part.FormName() {
				case "image":
					foundImage = true
					if part.FileName() != tt.req.ImageFilename {
						t.Errorf("image filename = %q, want %q", part.FileName(), tt.req.ImageFilename)
					}
				case "mask":
					foundMask = true
					if part.FileName() != tt.req.MaskFilename {
						t.Errorf("mask filename = %q, want %q", part.FileName(), tt.req.MaskFilename)
					}
				default:
					foundFields[part.FormName()] = true
				}
			}

			if tt.checkImageFile && !foundImage {
				t.Error("image file not found in form")
			}

			if tt.checkMaskFile && !foundMask {
				t.Error("mask file not found in form")
			}

			if !tt.checkMaskFile && foundMask {
				t.Error("mask file found in form when not expected")
			}

			for _, field := range tt.checkFields {
				if !foundFields[field] {
					t.Errorf("field %q not found in form", field)
				}
			}
		})
	}
}
