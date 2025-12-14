package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"

	"github.com/blue-context/warp"
)

// ImageEdit edits an image using DALL-E.
//
// Supports DALL-E 2 model with PNG images less than 4MB.
// The mask (if provided) specifies which areas to edit (transparent = edit).
//
// Example:
//
//	imageFile, _ := os.Open("original.png")
//	defer imageFile.Close()
//
//	resp, err := provider.ImageEdit(ctx, &warp.ImageEditRequest{
//	    Model:         "dall-e-2",
//	    Image:         imageFile,
//	    ImageFilename: "original.png",
//	    Prompt:        "Add a party hat",
//	})
func (p *Provider) ImageEdit(ctx context.Context, req *warp.ImageEditRequest) (*warp.ImageGenerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.Image == nil {
		return nil, fmt.Errorf("image is required")
	}

	if req.ImageFilename == "" {
		return nil, fmt.Errorf("image filename is required")
	}

	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}

	// Validate mask filename if mask provided
	if req.Mask != nil && req.MaskFilename == "" {
		return nil, fmt.Errorf("mask filename is required when mask is provided")
	}

	// Build form fields
	fields := make(map[string]string)
	fields["prompt"] = req.Prompt

	if req.Model != "" {
		fields["model"] = req.Model
	}
	if req.N != nil {
		fields["n"] = fmt.Sprintf("%d", *req.N)
	}
	if req.Size != "" {
		fields["size"] = req.Size
	}
	if req.ResponseFormat != "" {
		fields["response_format"] = req.ResponseFormat
	}
	if req.User != "" {
		fields["user"] = req.User
	}

	// Create multipart form with image and optional mask
	body, contentType, err := createImageEditForm(req, fields)
	if err != nil {
		return nil, fmt.Errorf("failed to create multipart form: %w", err)
	}

	// Determine API key and base URL
	apiKey := p.apiKey
	if req.APIKey != "" {
		apiKey = req.APIKey
	}

	apiBase := p.apiBase
	if req.APIBase != "" {
		apiBase = req.APIBase
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiBase+"/images/edits", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", contentType)
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer httpResp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		return nil, warp.ParseProviderError("openai", httpResp.StatusCode, respBody, nil)
	}

	// Parse response
	var resp warp.ImageGenerationResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// createImageEditForm creates a multipart form with image and optional mask.
//
// The form includes:
// - image file (required)
// - mask file (optional)
// - text fields (prompt, n, size, etc.)
//
// Returns the form body, content type header value, and any error.
func createImageEditForm(req *warp.ImageEditRequest, fields map[string]string) ([]byte, string, error) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add image file
	imagePart, err := writer.CreateFormFile("image", req.ImageFilename)
	if err != nil {
		return nil, "", fmt.Errorf("failed to create image field: %w", err)
	}
	if _, err := io.Copy(imagePart, req.Image); err != nil {
		return nil, "", fmt.Errorf("failed to copy image data: %w", err)
	}

	// Add mask file (optional)
	if req.Mask != nil {
		maskPart, err := writer.CreateFormFile("mask", req.MaskFilename)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create mask field: %w", err)
		}
		if _, err := io.Copy(maskPart, req.Mask); err != nil {
			return nil, "", fmt.Errorf("failed to copy mask data: %w", err)
		}
	}

	// Add text fields
	for key, value := range fields {
		if err := writer.WriteField(key, value); err != nil {
			return nil, "", fmt.Errorf("failed to write field %s: %w", key, err)
		}
	}

	// Close writer to finalize multipart message
	if err := writer.Close(); err != nil {
		return nil, "", fmt.Errorf("failed to close multipart writer: %w", err)
	}

	return buf.Bytes(), writer.FormDataContentType(), nil
}
