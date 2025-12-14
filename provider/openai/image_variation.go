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

// ImageVariation creates variations of an existing image using DALL-E.
//
// Supports DALL-E 2 model with PNG images less than 4MB.
// This endpoint generates new variations of the provided image,
// maintaining similar style and content but with variations in details.
//
// Example:
//
//	imageFile, _ := os.Open("original.png")
//	defer imageFile.Close()
//
//	resp, err := provider.ImageVariation(ctx, &warp.ImageVariationRequest{
//	    Model:         "dall-e-2",
//	    Image:         imageFile,
//	    ImageFilename: "original.png",
//	    N:             warp.IntPtr(3),
//	    Size:          "512x512",
//	})
func (p *Provider) ImageVariation(ctx context.Context, req *warp.ImageVariationRequest) (*warp.ImageGenerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.Image == nil {
		return nil, fmt.Errorf("image is required")
	}

	if req.ImageFilename == "" {
		return nil, fmt.Errorf("image filename is required")
	}

	// Build form fields
	fields := make(map[string]string)

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

	// Create multipart form with image
	body, contentType, err := createImageVariationForm(req, fields)
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
	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiBase+"/images/variations", bytes.NewReader(body))
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

// createImageVariationForm creates a multipart form with image file.
//
// The form includes:
// - image file (required)
// - text fields (n, size, response_format, etc.)
//
// Returns the form body, content type header value, and any error.
func createImageVariationForm(req *warp.ImageVariationRequest, fields map[string]string) ([]byte, string, error) {
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
