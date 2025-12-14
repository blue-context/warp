package warp

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
)

// ImageGeneration generates images from text prompts using the specified model.
//
// The request is routed to the appropriate provider based on the model name prefix.
// Supports OpenAI DALL-E, Azure DALL-E, and other image generation providers.
//
// Example:
//
//	resp, err := client.ImageGeneration(ctx, &warp.ImageGenerationRequest{
//	    Model:  "openai/dall-e-3",
//	    Prompt: "A cute baby sea otter",
//	    Size:   "1024x1024",
//	    Quality: "standard",
//	})
//	if err != nil {
//	    return err
//	}
//	fmt.Println("Image URL:", resp.Data[0].URL)
func (c *client) ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt is required")
	}

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Parse model
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	prov, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found: %w", providerName, err)
	}

	// Type assertion to get the image generation interface
	type imageProvider interface {
		ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error)
	}

	imgProvider, ok := prov.(imageProvider)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support image generation", providerName)
	}

	// Update model name (strip provider prefix)
	req.Model = modelName

	// Apply timeout if configured
	if c.config.DefaultTimeout > 0 && req.Timeout == 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	} else if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	}

	// Execute with retry logic
	var resp *ImageGenerationResponse
	err = c.withRetry(ctx, func() error {
		var retryErr error
		resp, retryErr = imgProvider.ImageGeneration(ctx, req)
		return retryErr
	})

	if err != nil {
		return nil, err
	}

	// Set metadata
	resp.Provider = providerName
	resp.Model = modelName

	return resp, nil
}

// ImageEdit edits an image using AI based on a text prompt.
//
// The request is routed to the appropriate provider based on the model name prefix.
// Supports OpenAI DALL-E 2 and other providers with image editing capabilities.
//
// The image must be a PNG file less than 4MB.
// The mask (if provided) indicates which areas to edit (transparent areas = edit).
//
// Example:
//
//	imageFile, _ := os.Open("original.png")
//	defer imageFile.Close()
//
//	resp, err := client.ImageEdit(ctx, &warp.ImageEditRequest{
//	    Model:         "openai/dall-e-2",
//	    Image:         imageFile,
//	    ImageFilename: "original.png",
//	    Prompt:        "Add a party hat to the cat",
//	})
//	if err != nil {
//	    return err
//	}
//	fmt.Println("Edited image URL:", resp.Data[0].URL)
func (c *client) ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageGenerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
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

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Parse model
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	prov, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found: %w", providerName, err)
	}

	// Type assertion to get the image edit interface
	type imageEditProvider interface {
		ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageGenerationResponse, error)
	}

	imgEditProvider, ok := prov.(imageEditProvider)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support image editing", providerName)
	}

	// Update model name (strip provider prefix)
	req.Model = modelName

	// Apply timeout if configured
	if c.config.DefaultTimeout > 0 && req.Timeout == 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	} else if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	}

	// Execute with retry logic
	var resp *ImageGenerationResponse
	err = c.withRetry(ctx, func() error {
		var retryErr error
		resp, retryErr = imgEditProvider.ImageEdit(ctx, req)
		return retryErr
	})

	if err != nil {
		return nil, err
	}

	// Set metadata
	resp.Provider = providerName
	resp.Model = modelName

	return resp, nil
}

// ImageVariation creates variations of an existing image.
//
// The request is routed to the appropriate provider based on the model name prefix.
// Supports OpenAI DALL-E 2 and other providers with image variation capabilities.
//
// The image must be a PNG file less than 4MB.
//
// Example:
//
//	imageFile, _ := os.Open("original.png")
//	defer imageFile.Close()
//
//	resp, err := client.ImageVariation(ctx, &warp.ImageVariationRequest{
//	    Model:         "openai/dall-e-2",
//	    Image:         imageFile,
//	    ImageFilename: "original.png",
//	    N:             warp.IntPtr(3),
//	    Size:          "512x512",
//	})
//	if err != nil {
//	    return err
//	}
//	for _, img := range resp.Data {
//	    fmt.Println("Variation URL:", img.URL)
//	}
func (c *client) ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageGenerationResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}

	if req.Image == nil {
		return nil, fmt.Errorf("image is required")
	}

	if req.ImageFilename == "" {
		return nil, fmt.Errorf("image filename is required")
	}

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Default model if not specified
	if req.Model == "" {
		req.Model = "openai/dall-e-2"
	}

	// Parse model
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	prov, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found: %w", providerName, err)
	}

	// Type assertion to get the image variation interface
	type imageVariationProvider interface {
		ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageGenerationResponse, error)
	}

	imgVarProvider, ok := prov.(imageVariationProvider)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support image variation", providerName)
	}

	// Update model name (strip provider prefix)
	req.Model = modelName

	// Apply timeout if configured
	if c.config.DefaultTimeout > 0 && req.Timeout == 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	} else if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	}

	// Execute with retry logic
	var resp *ImageGenerationResponse
	err = c.withRetry(ctx, func() error {
		var retryErr error
		resp, retryErr = imgVarProvider.ImageVariation(ctx, req)
		return retryErr
	})

	if err != nil {
		return nil, err
	}

	// Set metadata
	resp.Provider = providerName
	resp.Model = modelName

	return resp, nil
}

// SaveToFile downloads and saves an image to a file.
//
// Works with both URL and base64-encoded images.
// The file is created with 0644 permissions.
//
// Example:
//
//	err := imageData.SaveToFile(ctx, "output.png")
func (img *ImageData) SaveToFile(ctx context.Context, path string) error {
	var data []byte
	var err error

	if img.URL != "" {
		// Download from URL
		data, err = downloadImage(ctx, img.URL)
		if err != nil {
			return fmt.Errorf("failed to download image: %w", err)
		}
	} else if img.B64JSON != "" {
		// Decode base64
		data, err = base64.StdEncoding.DecodeString(img.B64JSON)
		if err != nil {
			return fmt.Errorf("failed to decode base64 image: %w", err)
		}
	} else {
		return fmt.Errorf("no image data available (neither URL nor base64)")
	}

	// Write to file
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// downloadImage downloads an image from a URL.
func downloadImage(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	return data, nil
}
