package warp

import (
	"context"
	"fmt"
	"io"
)

// Transcription transcribes audio to text using the specified model.
//
// The audio file is uploaded via multipart/form-data. The File reader will be
// fully consumed during the request. For files, use os.Open() and defer Close().
//
// Thread Safety: This method is safe for concurrent use.
//
// Example:
//
//	f, err := os.Open("meeting.mp3")
//	if err != nil {
//	    return err
//	}
//	defer f.Close()
//
//	resp, err := client.Transcription(ctx, &warp.TranscriptionRequest{
//	    Model:    "openai/whisper-1",
//	    File:     f,
//	    Filename: "meeting.mp3",
//	    Language: "en",
//	})
//	if err != nil {
//	    return err
//	}
//	fmt.Println(resp.Text)
func (c *client) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if req.File == nil {
		return nil, fmt.Errorf("file is required")
	}
	if req.Filename == "" {
		return nil, fmt.Errorf("filename is required for multipart upload")
	}

	// Add request ID if not present
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Parse model to extract provider and model name
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Add provider and model to context
	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	provider, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found: %w", providerName, err)
	}

	// Check if provider supports transcription
	// Note: Supports() returns interface{} to avoid import cycle
	// We use type assertion to check the Transcription field
	supportsVal := provider.Supports()

	// Define an interface that matches provider.Capabilities
	type capabilitiesWithTranscription interface {
		GetTranscription() bool
	}

	// Try to check if transcription is supported
	// The provider.Capabilities struct has a Transcription field
	supportsTranscription := false

	// Use reflection-free struct field access
	type capsStruct struct {
		Transcription bool
	}
	if v, ok := supportsVal.(struct {
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
	}); ok {
		supportsTranscription = v.Transcription
	}

	if !supportsTranscription {
		return nil, fmt.Errorf("provider %q does not support transcription", providerName)
	}

	// Update model name in request (strip provider prefix)
	req.Model = modelName

	// Apply default timeout if not specified
	if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	} else if c.config.DefaultTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	}

	// Call provider with retry logic
	var resp *TranscriptionResponse
	err = c.withRetry(ctx, func() error {
		var callErr error
		resp, callErr = provider.Transcription(ctx, req)
		return callErr
	})

	if err != nil {
		return nil, err
	}

	// Set metadata
	resp.Provider = providerName
	resp.Model = modelName

	return resp, nil
}

// Speech converts text to speech.
//
// Returns an io.ReadCloser containing the audio data.
// The caller MUST call Close() when done to release resources.
//
// Thread Safety: This method is safe for concurrent use.
//
// Example:
//
//	audio, err := client.Speech(ctx, &warp.SpeechRequest{
//	    Model: "openai/tts-1",
//	    Input: "Hello, world!",
//	    Voice: "alloy",
//	})
//	if err != nil {
//	    return err
//	}
//	defer audio.Close()
//
//	// Write to file
//	out, err := os.Create("output.mp3")
//	if err != nil {
//	    return err
//	}
//	defer out.Close()
//	io.Copy(out, audio)
func (c *client) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if req.Input == "" {
		return nil, fmt.Errorf("input text is required")
	}
	if req.Voice == "" {
		return nil, fmt.Errorf("voice is required")
	}

	// Add request ID if not present
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Parse model to extract provider and model name
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Add provider and model to context
	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	provider, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found: %w", providerName, err)
	}

	// Check if provider supports speech
	supportsVal := provider.Supports()

	// Check Speech capability
	supportsSpeech := false
	if v, ok := supportsVal.(struct {
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
	}); ok {
		supportsSpeech = v.Speech
	}

	if !supportsSpeech {
		return nil, fmt.Errorf("provider %q does not support text-to-speech", providerName)
	}

	// Update model name in request (strip provider prefix)
	req.Model = modelName

	// Apply default timeout if not specified
	if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	} else if c.config.DefaultTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	}

	// Call provider (NO retry - streaming response!)
	// Streaming responses cannot be retried as the body is consumed
	audio, err := provider.Speech(ctx, req)
	if err != nil {
		return nil, err
	}

	return audio, nil
}
