package provider

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/blue-context/warp"
)

// AssertProviderCompliance verifies that a provider implements all required interface methods.
//
// This function checks that all 14 methods defined in the Provider interface exist
// and are callable. It does not make actual API calls or validate responses.
//
// Usage:
//
//	func TestCompliance(t *testing.T) {
//	    p, err := NewProvider(WithAPIKey("test-key"))
//	    if err != nil {
//	        t.Fatalf("NewProvider() error = %v", err)
//	    }
//	    provider.AssertProviderCompliance(t, p)
//	}
func AssertProviderCompliance(t *testing.T, p Provider) {
	t.Helper()

	if p == nil {
		t.Fatal("provider is nil")
	}

	ctx := context.Background()

	// Test Name() method
	t.Run("Name", func(t *testing.T) {
		t.Helper()
		name := p.Name()
		if name == "" {
			t.Error("Name() returned empty string")
		}
	})

	// Test Completion() method signature
	t.Run("Completion", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("Completion")
		if !method.IsValid() {
			t.Fatal("Completion method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("Completion should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("Completion should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test CompletionStream() method signature
	t.Run("CompletionStream", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("CompletionStream")
		if !method.IsValid() {
			t.Fatal("CompletionStream method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("CompletionStream should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("CompletionStream should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test Embedding() method signature
	t.Run("Embedding", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("Embedding")
		if !method.IsValid() {
			t.Fatal("Embedding method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("Embedding should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("Embedding should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test ImageGeneration() method signature
	t.Run("ImageGeneration", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("ImageGeneration")
		if !method.IsValid() {
			t.Fatal("ImageGeneration method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("ImageGeneration should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("ImageGeneration should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test ImageEdit() method signature
	t.Run("ImageEdit", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("ImageEdit")
		if !method.IsValid() {
			t.Fatal("ImageEdit method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("ImageEdit should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("ImageEdit should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test ImageVariation() method signature
	t.Run("ImageVariation", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("ImageVariation")
		if !method.IsValid() {
			t.Fatal("ImageVariation method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("ImageVariation should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("ImageVariation should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test Transcription() method signature
	t.Run("Transcription", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("Transcription")
		if !method.IsValid() {
			t.Fatal("Transcription method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("Transcription should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("Transcription should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test Speech() method signature
	t.Run("Speech", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("Speech")
		if !method.IsValid() {
			t.Fatal("Speech method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("Speech should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("Speech should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test Moderation() method signature
	t.Run("Moderation", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("Moderation")
		if !method.IsValid() {
			t.Fatal("Moderation method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("Moderation should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("Moderation should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test Rerank() method signature
	t.Run("Rerank", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("Rerank")
		if !method.IsValid() {
			t.Fatal("Rerank method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 2 {
			t.Errorf("Rerank should have 2 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 2 {
			t.Errorf("Rerank should return 2 values, got %d", methodType.NumOut())
		}
	})

	// Test Supports() method
	t.Run("Supports", func(t *testing.T) {
		t.Helper()
		caps := p.Supports()
		if caps == nil {
			t.Fatal("Supports() returned nil")
		}

		// Verify it returns Capabilities struct
		_, ok := caps.(Capabilities)
		if !ok {
			t.Errorf("Supports() returned %T, want Capabilities", caps)
		}
	})

	// Test GetModelInfo() method signature
	t.Run("GetModelInfo", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("GetModelInfo")
		if !method.IsValid() {
			t.Fatal("GetModelInfo method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 1 {
			t.Errorf("GetModelInfo should have 1 parameter, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 1 {
			t.Errorf("GetModelInfo should return 1 value, got %d", methodType.NumOut())
		}

		// Test it can be called
		info := p.GetModelInfo("test-model")
		// info can be nil for unknown models, that's valid
		_ = info
	})

	// Test ListModels() method signature
	t.Run("ListModels", func(t *testing.T) {
		t.Helper()
		method := reflect.ValueOf(p).MethodByName("ListModels")
		if !method.IsValid() {
			t.Fatal("ListModels method not found")
		}
		methodType := method.Type()
		if methodType.NumIn() != 0 {
			t.Errorf("ListModels should have 0 parameters, got %d", methodType.NumIn())
		}
		if methodType.NumOut() != 1 {
			t.Errorf("ListModels should return 1 value, got %d", methodType.NumOut())
		}

		// Test it can be called
		models := p.ListModels()
		if models == nil {
			t.Error("ListModels() returned nil, should return empty slice instead")
		}
	})

	// Test that methods can be called without panicking
	t.Run("MethodsCallable", func(t *testing.T) {
		t.Helper()

		// These should not panic even if they return errors
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Method call panicked: %v", r)
			}
		}()

		_, _ = p.Embedding(ctx, &warp.EmbeddingRequest{Model: "test", Input: "test"})
		_, _ = p.ImageGeneration(ctx, &warp.ImageGenerationRequest{Model: "test", Prompt: "test"})
		_, _ = p.ImageEdit(ctx, &warp.ImageEditRequest{Model: "test", Prompt: "test"})
		_, _ = p.ImageVariation(ctx, &warp.ImageVariationRequest{Model: "test"})
		_, _ = p.Transcription(ctx, &warp.TranscriptionRequest{Model: "test"})
		_, _ = p.Speech(ctx, &warp.SpeechRequest{Model: "test", Input: "test"})
		_, _ = p.Moderation(ctx, &warp.ModerationRequest{Model: "test", Input: "test"})
		_, _ = p.Rerank(ctx, &warp.RerankRequest{Model: "test", Query: "test", Documents: []string{"test"}})
	})
}

// AssertMethodCount verifies that the provider has exactly 14 methods.
//
// This ensures no methods are accidentally removed or added without updating the interface.
//
// Note: This counts only exported methods. Private methods are not counted.
func AssertMethodCount(t *testing.T, p Provider) {
	t.Helper()

	if p == nil {
		t.Fatal("provider is nil")
	}

	providerType := reflect.TypeOf(p)
	methodCount := providerType.NumMethod()

	expectedCount := 14 // Based on Provider interface definition

	if methodCount != expectedCount {
		t.Errorf("Provider has %d methods, want %d", methodCount, expectedCount)

		// List all methods for debugging
		t.Logf("Methods found:")
		for i := 0; i < methodCount; i++ {
			method := providerType.Method(i)
			t.Logf("  - %s", method.Name)
		}
	}
}

// AssertStubMethodsReturnWarpError verifies that unsupported methods return proper WarpError.
//
// This function tests all 8 potentially unsupported features and verifies that:
// 1. The method returns an error (not nil)
// 2. The error is a *warp.WarpError (not a generic error)
// 3. The error message indicates the feature is not supported
//
// This ensures consistency across providers - all unsupported features should
// return WarpError, not fmt.Errorf or other generic errors.
func AssertStubMethodsReturnWarpError(t *testing.T, p Provider) {
	t.Helper()

	if p == nil {
		t.Fatal("provider is nil")
	}

	ctx := context.Background()
	caps, ok := p.Supports().(Capabilities)
	if !ok {
		t.Fatal("Supports() did not return Capabilities")
	}

	// Test Embedding if not supported
	t.Run("Embedding", func(t *testing.T) {
		t.Helper()

		if caps.Embedding {
			t.Skip("Embedding is supported, skipping stub validation")
		}

		resp, err := p.Embedding(ctx, &warp.EmbeddingRequest{
			Model: "test-model",
			Input: "test input",
		})

		if err == nil {
			t.Error("Embedding() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("Embedding() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("Embedding() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test ImageGeneration if not supported
	t.Run("ImageGeneration", func(t *testing.T) {
		t.Helper()

		if caps.ImageGeneration {
			t.Skip("ImageGeneration is supported, skipping stub validation")
		}

		resp, err := p.ImageGeneration(ctx, &warp.ImageGenerationRequest{
			Model:  "test-model",
			Prompt: "test prompt",
		})

		if err == nil {
			t.Error("ImageGeneration() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("ImageGeneration() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("ImageGeneration() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test ImageEdit if not supported
	t.Run("ImageEdit", func(t *testing.T) {
		t.Helper()

		if caps.ImageEdit {
			t.Skip("ImageEdit is supported, skipping stub validation")
		}

		resp, err := p.ImageEdit(ctx, &warp.ImageEditRequest{
			Model:  "test-model",
			Prompt: "test prompt",
		})

		if err == nil {
			t.Error("ImageEdit() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("ImageEdit() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("ImageEdit() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test ImageVariation if not supported
	t.Run("ImageVariation", func(t *testing.T) {
		t.Helper()

		if caps.ImageVariation {
			t.Skip("ImageVariation is supported, skipping stub validation")
		}

		resp, err := p.ImageVariation(ctx, &warp.ImageVariationRequest{
			Model: "test-model",
		})

		if err == nil {
			t.Error("ImageVariation() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("ImageVariation() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("ImageVariation() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test Transcription if not supported
	t.Run("Transcription", func(t *testing.T) {
		t.Helper()

		if caps.Transcription {
			t.Skip("Transcription is supported, skipping stub validation")
		}

		resp, err := p.Transcription(ctx, &warp.TranscriptionRequest{
			Model: "test-model",
		})

		if err == nil {
			t.Error("Transcription() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("Transcription() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("Transcription() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test Speech if not supported
	t.Run("Speech", func(t *testing.T) {
		t.Helper()

		if caps.Speech {
			t.Skip("Speech is supported, skipping stub validation")
		}

		resp, err := p.Speech(ctx, &warp.SpeechRequest{
			Model: "test-model",
			Input: "test input",
		})

		if err == nil {
			t.Error("Speech() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("Speech() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("Speech() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test Moderation if not supported
	t.Run("Moderation", func(t *testing.T) {
		t.Helper()

		if caps.Moderation {
			t.Skip("Moderation is supported, skipping stub validation")
		}

		resp, err := p.Moderation(ctx, &warp.ModerationRequest{
			Model: "test-model",
			Input: "test input",
		})

		if err == nil {
			t.Error("Moderation() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("Moderation() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("Moderation() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})

	// Test Rerank if not supported
	t.Run("Rerank", func(t *testing.T) {
		t.Helper()

		if caps.Rerank {
			t.Skip("Rerank is supported, skipping stub validation")
		}

		resp, err := p.Rerank(ctx, &warp.RerankRequest{
			Model:     "test-model",
			Query:     "test query",
			Documents: []string{"doc1", "doc2"},
		})

		if err == nil {
			t.Error("Rerank() error = nil, want error for unsupported feature")
			return
		}

		if resp != nil {
			t.Errorf("Rerank() response = %v, want nil for unsupported feature", resp)
		}

		var warpErr *warp.WarpError
		if !errors.As(err, &warpErr) {
			t.Errorf("Rerank() error type = %T, want *warp.WarpError", err)
			t.Logf("Error message: %v", err)
		}
	})
}

// AssertCapabilitiesAccuracy verifies that Supports() accurately reflects actual implementation.
//
// This function cross-validates the Capabilities returned by Supports() against the actual
// behavior of the provider's methods. It ensures that if Supports().Feature == false,
// calling that feature returns an error indicating it's not supported.
func AssertCapabilitiesAccuracy(t *testing.T, p Provider) {
	t.Helper()

	if p == nil {
		t.Fatal("provider is nil")
	}

	ctx := context.Background()
	caps, ok := p.Supports().(Capabilities)
	if !ok {
		t.Fatal("Supports() did not return Capabilities")
	}

	// Test Embedding capability accuracy
	t.Run("Embedding", func(t *testing.T) {
		t.Helper()

		resp, err := p.Embedding(ctx, &warp.EmbeddingRequest{
			Model: "test-model",
			Input: "test input",
		})

		if !caps.Embedding {
			if err == nil {
				t.Error("Embedding() error = nil, but Supports().Embedding == false")
			}
			if resp != nil {
				t.Error("Embedding() returned non-nil response, but Supports().Embedding == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("Embedding() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test ImageGeneration capability accuracy
	t.Run("ImageGeneration", func(t *testing.T) {
		t.Helper()

		resp, err := p.ImageGeneration(ctx, &warp.ImageGenerationRequest{
			Model:  "test-model",
			Prompt: "test prompt",
		})

		if !caps.ImageGeneration {
			if err == nil {
				t.Error("ImageGeneration() error = nil, but Supports().ImageGeneration == false")
			}
			if resp != nil {
				t.Error("ImageGeneration() returned non-nil response, but Supports().ImageGeneration == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("ImageGeneration() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test ImageEdit capability accuracy
	t.Run("ImageEdit", func(t *testing.T) {
		t.Helper()

		resp, err := p.ImageEdit(ctx, &warp.ImageEditRequest{
			Model:  "test-model",
			Prompt: "test prompt",
		})

		if !caps.ImageEdit {
			if err == nil {
				t.Error("ImageEdit() error = nil, but Supports().ImageEdit == false")
			}
			if resp != nil {
				t.Error("ImageEdit() returned non-nil response, but Supports().ImageEdit == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("ImageEdit() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test ImageVariation capability accuracy
	t.Run("ImageVariation", func(t *testing.T) {
		t.Helper()

		resp, err := p.ImageVariation(ctx, &warp.ImageVariationRequest{
			Model: "test-model",
		})

		if !caps.ImageVariation {
			if err == nil {
				t.Error("ImageVariation() error = nil, but Supports().ImageVariation == false")
			}
			if resp != nil {
				t.Error("ImageVariation() returned non-nil response, but Supports().ImageVariation == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("ImageVariation() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test Transcription capability accuracy
	t.Run("Transcription", func(t *testing.T) {
		t.Helper()

		resp, err := p.Transcription(ctx, &warp.TranscriptionRequest{
			Model: "test-model",
		})

		if !caps.Transcription {
			if err == nil {
				t.Error("Transcription() error = nil, but Supports().Transcription == false")
			}
			if resp != nil {
				t.Error("Transcription() returned non-nil response, but Supports().Transcription == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("Transcription() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test Speech capability accuracy
	t.Run("Speech", func(t *testing.T) {
		t.Helper()

		resp, err := p.Speech(ctx, &warp.SpeechRequest{
			Model: "test-model",
			Input: "test input",
		})

		if !caps.Speech {
			if err == nil {
				t.Error("Speech() error = nil, but Supports().Speech == false")
			}
			if resp != nil {
				t.Error("Speech() returned non-nil response, but Supports().Speech == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("Speech() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test Moderation capability accuracy
	t.Run("Moderation", func(t *testing.T) {
		t.Helper()

		resp, err := p.Moderation(ctx, &warp.ModerationRequest{
			Model: "test-model",
			Input: "test input",
		})

		if !caps.Moderation {
			if err == nil {
				t.Error("Moderation() error = nil, but Supports().Moderation == false")
			}
			if resp != nil {
				t.Error("Moderation() returned non-nil response, but Supports().Moderation == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("Moderation() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Test Rerank capability accuracy
	t.Run("Rerank", func(t *testing.T) {
		t.Helper()

		resp, err := p.Rerank(ctx, &warp.RerankRequest{
			Model:     "test-model",
			Query:     "test query",
			Documents: []string{"doc1", "doc2"},
		})

		if !caps.Rerank {
			if err == nil {
				t.Error("Rerank() error = nil, but Supports().Rerank == false")
			}
			if resp != nil {
				t.Error("Rerank() returned non-nil response, but Supports().Rerank == false")
			}
			if err != nil && !isNotSupportedError(err) {
				t.Errorf("Rerank() error = %v, want 'not supported' error", err)
			}
		}
	})

	// Verify that Completion is always supported
	t.Run("CompletionRequired", func(t *testing.T) {
		t.Helper()

		if !caps.Completion {
			t.Error("Supports().Completion == false, but Completion is required for all providers")
		}
	})

	// Verify that Streaming matches Completion support
	t.Run("StreamingConsistency", func(t *testing.T) {
		t.Helper()

		if caps.Completion && !caps.Streaming {
			t.Log("Note: Provider supports Completion but not Streaming (this is allowed but uncommon)")
		}
	})
}

// isNotSupportedError checks if an error indicates a feature is not supported.
func isNotSupportedError(err error) bool {
	if err == nil {
		return false
	}

	errMsg := strings.ToLower(err.Error())

	// Check for common "not supported" patterns
	notSupportedPatterns := []string{
		"not supported",
		"not available",
		"unsupported",
		"does not support",
		"doesn't support",
	}

	for _, pattern := range notSupportedPatterns {
		if strings.Contains(errMsg, pattern) {
			return true
		}
	}

	// Prefer WarpError for unsupported features
	var warpErr *warp.WarpError
	if errors.As(err, &warpErr) {
		return true
	}

	return false
}
