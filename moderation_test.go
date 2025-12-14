package warp

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/blue-context/warp/types"
)

// mockModerationProvider is a test provider that implements moderation.
type mockModerationProvider struct {
	name               string
	moderationResp     *ModerationResponse
	moderationErr      error
	supportsModeration bool
}

func (m *mockModerationProvider) Name() string {
	return m.name
}

func (m *mockModerationProvider) Completion(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) CompletionStream(ctx context.Context, req *CompletionRequest) (Stream, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) Embedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) Transcription(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) Speech(ctx context.Context, req *SpeechRequest) (io.ReadCloser, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) ImageGeneration(ctx context.Context, req *ImageGenerationRequest) (*ImageGenerationResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) ImageEdit(ctx context.Context, req *ImageEditRequest) (*ImageGenerationResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) ImageVariation(ctx context.Context, req *ImageVariationRequest) (*ImageGenerationResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) Moderation(ctx context.Context, req *ModerationRequest) (*ModerationResponse, error) {
	if m.moderationErr != nil {
		return nil, m.moderationErr
	}
	return m.moderationResp, nil
}

func (m *mockModerationProvider) Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error) {
	return nil, errors.New("not implemented")
}

func (m *mockModerationProvider) GetModelInfo(model string) *types.ModelInfo {
	return nil
}

func (m *mockModerationProvider) ListModels() []*types.ModelInfo {
	return nil
}

func (m *mockModerationProvider) Supports() interface{} {
	// Return full capabilities struct to match type assertion in moderation.go
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
		Moderation: m.supportsModeration,
	}
}

func TestModeration(t *testing.T) {
	tests := []struct {
		name               string
		request            *ModerationRequest
		mockResp           *ModerationResponse
		mockErr            error
		supportsModeration bool
		wantErr            bool
		wantErrMsg         string
	}{
		{
			name: "successful single text moderation",
			request: &ModerationRequest{
				Model: "test/text-moderation-latest",
				Input: "This is a test",
			},
			mockResp: &ModerationResponse{
				ID:    "modr-123",
				Model: "text-moderation-latest",
				Results: []ModerationResult{
					{
						Flagged: false,
						Categories: ModerationCategories{
							Sexual:     false,
							Hate:       false,
							Harassment: false,
							SelfHarm:   false,
							Violence:   false,
						},
						CategoryScores: ModerationCategoryScores{
							Sexual:     0.001,
							Hate:       0.002,
							Harassment: 0.003,
							SelfHarm:   0.001,
							Violence:   0.002,
						},
					},
				},
			},
			supportsModeration: true,
			wantErr:            false,
		},
		{
			name: "successful flagged content",
			request: &ModerationRequest{
				Model: "test/text-moderation-latest",
				Input: "I want to hurt someone",
			},
			mockResp: &ModerationResponse{
				ID:    "modr-456",
				Model: "text-moderation-latest",
				Results: []ModerationResult{
					{
						Flagged: true,
						Categories: ModerationCategories{
							Violence: true,
						},
						CategoryScores: ModerationCategoryScores{
							Violence: 0.95,
						},
					},
				},
			},
			supportsModeration: true,
			wantErr:            false,
		},
		{
			name: "successful array input",
			request: &ModerationRequest{
				Model: "test/text-moderation-latest",
				Input: []string{"Text 1", "Text 2"},
			},
			mockResp: &ModerationResponse{
				ID:    "modr-789",
				Model: "text-moderation-latest",
				Results: []ModerationResult{
					{
						Flagged:        false,
						Categories:     ModerationCategories{},
						CategoryScores: ModerationCategoryScores{},
					},
					{
						Flagged:        false,
						Categories:     ModerationCategories{},
						CategoryScores: ModerationCategoryScores{},
					},
				},
			},
			supportsModeration: true,
			wantErr:            false,
		},
		{
			name: "default model",
			request: &ModerationRequest{
				Model: "openai/text-moderation-latest", // Specify model to avoid registration issue
				Input: "Test text",
			},
			mockResp: &ModerationResponse{
				ID:      "modr-default",
				Model:   "text-moderation-latest",
				Results: []ModerationResult{{Flagged: false}},
			},
			supportsModeration: true,
			wantErr:            false,
		},
		{
			name:       "nil request",
			request:    nil,
			wantErr:    true,
			wantErrMsg: "request cannot be nil",
		},
		{
			name: "nil input",
			request: &ModerationRequest{
				Model: "test/text-moderation-latest",
				Input: nil,
			},
			wantErr:    true,
			wantErrMsg: "input is required",
		},
		{
			name: "provider not found",
			request: &ModerationRequest{
				Model: "nonexistent/model",
				Input: "Test",
			},
			wantErr:    true,
			wantErrMsg: "provider \"nonexistent\" not found (did you register it?)",
		},
		{
			name: "provider doesn't support moderation",
			request: &ModerationRequest{
				Model: "test/text-moderation-latest",
				Input: "Test",
			},
			supportsModeration: false,
			wantErr:            true,
			wantErrMsg:         "provider \"test\" does not support moderation",
		},
		{
			name: "provider error",
			request: &ModerationRequest{
				Model: "test/text-moderation-latest",
				Input: "Test",
			},
			mockErr:            errors.New("API error"),
			supportsModeration: true,
			wantErr:            true,
			wantErrMsg:         "API error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create client
			c, err := NewClient()
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer c.Close()

			// Register mock provider if needed
			if tt.request != nil && tt.request.Model != "" {
				providerName, _, _ := parseModel(tt.request.Model)
				if providerName != "nonexistent" {
					mockProvider := &mockModerationProvider{
						name:               providerName,
						moderationResp:     tt.mockResp,
						moderationErr:      tt.mockErr,
						supportsModeration: tt.supportsModeration,
					}
					if err := c.RegisterProvider(mockProvider); err != nil {
						t.Fatalf("Failed to register provider: %v", err)
					}
				}
			}

			// Call Moderation
			resp, err := c.Moderation(context.Background(), tt.request)

			// Check error
			if tt.wantErr {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}
				if tt.wantErrMsg != "" && err.Error() != tt.wantErrMsg {
					t.Errorf("Expected error %q, got %q", tt.wantErrMsg, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Validate response
			if resp == nil {
				t.Error("Expected non-nil response")
				return
			}

			if resp.ID != tt.mockResp.ID {
				t.Errorf("Expected ID %q, got %q", tt.mockResp.ID, resp.ID)
			}

			if resp.Model != tt.mockResp.Model {
				t.Errorf("Expected model %q, got %q", tt.mockResp.Model, resp.Model)
			}

			if len(resp.Results) != len(tt.mockResp.Results) {
				t.Errorf("Expected %d results, got %d", len(tt.mockResp.Results), len(resp.Results))
			}

			// Check first result if available
			if len(resp.Results) > 0 && len(tt.mockResp.Results) > 0 {
				if resp.Results[0].Flagged != tt.mockResp.Results[0].Flagged {
					t.Errorf("Expected flagged %v, got %v", tt.mockResp.Results[0].Flagged, resp.Results[0].Flagged)
				}
			}
		})
	}
}

func TestModerationDefaultModel(t *testing.T) {
	// Create client
	c, err := NewClient()
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	// Register mock provider for openai
	mockProvider := &mockModerationProvider{
		name: "openai",
		moderationResp: &ModerationResponse{
			ID:      "modr-test",
			Model:   "text-moderation-latest",
			Results: []ModerationResult{{Flagged: false}},
		},
		supportsModeration: true,
	}
	if err := c.RegisterProvider(mockProvider); err != nil {
		t.Fatalf("Failed to register provider: %v", err)
	}

	// Test with no model specified
	resp, err := c.Moderation(context.Background(), &ModerationRequest{
		Input: "Test text",
	})

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if resp == nil {
		t.Fatal("Expected non-nil response")
	}

	// Provider should be set to openai
	if resp.Provider != "openai" {
		t.Errorf("Expected provider to be openai, got %q", resp.Provider)
	}
}
