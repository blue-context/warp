package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/blue-context/warp"
)

func TestModeration(t *testing.T) {
	tests := []struct {
		name           string
		request        *warp.ModerationRequest
		mockResponse   *warp.ModerationResponse
		mockStatusCode int
		wantErr        bool
	}{
		{
			name: "successful single text moderation",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "This is a test",
			},
			mockResponse: &warp.ModerationResponse{
				ID:    "modr-123",
				Model: "text-moderation-latest",
				Results: []warp.ModerationResult{
					{
						Flagged: false,
						Categories: warp.ModerationCategories{
							Sexual:                false,
							Hate:                  false,
							Harassment:            false,
							SelfHarm:              false,
							SexualMinors:          false,
							HateThreatening:       false,
							ViolenceGraphic:       false,
							SelfHarmIntent:        false,
							SelfHarmInstructions:  false,
							HarassmentThreatening: false,
							Violence:              false,
						},
						CategoryScores: warp.ModerationCategoryScores{
							Sexual:                0.0001,
							Hate:                  0.0002,
							Harassment:            0.0003,
							SelfHarm:              0.0001,
							SexualMinors:          0.0001,
							HateThreatening:       0.0001,
							ViolenceGraphic:       0.0002,
							SelfHarmIntent:        0.0001,
							SelfHarmInstructions:  0.0001,
							HarassmentThreatening: 0.0002,
							Violence:              0.0003,
						},
					},
				},
			},
			mockStatusCode: http.StatusOK,
			wantErr:        false,
		},
		{
			name: "flagged content - violence",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "I want to hurt someone",
			},
			mockResponse: &warp.ModerationResponse{
				ID:    "modr-456",
				Model: "text-moderation-latest",
				Results: []warp.ModerationResult{
					{
						Flagged: true,
						Categories: warp.ModerationCategories{
							Sexual:                false,
							Hate:                  false,
							Harassment:            false,
							SelfHarm:              false,
							SexualMinors:          false,
							HateThreatening:       false,
							ViolenceGraphic:       false,
							SelfHarmIntent:        false,
							SelfHarmInstructions:  false,
							HarassmentThreatening: false,
							Violence:              true,
						},
						CategoryScores: warp.ModerationCategoryScores{
							Sexual:                0.0001,
							Hate:                  0.0002,
							Harassment:            0.0003,
							SelfHarm:              0.0001,
							SexualMinors:          0.0001,
							HateThreatening:       0.0001,
							ViolenceGraphic:       0.0002,
							SelfHarmIntent:        0.0001,
							SelfHarmInstructions:  0.0001,
							HarassmentThreatening: 0.0002,
							Violence:              0.95,
						},
					},
				},
			},
			mockStatusCode: http.StatusOK,
			wantErr:        false,
		},
		{
			name: "array input - multiple texts",
			request: &warp.ModerationRequest{
				Model: "text-moderation-stable",
				Input: []string{"This is fine", "I want to hurt someone"},
			},
			mockResponse: &warp.ModerationResponse{
				ID:    "modr-789",
				Model: "text-moderation-stable",
				Results: []warp.ModerationResult{
					{
						Flagged: false,
						Categories: warp.ModerationCategories{
							Violence: false,
						},
						CategoryScores: warp.ModerationCategoryScores{
							Violence: 0.001,
						},
					},
					{
						Flagged: true,
						Categories: warp.ModerationCategories{
							Violence: true,
						},
						CategoryScores: warp.ModerationCategoryScores{
							Violence: 0.95,
						},
					},
				},
			},
			mockStatusCode: http.StatusOK,
			wantErr:        false,
		},
		{
			name: "no model specified",
			request: &warp.ModerationRequest{
				Input: "Test text",
			},
			mockResponse: &warp.ModerationResponse{
				ID:      "modr-default",
				Model:   "text-moderation-latest",
				Results: []warp.ModerationResult{{Flagged: false}},
			},
			mockStatusCode: http.StatusOK,
			wantErr:        false,
		},
		{
			name: "all categories flagged",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "Very harmful content",
			},
			mockResponse: &warp.ModerationResponse{
				ID:    "modr-all-flagged",
				Model: "text-moderation-latest",
				Results: []warp.ModerationResult{
					{
						Flagged: true,
						Categories: warp.ModerationCategories{
							Sexual:                true,
							Hate:                  true,
							Harassment:            true,
							SelfHarm:              true,
							SexualMinors:          true,
							HateThreatening:       true,
							ViolenceGraphic:       true,
							SelfHarmIntent:        true,
							SelfHarmInstructions:  true,
							HarassmentThreatening: true,
							Violence:              true,
						},
						CategoryScores: warp.ModerationCategoryScores{
							Sexual:                0.95,
							Hate:                  0.96,
							Harassment:            0.97,
							SelfHarm:              0.98,
							SexualMinors:          0.99,
							HateThreatening:       0.95,
							ViolenceGraphic:       0.96,
							SelfHarmIntent:        0.97,
							SelfHarmInstructions:  0.98,
							HarassmentThreatening: 0.99,
							Violence:              0.95,
						},
					},
				},
			},
			mockStatusCode: http.StatusOK,
			wantErr:        false,
		},
		{
			name: "API error - 400 Bad Request",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "Test",
			},
			mockStatusCode: http.StatusBadRequest,
			wantErr:        true,
		},
		{
			name: "API error - 401 Unauthorized",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "Test",
			},
			mockStatusCode: http.StatusUnauthorized,
			wantErr:        true,
		},
		{
			name: "API error - 429 Rate Limit",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "Test",
			},
			mockStatusCode: http.StatusTooManyRequests,
			wantErr:        true,
		},
		{
			name: "API error - 500 Internal Server Error",
			request: &warp.ModerationRequest{
				Model: "text-moderation-latest",
				Input: "Test",
			},
			mockStatusCode: http.StatusInternalServerError,
			wantErr:        true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request method and path
				if r.Method != "POST" {
					t.Errorf("Expected POST request, got %s", r.Method)
				}
				if r.URL.Path != "/moderations" {
					t.Errorf("Expected /moderations path, got %s", r.URL.Path)
				}

				// Verify Authorization header
				authHeader := r.Header.Get("Authorization")
				if authHeader != "Bearer test-key" {
					t.Errorf("Expected Authorization header with Bearer token, got %s", authHeader)
				}

				// Verify Content-Type
				contentType := r.Header.Get("Content-Type")
				if contentType != "application/json" {
					t.Errorf("Expected Content-Type application/json, got %s", contentType)
				}

				// For successful responses, verify request body and send response
				if tt.mockStatusCode == http.StatusOK {
					var reqBody map[string]any
					if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
						t.Errorf("Failed to decode request body: %v", err)
					}

					// Verify input is present
					if _, ok := reqBody["input"]; !ok {
						t.Error("Request body missing 'input' field")
					}

					// Write successful response
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tt.mockStatusCode)
					if err := json.NewEncoder(w).Encode(tt.mockResponse); err != nil {
						t.Errorf("Failed to encode response: %v", err)
					}
				} else {
					// Write error response
					w.WriteHeader(tt.mockStatusCode)
					errorResp := map[string]any{
						"error": map[string]any{
							"message": "API error",
							"type":    "invalid_request_error",
						},
					}
					json.NewEncoder(w).Encode(errorResp)
				}
			}))
			defer server.Close()

			// Create provider
			provider, err := NewProvider(
				WithAPIKey("test-key"),
				WithAPIBase(server.URL),
			)
			if err != nil {
				t.Fatalf("Failed to create provider: %v", err)
			}

			// Call Moderation
			resp, err := provider.Moderation(context.Background(), tt.request)

			// Check error expectation
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
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

			if resp.ID != tt.mockResponse.ID {
				t.Errorf("Expected ID %q, got %q", tt.mockResponse.ID, resp.ID)
			}

			if resp.Model != tt.mockResponse.Model {
				t.Errorf("Expected model %q, got %q", tt.mockResponse.Model, resp.Model)
			}

			if len(resp.Results) != len(tt.mockResponse.Results) {
				t.Errorf("Expected %d results, got %d", len(tt.mockResponse.Results), len(resp.Results))
				return
			}

			// Validate first result
			if len(resp.Results) > 0 {
				result := resp.Results[0]
				expected := tt.mockResponse.Results[0]

				if result.Flagged != expected.Flagged {
					t.Errorf("Expected flagged %v, got %v", expected.Flagged, result.Flagged)
				}

				// Check some category flags
				if result.Categories.Violence != expected.Categories.Violence {
					t.Errorf("Expected Violence category %v, got %v", expected.Categories.Violence, result.Categories.Violence)
				}

				// Check category scores
				if result.CategoryScores.Violence != expected.CategoryScores.Violence {
					t.Errorf("Expected Violence score %f, got %f", expected.CategoryScores.Violence, result.CategoryScores.Violence)
				}
			}
		})
	}
}

func TestModerationContextCancellation(t *testing.T) {
	// Create a server that delays response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// This won't actually execute because context will be cancelled
		select {
		case <-r.Context().Done():
			return
		}
	}))
	defer server.Close()

	provider, err := NewProvider(
		WithAPIKey("test-key"),
		WithAPIBase(server.URL),
	)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Call should fail with context cancellation
	_, err = provider.Moderation(ctx, &warp.ModerationRequest{
		Input: "Test",
	})

	if err == nil {
		t.Error("Expected error from cancelled context")
	}
}
