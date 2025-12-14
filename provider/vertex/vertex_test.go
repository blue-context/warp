package vertex

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/blue-context/warp"
	prov "github.com/blue-context/warp/provider"
)

func TestNewProvider(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)

	tests := []struct {
		name        string
		opts        []Option
		wantErr     bool
		errContains string
	}{
		{
			name: "valid provider",
			opts: []Option{
				WithProjectID("test-project"),
				WithServiceAccountKey(keyJSON),
			},
			wantErr: false,
		},
		{
			name: "valid with location",
			opts: []Option{
				WithProjectID("test-project"),
				WithLocation("europe-west1"),
				WithServiceAccountKey(keyJSON),
			},
			wantErr: false,
		},
		{
			name: "missing project ID",
			opts: []Option{
				WithServiceAccountKey(keyJSON),
			},
			wantErr:     true,
			errContains: "project ID",
		},
		{
			name: "missing credentials",
			opts: []Option{
				WithProjectID("test-project"),
			},
			wantErr:     true,
			errContains: "credentials",
		},
		{
			name: "empty project ID",
			opts: []Option{
				WithProjectID(""),
				WithServiceAccountKey(keyJSON),
			},
			wantErr:     true,
			errContains: "project ID",
		},
		{
			name: "empty location",
			opts: []Option{
				WithProjectID("test-project"),
				WithLocation(""),
				WithServiceAccountKey(keyJSON),
			},
			wantErr:     true,
			errContains: "location",
		},
		{
			name: "invalid service account key",
			opts: []Option{
				WithProjectID("test-project"),
				WithServiceAccountKey([]byte("invalid")),
			},
			wantErr:     true,
			errContains: "token provider",
		},
		{
			name: "nil HTTP client",
			opts: []Option{
				WithProjectID("test-project"),
				WithServiceAccountKey(keyJSON),
				WithHTTPClient(nil),
			},
			wantErr:     true,
			errContains: "HTTP client",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewProvider(tt.opts...)

			if tt.wantErr {
				if err == nil {
					t.Errorf("NewProvider() expected error, got nil")
					return
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("NewProvider() error = %v, want error containing %q", err, tt.errContains)
				}
				return
			}

			if err != nil {
				t.Errorf("NewProvider() unexpected error = %v", err)
				return
			}

			if provider == nil {
				t.Error("NewProvider() returned nil provider")
			}
		})
	}
}

func TestProvider_Name(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)
	provider, err := NewProvider(
		WithProjectID("test-project"),
		WithServiceAccountKey(keyJSON),
	)
	if err != nil {
		t.Fatalf("NewProvider() failed: %v", err)
	}

	if got := provider.Name(); got != "vertex" {
		t.Errorf("Name() = %v, want %v", got, "vertex")
	}
}

func TestProvider_Supports(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)
	provider, err := NewProvider(
		WithProjectID("test-project"),
		WithServiceAccountKey(keyJSON),
	)
	if err != nil {
		t.Fatalf("NewProvider() failed: %v", err)
	}

	capsInterface := provider.Supports()
	caps, ok := capsInterface.(prov.Capabilities)
	if !ok {
		t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
	}

	// Check expected capabilities
	expected := map[string]bool{
		"Completion":      true,
		"Streaming":       true,
		"FunctionCalling": true,
		"Vision":          true,
		"JSON":            true,
		"Embedding":       false,
		"ImageGeneration": false,
	}

	if caps.Completion != expected["Completion"] {
		t.Errorf("Completion = %v, want %v", caps.Completion, expected["Completion"])
	}
	if caps.Streaming != expected["Streaming"] {
		t.Errorf("Streaming = %v, want %v", caps.Streaming, expected["Streaming"])
	}
	if caps.FunctionCalling != expected["FunctionCalling"] {
		t.Errorf("FunctionCalling = %v, want %v", caps.FunctionCalling, expected["FunctionCalling"])
	}
	if caps.Vision != expected["Vision"] {
		t.Errorf("Vision = %v, want %v", caps.Vision, expected["Vision"])
	}
	if caps.JSON != expected["JSON"] {
		t.Errorf("JSON = %v, want %v", caps.JSON, expected["JSON"])
	}
	if caps.Embedding != expected["Embedding"] {
		t.Errorf("Embedding = %v, want %v", caps.Embedding, expected["Embedding"])
	}
	if caps.ImageGeneration != expected["ImageGeneration"] {
		t.Errorf("ImageGeneration = %v, want %v", caps.ImageGeneration, expected["ImageGeneration"])
	}
}

func TestProvider_BuildEndpoint(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)
	provider, err := NewProvider(
		WithProjectID("test-project"),
		WithLocation("us-central1"),
		WithServiceAccountKey(keyJSON),
	)
	if err != nil {
		t.Fatalf("NewProvider() failed: %v", err)
	}

	tests := []struct {
		name     string
		model    string
		stream   bool
		expected string
	}{
		{
			name:     "non-streaming",
			model:    "gemini-pro",
			stream:   false,
			expected: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:generateContent",
		},
		{
			name:     "streaming",
			model:    "gemini-pro",
			stream:   true,
			expected: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:streamGenerateContent",
		},
		{
			name:     "model with vertex prefix",
			model:    "vertex/gemini-pro",
			stream:   false,
			expected: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-pro:generateContent",
		},
		{
			name:     "different model",
			model:    "gemini-1.5-pro",
			stream:   false,
			expected: "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := provider.buildEndpoint(tt.model, tt.stream)
			if got != tt.expected {
				t.Errorf("buildEndpoint() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestProvider_Completion(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)

	// Mock token server
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"access_token": "test-token",
			"expires_in":   3600,
			"token_type":   "Bearer",
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer tokenServer.Close()

	// Modify key to use mock token server
	var key ServiceAccountKey
	json.Unmarshal(keyJSON, &key)
	key.TokenURI = tokenServer.URL
	modifiedKeyJSON, _ := json.Marshal(key)

	// Mock Vertex AI server
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			t.Errorf("expected Bearer token, got %s", auth)
		}

		// Parse request
		body, _ := io.ReadAll(r.Body)
		var req vertexRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to parse request: %v", err)
		}

		// Return mock response
		resp := vertexResponse{
			Candidates: []vertexCandidate{
				{
					Content: vertexContent{
						Role: "model",
						Parts: []vertexPart{
							{Text: "Hello! How can I help you?"},
						},
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &vertexUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 20,
				TotalTokenCount:      30,
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer vertexServer.Close()

	// Create provider with custom HTTP client
	client := &http.Client{
		Transport: &customTransport{
			vertexURL: vertexServer.URL,
		},
	}

	provider, err := NewProvider(
		WithProjectID("test-project"),
		WithServiceAccountKey(modifiedKeyJSON),
		WithHTTPClient(client),
	)
	if err != nil {
		t.Fatalf("NewProvider() failed: %v", err)
	}

	// Test completion
	resp, err := provider.Completion(context.Background(), &warp.CompletionRequest{
		Model: "gemini-pro",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err != nil {
		t.Fatalf("Completion() failed: %v", err)
	}

	if resp == nil {
		t.Fatal("Completion() returned nil response")
	}

	if len(resp.Choices) == 0 {
		t.Error("Completion() returned no choices")
	}

	if resp.Choices[0].Message.Content == "" {
		t.Error("Completion() returned empty content")
	}

	if resp.Usage == nil {
		t.Error("Completion() returned nil usage")
	}

	if resp.Usage.TotalTokens != 30 {
		t.Errorf("Completion() total tokens = %d, want 30", resp.Usage.TotalTokens)
	}
}

func TestProvider_Completion_Error(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)

	// Mock token server
	tokenServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"access_token": "test-token",
			"expires_in":   3600,
			"token_type":   "Bearer",
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer tokenServer.Close()

	// Modify key
	var key ServiceAccountKey
	json.Unmarshal(keyJSON, &key)
	key.TokenURI = tokenServer.URL
	modifiedKeyJSON, _ := json.Marshal(key)

	tests := []struct {
		name        string
		statusCode  int
		response    interface{}
		wantErr     bool
		errContains string
	}{
		{
			name:       "400 bad request",
			statusCode: http.StatusBadRequest,
			response: map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Invalid request",
				},
			},
			wantErr:     true,
			errContains: "Invalid request",
		},
		{
			name:       "401 unauthorized",
			statusCode: http.StatusUnauthorized,
			response: map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Unauthorized",
				},
			},
			wantErr:     true,
			errContains: "Unauthorized",
		},
		{
			name:       "429 rate limit",
			statusCode: http.StatusTooManyRequests,
			response: map[string]interface{}{
				"error": map[string]interface{}{
					"message": "Rate limit exceeded",
				},
			},
			wantErr:     true,
			errContains: "Rate limit",
		},
		{
			name:       "content blocked",
			statusCode: http.StatusOK,
			response: vertexResponse{
				PromptFeedback: &vertexPromptFeedback{
					BlockReason: "SAFETY",
				},
			},
			wantErr:     true,
			errContains: "blocked",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Mock Vertex AI server
			vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(tt.response)
			}))
			defer vertexServer.Close()

			client := &http.Client{
				Transport: &customTransport{
					vertexURL: vertexServer.URL,
				},
			}

			provider, err := NewProvider(
				WithProjectID("test-project"),
				WithServiceAccountKey(modifiedKeyJSON),
				WithHTTPClient(client),
			)
			if err != nil {
				t.Fatalf("NewProvider() failed: %v", err)
			}

			_, err = provider.Completion(context.Background(), &warp.CompletionRequest{
				Model: "gemini-pro",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			})

			if tt.wantErr && err == nil {
				t.Error("Completion() expected error, got nil")
			}

			if !tt.wantErr && err != nil {
				t.Errorf("Completion() unexpected error = %v", err)
			}

			if tt.wantErr && tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("Completion() error = %v, want error containing %q", err, tt.errContains)
			}
		})
	}
}

func TestProvider_Embedding(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)
	provider, err := NewProvider(
		WithProjectID("test-project"),
		WithServiceAccountKey(keyJSON),
	)
	if err != nil {
		t.Fatalf("NewProvider() failed: %v", err)
	}

	_, err = provider.Embedding(context.Background(), &warp.EmbeddingRequest{
		Model: "text-embedding-gecko",
		Input: "test",
	})

	if err == nil {
		t.Error("Embedding() expected error, got nil")
	}

	if !strings.Contains(err.Error(), "separate") {
		t.Errorf("Embedding() error = %v, want error about separate API", err)
	}
}

// customTransport redirects Vertex AI requests to the test server.
type customTransport struct {
	vertexURL string
}

func (t *customTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Redirect Vertex AI requests to test server
	if strings.Contains(req.URL.Host, "aiplatform.googleapis.com") {
		req.URL.Scheme = "http"
		req.URL.Host = strings.TrimPrefix(t.vertexURL, "http://")
		req.URL.Path = "/"
	}

	return http.DefaultTransport.RoundTrip(req)
}
