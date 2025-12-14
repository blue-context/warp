package cohere

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/blue-context/warp"
)

func TestRerank(t *testing.T) {
	tests := []struct {
		name         string
		req          *warp.RerankRequest
		serverResp   map[string]any
		serverStatus int
		wantErr      bool
		errContains  string
		validate     func(t *testing.T, resp *warp.RerankResponse)
	}{
		{
			name: "successful rerank",
			req: &warp.RerankRequest{
				Model: "rerank-english-v3.0",
				Query: "What is the capital of France?",
				Documents: []string{
					"Paris is the capital of France",
					"London is the capital of England",
					"Berlin is the capital of Germany",
				},
			},
			serverResp: map[string]any{
				"id": "test-rerank-123",
				"results": []map[string]any{
					{"index": 0, "relevance_score": 0.95},
					{"index": 2, "relevance_score": 0.12},
					{"index": 1, "relevance_score": 0.08},
				},
			},
			serverStatus: http.StatusOK,
			wantErr:      false,
			validate: func(t *testing.T, resp *warp.RerankResponse) {
				if resp.ID != "test-rerank-123" {
					t.Errorf("ID = %q, want %q", resp.ID, "test-rerank-123")
				}
				if len(resp.Results) != 3 {
					t.Fatalf("len(Results) = %d, want 3", len(resp.Results))
				}
				if resp.Results[0].Index != 0 {
					t.Errorf("Results[0].Index = %d, want 0", resp.Results[0].Index)
				}
				if resp.Results[0].RelevanceScore != 0.95 {
					t.Errorf("Results[0].RelevanceScore = %f, want 0.95", resp.Results[0].RelevanceScore)
				}
			},
		},
		{
			name: "rerank with top_n",
			req: &warp.RerankRequest{
				Model: "rerank-english-v3.0",
				Query: "test query",
				Documents: []string{
					"doc1", "doc2", "doc3",
				},
				TopN: warp.IntPtr(2),
			},
			serverResp: map[string]any{
				"id": "test-rerank-456",
				"results": []map[string]any{
					{"index": 0, "relevance_score": 0.95},
					{"index": 2, "relevance_score": 0.85},
				},
			},
			serverStatus: http.StatusOK,
			wantErr:      false,
			validate: func(t *testing.T, resp *warp.RerankResponse) {
				if len(resp.Results) != 2 {
					t.Fatalf("len(Results) = %d, want 2", len(resp.Results))
				}
			},
		},
		{
			name: "rerank with return_documents",
			req: &warp.RerankRequest{
				Model: "rerank-multilingual-v3.0",
				Query: "test query",
				Documents: []string{
					"This is document one",
					"This is document two",
				},
				ReturnDocuments: warp.BoolPtr(true),
			},
			serverResp: map[string]any{
				"id": "test-rerank-789",
				"results": []map[string]any{
					{
						"index":           0,
						"relevance_score": 0.95,
						"document": map[string]any{
							"text": "This is document one",
						},
					},
					{
						"index":           1,
						"relevance_score": 0.75,
						"document": map[string]any{
							"text": "This is document two",
						},
					},
				},
			},
			serverStatus: http.StatusOK,
			wantErr:      false,
			validate: func(t *testing.T, resp *warp.RerankResponse) {
				if len(resp.Results) != 2 {
					t.Fatalf("len(Results) = %d, want 2", len(resp.Results))
				}
				if resp.Results[0].Document != "This is document one" {
					t.Errorf("Results[0].Document = %q, want %q", resp.Results[0].Document, "This is document one")
				}
				if resp.Results[1].Document != "This is document two" {
					t.Errorf("Results[1].Document = %q, want %q", resp.Results[1].Document, "This is document two")
				}
			},
		},
		{
			name: "rerank with max_chunks_per_doc",
			req: &warp.RerankRequest{
				Model:           "rerank-english-v3.0",
				Query:           "test query",
				Documents:       []string{"doc1", "doc2"},
				MaxChunksPerDoc: warp.IntPtr(5),
			},
			serverResp: map[string]any{
				"id": "test-rerank-999",
				"results": []map[string]any{
					{"index": 0, "relevance_score": 0.95},
					{"index": 1, "relevance_score": 0.85},
				},
			},
			serverStatus: http.StatusOK,
			wantErr:      false,
			validate: func(t *testing.T, resp *warp.RerankResponse) {
				if resp.ID != "test-rerank-999" {
					t.Errorf("ID = %q, want %q", resp.ID, "test-rerank-999")
				}
			},
		},
		{
			name: "rerank with metadata",
			req: &warp.RerankRequest{
				Model:     "rerank-english-v3.0",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			serverResp: map[string]any{
				"id": "test-rerank-meta",
				"results": []map[string]any{
					{"index": 0, "relevance_score": 0.95},
				},
				"meta": map[string]any{
					"api_version": map[string]any{
						"version": "1.0",
					},
					"billed_units": map[string]any{
						"search_units": 10,
					},
				},
			},
			serverStatus: http.StatusOK,
			wantErr:      false,
			validate: func(t *testing.T, resp *warp.RerankResponse) {
				if resp.Meta == nil {
					t.Fatal("Meta is nil")
				}
				if resp.Meta.APIVersion == nil {
					t.Fatal("Meta.APIVersion is nil")
				}
				if resp.Meta.APIVersion.Version != "1.0" {
					t.Errorf("Meta.APIVersion.Version = %q, want %q", resp.Meta.APIVersion.Version, "1.0")
				}
				if resp.Meta.BilledUnits == nil {
					t.Fatal("Meta.BilledUnits is nil")
				}
				if resp.Meta.BilledUnits.SearchUnits != 10 {
					t.Errorf("Meta.BilledUnits.SearchUnits = %d, want 10", resp.Meta.BilledUnits.SearchUnits)
				}
			},
		},
		{
			name: "API error",
			req: &warp.RerankRequest{
				Model:     "rerank-english-v3.0",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			serverResp: map[string]any{
				"message": "Invalid API key",
			},
			serverStatus: http.StatusUnauthorized,
			wantErr:      true,
			errContains:  "cohere",
		},
		{
			name: "rate limit error",
			req: &warp.RerankRequest{
				Model:     "rerank-english-v3.0",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			serverResp: map[string]any{
				"message": "Rate limit exceeded",
			},
			serverStatus: http.StatusTooManyRequests,
			wantErr:      true,
			errContains:  "cohere",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Verify request method and path
				if r.Method != "POST" {
					t.Errorf("Method = %s, want POST", r.Method)
				}
				if r.URL.Path != "/rerank" {
					t.Errorf("Path = %s, want /rerank", r.URL.Path)
				}

				// Verify headers
				if r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("Content-Type = %s, want application/json", r.Header.Get("Content-Type"))
				}
				if r.Header.Get("Authorization") != "Bearer test-key" {
					t.Errorf("Authorization = %s, want Bearer test-key", r.Header.Get("Authorization"))
				}

				// Decode and verify request body
				var reqBody map[string]any
				if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
					t.Fatalf("Failed to decode request body: %v", err)
				}

				if reqBody["model"] != tt.req.Model {
					t.Errorf("Request model = %v, want %v", reqBody["model"], tt.req.Model)
				}
				if reqBody["query"] != tt.req.Query {
					t.Errorf("Request query = %v, want %v", reqBody["query"], tt.req.Query)
				}

				// Check optional parameters
				if tt.req.TopN != nil {
					if topN, ok := reqBody["top_n"].(float64); !ok || int(topN) != *tt.req.TopN {
						t.Errorf("Request top_n = %v, want %v", reqBody["top_n"], *tt.req.TopN)
					}
				}
				if tt.req.ReturnDocuments != nil {
					if returnDocs, ok := reqBody["return_documents"].(bool); !ok || returnDocs != *tt.req.ReturnDocuments {
						t.Errorf("Request return_documents = %v, want %v", reqBody["return_documents"], *tt.req.ReturnDocuments)
					}
				}
				if tt.req.MaxChunksPerDoc != nil {
					if maxChunks, ok := reqBody["max_chunks_per_doc"].(float64); !ok || int(maxChunks) != *tt.req.MaxChunksPerDoc {
						t.Errorf("Request max_chunks_per_doc = %v, want %v", reqBody["max_chunks_per_doc"], *tt.req.MaxChunksPerDoc)
					}
				}

				// Send response
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.serverStatus)
				json.NewEncoder(w).Encode(tt.serverResp)
			}))
			defer server.Close()

			// Create provider with test server
			provider, err := NewProvider(
				WithAPIKey("test-key"),
				WithAPIBase(server.URL),
			)
			if err != nil {
				t.Fatalf("NewProvider() error: %v", err)
			}

			// Call Rerank
			resp, err := provider.Rerank(context.Background(), tt.req)

			// Check error
			if tt.wantErr {
				if err == nil {
					t.Errorf("Rerank() expected error, got nil")
					return
				}
				if tt.errContains != "" {
					errMsg := err.Error()
					found := false
					for i := 0; i <= len(errMsg)-len(tt.errContains); i++ {
						if errMsg[i:i+len(tt.errContains)] == tt.errContains {
							found = true
							break
						}
					}
					if !found {
						t.Errorf("Rerank() error = %q, want it to contain %q", errMsg, tt.errContains)
					}
				}
				return
			}

			if err != nil {
				t.Errorf("Rerank() unexpected error: %v", err)
				return
			}

			// Validate response
			if tt.validate != nil {
				tt.validate(t, resp)
			}
		})
	}
}

func TestRerankContextCancellation(t *testing.T) {
	// Create provider with non-existent server
	provider, err := NewProvider(
		WithAPIKey("test-key"),
		WithAPIBase("http://localhost:99999"),
	)
	if err != nil {
		t.Fatalf("NewProvider() error: %v", err)
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Call Rerank with cancelled context
	_, err = provider.Rerank(ctx, &warp.RerankRequest{
		Model:     "rerank-english-v3.0",
		Query:     "test query",
		Documents: []string{"doc1"},
	})

	// Should get context cancelled error
	if err == nil {
		t.Error("Rerank() expected error with cancelled context, got nil")
	}
}
