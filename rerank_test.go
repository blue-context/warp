package warp

import (
	"context"
	"errors"
	"testing"
)

func TestRerank(t *testing.T) {
	tests := []struct {
		name           string
		req            *RerankRequest
		mockResp       *RerankResponse
		mockErr        error
		supportsRerank bool
		wantErr        bool
		errContains    string
	}{
		{
			name: "successful rerank",
			req: &RerankRequest{
				Model: "test/rerank-english-v3.0",
				Query: "What is the capital of France?",
				Documents: []string{
					"Paris is the capital of France",
					"London is the capital of England",
					"Berlin is the capital of Germany",
				},
				TopN: IntPtr(2),
			},
			mockResp: &RerankResponse{
				ID: "test-id",
				Results: []RerankResult{
					{Index: 0, RelevanceScore: 0.95},
					{Index: 2, RelevanceScore: 0.12},
				},
			},
			supportsRerank: true,
			wantErr:        false,
		},
		{
			name: "successful rerank with return documents",
			req: &RerankRequest{
				Model: "test/rerank-english-v3.0",
				Query: "What is the capital of France?",
				Documents: []string{
					"Paris is the capital of France",
					"London is the capital of England",
				},
				TopN:            IntPtr(1),
				ReturnDocuments: BoolPtr(true),
			},
			mockResp: &RerankResponse{
				ID: "test-id",
				Results: []RerankResult{
					{Index: 0, RelevanceScore: 0.95, Document: "Paris is the capital of France"},
				},
			},
			supportsRerank: true,
			wantErr:        false,
		},
		{
			name:        "nil request",
			req:         nil,
			wantErr:     true,
			errContains: "request cannot be nil",
		},
		{
			name: "empty query",
			req: &RerankRequest{
				Model:     "cohere/rerank-english-v3.0",
				Query:     "",
				Documents: []string{"doc1"},
			},
			wantErr:     true,
			errContains: "query is required",
		},
		{
			name: "empty documents",
			req: &RerankRequest{
				Model:     "cohere/rerank-english-v3.0",
				Query:     "test query",
				Documents: []string{},
			},
			wantErr:     true,
			errContains: "documents are required",
		},
		{
			name: "invalid model format",
			req: &RerankRequest{
				Model:     "invalid-model",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			wantErr:     true,
			errContains: "invalid model format",
		},
		{
			name: "provider not found",
			req: &RerankRequest{
				Model:     "unknown/model",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			wantErr:     true,
			errContains: "provider \"unknown\" not found",
		},
		{
			name: "provider does not support rerank",
			req: &RerankRequest{
				Model:     "test/model",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			supportsRerank: false,
			wantErr:        true,
			errContains:    "does not support rerank",
		},
		{
			name: "provider returns error",
			req: &RerankRequest{
				Model:     "test/model",
				Query:     "test query",
				Documents: []string{"doc1"},
			},
			mockErr:        errors.New("provider error"),
			supportsRerank: true,
			wantErr:        true,
			errContains:    "provider error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock provider
			mock := &mockProvider{
				name:         "test",
				rerankResp:   tt.mockResp,
				rerankErr:    tt.mockErr,
				capabilities: Capabilities{Rerank: tt.supportsRerank},
			}

			// Create client
			c := &client{
				config:    defaultConfig(),
				providers: map[string]Provider{},
			}

			// Register mock provider if not testing provider not found
			if tt.req != nil && tt.req.Model != "unknown/model" {
				c.providers["test"] = mock
			}

			// Call Rerank
			resp, err := c.Rerank(context.Background(), tt.req)

			// Check error
			if tt.wantErr {
				if err == nil {
					t.Errorf("Rerank() expected error, got nil")
					return
				}
				if tt.errContains != "" && !contains(err.Error(), tt.errContains) {
					t.Errorf("Rerank() error = %q, want it to contain %q", err.Error(), tt.errContains)
				}
				return
			}

			if err != nil {
				t.Errorf("Rerank() unexpected error: %v", err)
				return
			}

			// Check response
			if resp == nil {
				t.Errorf("Rerank() returned nil response")
				return
			}

			if resp.ID != tt.mockResp.ID {
				t.Errorf("Rerank() ID = %q, want %q", resp.ID, tt.mockResp.ID)
			}

			if len(resp.Results) != len(tt.mockResp.Results) {
				t.Errorf("Rerank() results count = %d, want %d", len(resp.Results), len(tt.mockResp.Results))
			}

			// Check provider is set (model will vary based on test case)
			if resp.Provider != "test" {
				t.Errorf("Rerank() Provider = %q, want %q", resp.Provider, "test")
			}
			if resp.Model == "" {
				t.Errorf("Rerank() Model is empty")
			}
		})
	}
}

func TestRerankWithMaxChunksPerDoc(t *testing.T) {
	// Create mock provider
	mock := &mockProvider{
		name: "test",
		rerankResp: &RerankResponse{
			ID: "test-id",
			Results: []RerankResult{
				{Index: 0, RelevanceScore: 0.95},
			},
		},
		capabilities: Capabilities{Rerank: true},
	}

	// Create client
	c := &client{
		config:    defaultConfig(),
		providers: map[string]Provider{"test": mock},
	}

	// Call Rerank with MaxChunksPerDoc
	resp, err := c.Rerank(context.Background(), &RerankRequest{
		Model:           "test/model",
		Query:           "test query",
		Documents:       []string{"doc1", "doc2"},
		MaxChunksPerDoc: IntPtr(5),
	})

	if err != nil {
		t.Fatalf("Rerank() unexpected error: %v", err)
	}

	if resp == nil {
		t.Fatal("Rerank() returned nil response")
	}

	// Verify the request was passed to provider
	if mock.rerankReq == nil {
		t.Fatal("Provider did not receive rerank request")
	}

	if mock.rerankReq.MaxChunksPerDoc == nil {
		t.Error("MaxChunksPerDoc not passed to provider")
	} else if *mock.rerankReq.MaxChunksPerDoc != 5 {
		t.Errorf("MaxChunksPerDoc = %d, want 5", *mock.rerankReq.MaxChunksPerDoc)
	}
}

func TestRerankContext(t *testing.T) {
	// Create mock provider
	mock := &mockProvider{
		name: "test",
		rerankResp: &RerankResponse{
			ID:      "test-id",
			Results: []RerankResult{{Index: 0, RelevanceScore: 0.95}},
		},
		capabilities: Capabilities{Rerank: true},
	}

	// Create client
	c := &client{
		config:    defaultConfig(),
		providers: map[string]Provider{"test": mock},
	}

	// Create context with request ID
	ctx := WithRequestID(context.Background(), "custom-id")

	// Call Rerank
	_, err := c.Rerank(ctx, &RerankRequest{
		Model:     "test/model",
		Query:     "test query",
		Documents: []string{"doc1"},
	})

	if err != nil {
		t.Fatalf("Rerank() unexpected error: %v", err)
	}

	// Verify context was passed to provider
	if mock.rerankCtx == nil {
		t.Fatal("Provider did not receive context")
	}

	reqID := RequestIDFromContext(mock.rerankCtx)
	if reqID != "custom-id" {
		t.Errorf("Context request ID = %q, want %q", reqID, "custom-id")
	}

	provider := ProviderFromContext(mock.rerankCtx)
	if provider != "test" {
		t.Errorf("Context provider = %q, want %q", provider, "test")
	}

	model := ModelFromContext(mock.rerankCtx)
	if model == "" {
		t.Errorf("Context model is empty")
	}
}
