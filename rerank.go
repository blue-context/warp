package warp

import (
	"context"
	"fmt"
	"time"
)

// Rerank ranks documents by relevance to a query.
//
// Used for RAG (Retrieval-Augmented Generation) applications to rank
// retrieved documents before feeding them to an LLM. This improves
// the quality of the final response by ensuring the most relevant
// documents are used.
//
// Example:
//
//	resp, err := client.Rerank(ctx, &warp.RerankRequest{
//	    Model: "cohere/rerank-english-v3.0",
//	    Query: "What is the capital of France?",
//	    Documents: []string{
//	        "Paris is the capital of France",
//	        "London is the capital of England",
//	        "Berlin is the capital of Germany",
//	    },
//	    TopN: warp.IntPtr(2),
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	for _, result := range resp.Results {
//	    fmt.Printf("Document %d: score=%.3f\n", result.Index, result.RelevanceScore)
//	}
//
// RAG Pipeline Example:
//
//	// 1. Retrieve documents from vector store
//	docs := vectorStore.Search(query, topK=100)
//
//	// 2. Rerank to find most relevant
//	reranked, _ := client.Rerank(ctx, &warp.RerankRequest{
//	    Model: "cohere/rerank-english-v3.0",
//	    Query: query,
//	    Documents: docs,
//	    TopN: warp.IntPtr(5),
//	})
//
//	// 3. Use top results in completion
//	context := ""
//	for _, result := range reranked.Results {
//	    context += docs[result.Index] + "\n\n"
//	}
//	completion, _ := client.Completion(ctx, &warp.CompletionRequest{
//	    Model: "openai/gpt-4",
//	    Messages: []warp.Message{
//	        {Role: "system", Content: "Answer based on this context:\n" + context},
//	        {Role: "user", Content: query},
//	    },
//	})
func (c *client) Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("request cannot be nil")
	}
	if req.Query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if len(req.Documents) == 0 {
		return nil, fmt.Errorf("documents are required")
	}

	// Add request ID to context
	if RequestIDFromContext(ctx) == "" {
		ctx = WithGeneratedRequestID(ctx)
	}

	// Add start time to context
	ctx = WithStartTime(ctx, time.Now())

	// Parse model string
	providerName, modelName, err := parseModel(req.Model)
	if err != nil {
		return nil, err
	}

	// Add provider and model to context
	ctx = WithProvider(ctx, providerName)
	ctx = WithModel(ctx, modelName)

	// Get provider
	p, err := c.getProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("provider %q not found (did you register it?)", providerName)
	}

	// Check if provider supports reranking
	// Note: Supports() returns interface{} to avoid import cycle
	// We use type assertion to check the Rerank field
	supportsVal := p.Supports()

	// Try to check if rerank is supported
	// The provider.Capabilities struct has a Rerank field
	supportsRerank := false

	// Use reflection-free struct field access
	if v, ok := supportsVal.(struct {
		Completion      bool
		Streaming       bool
		Embedding       bool
		ImageGeneration bool
		ImageEdit       bool
		ImageVariation  bool
		Transcription   bool
		Speech          bool
		Moderation      bool
		FunctionCalling bool
		Vision          bool
		JSON            bool
		Rerank          bool
	}); ok {
		supportsRerank = v.Rerank
	}

	if !supportsRerank {
		return nil, fmt.Errorf("provider %q does not support rerank", providerName)
	}

	// Update model name in request
	req.Model = modelName

	// Apply timeout
	if c.config.DefaultTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.config.DefaultTimeout)
		defer cancel()
	}

	// Call provider with retries
	var resp *RerankResponse
	err = c.withRetry(ctx, func() error {
		var callErr error
		resp, callErr = p.Rerank(ctx, req)
		return callErr
	})

	if err != nil {
		return nil, err
	}

	resp.Provider = providerName
	resp.Model = modelName

	return resp, nil
}
