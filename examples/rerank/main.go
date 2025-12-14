// Package main demonstrates using the rerank API in a RAG (Retrieval-Augmented Generation) pipeline.
//
// This example shows how to:
// 1. Retrieve documents from a knowledge base (simulated)
// 2. Rerank them by relevance to a query using Cohere
// 3. Use the top results as context for a completion request
//
// Run:
//
//	export COHERE_API_KEY=your-api-key-here
//	export OPENAI_API_KEY=your-api-key-here
//	go run main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/blue-context/warp"
	"github.com/blue-context/warp/provider/cohere"
	"github.com/blue-context/warp/provider/openai"
)

func main() {
	// Create Warp client
	client, err := warp.NewClient()
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	// Register Cohere provider for reranking
	cohereProvider, err := cohere.NewProvider(
		cohere.WithAPIKey(os.Getenv("COHERE_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}
	if err := client.RegisterProvider(cohereProvider); err != nil {
		log.Fatal(err)
	}

	// Register OpenAI provider for completions
	openaiProvider, err := openai.NewProvider(
		openai.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}
	if err := client.RegisterProvider(openaiProvider); err != nil {
		log.Fatal(err)
	}

	// Example RAG pipeline
	query := "What is the capital of France?"

	// Step 1: Retrieve documents from knowledge base (simulated)
	// In a real application, this would come from a vector database,
	// search engine, or other retrieval system.
	documents := []string{
		"Paris is the capital and most populous city of France. It has been one of Europe's major centers of finance, diplomacy, commerce, fashion, science, and the arts.",
		"London is the capital and largest city of England and the United Kingdom. It stands on the River Thames in south-east England.",
		"Berlin is the capital and largest city of Germany by both area and population. Its 3.7 million inhabitants make it the European Union's most populous city.",
		"Madrid is the capital and most populous city of Spain. The city has almost 3.4 million inhabitants.",
		"Rome is the capital city of Italy. It is also the capital of the Lazio region.",
		"France is a country located in Western Europe. It is the largest country in the European Union by area.",
		"The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
		"The French Revolution was a period of social and political upheaval in France and its colonies.",
	}

	fmt.Printf("Query: %s\n\n", query)
	fmt.Printf("Retrieved %d documents from knowledge base\n\n", len(documents))

	// Step 2: Rerank documents by relevance
	fmt.Println("Reranking documents...")
	rerankResp, err := client.Rerank(context.Background(), &warp.RerankRequest{
		Model:           "cohere/rerank-english-v3.0",
		Query:           query,
		Documents:       documents,
		TopN:            warp.IntPtr(3), // Only keep top 3 most relevant
		ReturnDocuments: warp.BoolPtr(true),
	})
	if err != nil {
		log.Fatal(err)
	}

	// Display reranked results
	fmt.Println("\nTop 3 most relevant documents:")
	for i, result := range rerankResp.Results {
		fmt.Printf("\n%d. (Score: %.3f, Original Index: %d)\n", i+1, result.RelevanceScore, result.Index)
		fmt.Printf("   %s\n", result.Document)
	}

	// Step 3: Build context from top results
	contextStr := ""
	for _, result := range rerankResp.Results {
		contextStr += result.Document + "\n\n"
	}

	// Step 4: Use reranked context in completion
	fmt.Println("\n\nGenerating answer using top-ranked context...")
	completion, err := client.Completion(context.Background(), &warp.CompletionRequest{
		Model: "openai/gpt-3.5-turbo",
		Messages: []warp.Message{
			{
				Role:    "system",
				Content: "You are a helpful assistant. Answer the question based ONLY on the provided context. If the context doesn't contain the answer, say so.",
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Context:\n%s\n\nQuestion: %s", contextStr, query),
			},
		},
		Temperature: warp.Float64Ptr(0.0), // Deterministic for factual answers
		MaxTokens:   warp.IntPtr(100),
	})
	if err != nil {
		log.Fatal(err)
	}

	// Display final answer
	fmt.Println("\nFinal Answer:")
	fmt.Println(completion.Choices[0].Message.Content)

	// Display usage statistics
	fmt.Printf("\n\n=== Usage Statistics ===\n")
	fmt.Printf("Rerank Results: %d documents processed\n", len(rerankResp.Results))
	if rerankResp.Meta != nil && rerankResp.Meta.BilledUnits != nil {
		fmt.Printf("Rerank Cost: %d search units\n", rerankResp.Meta.BilledUnits.SearchUnits)
	}
	if completion.Usage != nil {
		fmt.Printf("Completion Tokens: %d prompt + %d completion = %d total\n",
			completion.Usage.PromptTokens,
			completion.Usage.CompletionTokens,
			completion.Usage.TotalTokens)
	}
}
