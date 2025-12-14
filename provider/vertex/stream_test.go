package vertex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/blue-context/warp"
)

func TestProvider_CompletionStream(t *testing.T) {
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

	// Mock Vertex AI streaming server
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			t.Errorf("expected Bearer token, got %s", auth)
		}

		// Send SSE chunks
		w.Header().Set("Content-Type", "text/event-stream")

		// Chunk 1
		chunk1 := vertexResponse{
			Candidates: []vertexCandidate{
				{
					Content: vertexContent{
						Role:  "model",
						Parts: []vertexPart{{Text: "Hello"}},
					},
					Index: 0,
				},
			},
		}
		data1, _ := json.Marshal(chunk1)
		fmt.Fprintf(w, "data: %s\n\n", data1)

		// Chunk 2
		chunk2 := vertexResponse{
			Candidates: []vertexCandidate{
				{
					Content: vertexContent{
						Role:  "model",
						Parts: []vertexPart{{Text: " World"}},
					},
					Index: 0,
				},
			},
		}
		data2, _ := json.Marshal(chunk2)
		fmt.Fprintf(w, "data: %s\n\n", data2)

		// Final chunk with finish reason
		chunk3 := vertexResponse{
			Candidates: []vertexCandidate{
				{
					Content: vertexContent{
						Role:  "model",
						Parts: []vertexPart{{Text: "!"}},
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			UsageMetadata: &vertexUsageMetadata{
				PromptTokenCount:     5,
				CandidatesTokenCount: 3,
				TotalTokenCount:      8,
			},
		}
		data3, _ := json.Marshal(chunk3)
		fmt.Fprintf(w, "data: %s\n\n", data3)
	}))
	defer vertexServer.Close()

	// Create provider
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

	// Test streaming
	stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "gemini-pro",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err != nil {
		t.Fatalf("CompletionStream() failed: %v", err)
	}
	defer stream.Close()

	// Collect chunks
	var chunks []*warp.CompletionChunk
	var fullContent string

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Recv() failed: %v", err)
		}

		chunks = append(chunks, chunk)

		if len(chunk.Choices) > 0 {
			fullContent += chunk.Choices[0].Delta.Content
		}
	}

	// Verify results
	if len(chunks) != 3 {
		t.Errorf("expected 3 chunks, got %d", len(chunks))
	}

	if fullContent != "Hello World!" {
		t.Errorf("fullContent = %q, want %q", fullContent, "Hello World!")
	}

	// Verify last chunk has finish reason
	lastChunk := chunks[len(chunks)-1]
	if lastChunk.Choices[0].FinishReason == nil {
		t.Error("last chunk should have finish reason")
	} else if *lastChunk.Choices[0].FinishReason != "stop" {
		t.Errorf("finish reason = %q, want %q", *lastChunk.Choices[0].FinishReason, "stop")
	}

	// Verify usage in last chunk
	if lastChunk.Usage == nil {
		t.Error("last chunk should have usage metadata")
	} else if lastChunk.Usage.TotalTokens != 8 {
		t.Errorf("total tokens = %d, want 8", lastChunk.Usage.TotalTokens)
	}
}

func TestProvider_CompletionStream_BlockedContent(t *testing.T) {
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

	// Mock Vertex AI server with blocked content
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Send blocked chunk
		blockedChunk := vertexResponse{
			PromptFeedback: &vertexPromptFeedback{
				BlockReason: "SAFETY",
			},
		}
		data, _ := json.Marshal(blockedChunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
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

	stream, err := provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "gemini-pro",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err != nil {
		t.Fatalf("CompletionStream() failed: %v", err)
	}
	defer stream.Close()

	// Should get error on first Recv
	_, err = stream.Recv()
	if err == nil {
		t.Error("Recv() expected error for blocked content, got nil")
	}

	if !strings.Contains(err.Error(), "blocked") {
		t.Errorf("Recv() error = %v, want error containing 'blocked'", err)
	}
}

func TestProvider_CompletionStream_HTTPError(t *testing.T) {
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

	// Mock Vertex AI server with error
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error": {"message": "Bad request"}}`))
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

	_, err = provider.CompletionStream(context.Background(), &warp.CompletionRequest{
		Model: "gemini-pro",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	})

	if err == nil {
		t.Error("CompletionStream() expected error for HTTP error, got nil")
	}
}

func TestVertexStream_Close(t *testing.T) {
	// Create mock response body
	body := bytes.NewBufferString("data: {}\n\n")

	resp := &http.Response{
		Body: io.NopCloser(body),
	}

	stream := &vertexStream{
		model:    "gemini-pro",
		response: resp,
		reader:   bufio.NewReader(bytes.NewReader([]byte{})),
	}

	// Close once
	err := stream.Close()
	if err != nil {
		t.Errorf("Close() unexpected error = %v", err)
	}

	// Close again (should be safe)
	err = stream.Close()
	if err != nil {
		t.Errorf("Close() second call unexpected error = %v", err)
	}
}

func TestVertexStream_Recv_EOF(t *testing.T) {
	// Create stream with empty body (immediate EOF)
	stream := &vertexStream{
		model:  "gemini-pro",
		reader: bufio.NewReader(bytes.NewReader([]byte{})),
	}

	_, err := stream.Recv()
	if err != io.EOF {
		t.Errorf("Recv() expected io.EOF, got %v", err)
	}

	// Subsequent calls should return same error
	_, err = stream.Recv()
	if err != io.EOF {
		t.Errorf("Recv() second call expected io.EOF, got %v", err)
	}
}

func TestVertexStream_Recv_InvalidJSON(t *testing.T) {
	// Create stream with invalid JSON
	stream := &vertexStream{
		model:  "gemini-pro",
		reader: bufio.NewReader(bytes.NewReader([]byte("data: {invalid json}\n\n"))),
	}

	_, err := stream.Recv()
	if err == nil {
		t.Error("Recv() expected error for invalid JSON, got nil")
	}

	if !strings.Contains(err.Error(), "parse") {
		t.Errorf("Recv() error = %v, want parse error", err)
	}
}

func TestVertexStream_Recv_SkipEmptyLines(t *testing.T) {
	// Create stream with empty lines and valid chunk
	chunk := vertexResponse{
		Candidates: []vertexCandidate{
			{
				Content: vertexContent{
					Role:  "model",
					Parts: []vertexPart{{Text: "Hello"}},
				},
				Index: 0,
			},
		},
	}
	chunkJSON, _ := json.Marshal(chunk)

	data := fmt.Sprintf("\n\ndata: %s\n\n", chunkJSON)

	stream := &vertexStream{
		model:  "gemini-pro",
		reader: bufio.NewReader(bytes.NewReader([]byte(data))),
	}

	chunk1, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv() unexpected error = %v", err)
	}

	if len(chunk1.Choices) != 1 {
		t.Errorf("expected 1 choice, got %d", len(chunk1.Choices))
	}

	if chunk1.Choices[0].Delta.Content != "Hello" {
		t.Errorf("content = %q, want %q", chunk1.Choices[0].Delta.Content, "Hello")
	}
}

func TestVertexStream_Recv_SkipNonDataLines(t *testing.T) {
	// Create stream with event lines and data line
	chunk := vertexResponse{
		Candidates: []vertexCandidate{
			{
				Content: vertexContent{
					Role:  "model",
					Parts: []vertexPart{{Text: "Hello"}},
				},
				Index: 0,
			},
		},
	}
	chunkJSON, _ := json.Marshal(chunk)

	data := fmt.Sprintf("event: message\ndata: %s\n\n", chunkJSON)

	stream := &vertexStream{
		model:  "gemini-pro",
		reader: bufio.NewReader(bytes.NewReader([]byte(data))),
	}

	chunk1, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv() unexpected error = %v", err)
	}

	if chunk1.Choices[0].Delta.Content != "Hello" {
		t.Errorf("content = %q, want %q", chunk1.Choices[0].Delta.Content, "Hello")
	}
}

func TestVertexStream_Recv_DoneMessage(t *testing.T) {
	// Create stream with [DONE] message
	data := "data: [DONE]\n\n"

	stream := &vertexStream{
		model:  "gemini-pro",
		reader: bufio.NewReader(bytes.NewReader([]byte(data))),
	}

	_, err := stream.Recv()
	if err != io.EOF {
		t.Errorf("Recv() expected io.EOF for [DONE], got %v", err)
	}
}

func TestVertexStream_Recv_EmptyCandidates(t *testing.T) {
	// Create stream with chunk that has no candidates
	chunk1 := vertexResponse{
		Candidates: []vertexCandidate{},
	}
	chunk1JSON, _ := json.Marshal(chunk1)

	chunk2 := vertexResponse{
		Candidates: []vertexCandidate{
			{
				Content: vertexContent{
					Role:  "model",
					Parts: []vertexPart{{Text: "Hello"}},
				},
				Index: 0,
			},
		},
	}
	chunk2JSON, _ := json.Marshal(chunk2)

	data := fmt.Sprintf("data: %s\n\ndata: %s\n\n", chunk1JSON, chunk2JSON)

	stream := &vertexStream{
		model:  "gemini-pro",
		reader: bufio.NewReader(bytes.NewReader([]byte(data))),
	}

	// Should skip empty chunk and return second chunk
	result, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv() unexpected error = %v", err)
	}

	if result.Choices[0].Delta.Content != "Hello" {
		t.Errorf("content = %q, want %q", result.Choices[0].Delta.Content, "Hello")
	}
}

func TestTransformStreamChunk(t *testing.T) {
	vResp := &vertexResponse{
		Candidates: []vertexCandidate{
			{
				Content: vertexContent{
					Role:  "model",
					Parts: []vertexPart{{Text: "Hello"}},
				},
				FinishReason: "STOP",
				Index:        0,
			},
		},
		UsageMetadata: &vertexUsageMetadata{
			PromptTokenCount:     5,
			CandidatesTokenCount: 2,
			TotalTokenCount:      7,
		},
	}

	chunk := transformStreamChunk(vResp, "gemini-pro")

	if chunk == nil {
		t.Fatal("transformStreamChunk() returned nil")
	}

	if chunk.Model != "gemini-pro" {
		t.Errorf("model = %q, want %q", chunk.Model, "gemini-pro")
	}

	if len(chunk.Choices) != 1 {
		t.Errorf("expected 1 choice, got %d", len(chunk.Choices))
	}

	if chunk.Choices[0].Delta.Content != "Hello" {
		t.Errorf("content = %q, want %q", chunk.Choices[0].Delta.Content, "Hello")
	}

	if chunk.Choices[0].FinishReason == nil {
		t.Error("finish reason is nil")
	} else if *chunk.Choices[0].FinishReason != "stop" {
		t.Errorf("finish reason = %q, want %q", *chunk.Choices[0].FinishReason, "stop")
	}

	if chunk.Usage == nil {
		t.Error("usage is nil")
	} else if chunk.Usage.TotalTokens != 7 {
		t.Errorf("total tokens = %d, want 7", chunk.Usage.TotalTokens)
	}
}

func TestTransformContentDelta(t *testing.T) {
	tests := []struct {
		name     string
		content  vertexContent
		validate func(*testing.T, warp.MessageDelta)
	}{
		{
			name: "text delta",
			content: vertexContent{
				Role:  "model",
				Parts: []vertexPart{{Text: "Hello"}},
			},
			validate: func(t *testing.T, delta warp.MessageDelta) {
				if delta.Role != "assistant" {
					t.Errorf("role = %q, want %q", delta.Role, "assistant")
				}
				if delta.Content != "Hello" {
					t.Errorf("content = %q, want %q", delta.Content, "Hello")
				}
			},
		},
		{
			name: "with function call",
			content: vertexContent{
				Role: "model",
				Parts: []vertexPart{
					{
						FunctionCall: &vertexFunctionCall{
							Name: "get_weather",
							Args: map[string]interface{}{"location": "SF"},
						},
					},
				},
			},
			validate: func(t *testing.T, delta warp.MessageDelta) {
				if len(delta.ToolCalls) != 1 {
					t.Errorf("expected 1 tool call, got %d", len(delta.ToolCalls))
				}
				if delta.ToolCalls[0].Function.Name != "get_weather" {
					t.Errorf("function name = %q, want %q", delta.ToolCalls[0].Function.Name, "get_weather")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			delta := transformContentDelta(tt.content)
			if tt.validate != nil {
				tt.validate(t, delta)
			}
		})
	}
}
