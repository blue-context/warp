package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// CompletionStream sends a streaming chat completion request to Ollama.
//
// This method returns a Stream that delivers response chunks incrementally
// as they become available from the server.
//
// The caller must close the returned stream to release resources.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "llama3",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Tell me a story"},
//	    },
//	})
//	if err != nil {
//	    return err
//	}
//	defer stream.Close()
//
//	for {
//	    chunk, err := stream.Recv()
//	    if err == io.EOF {
//	        break
//	    }
//	    if err != nil {
//	        return err
//	    }
//	    fmt.Print(chunk.Choices[0].Delta.Content)
//	}
func (p *Provider) CompletionStream(ctx context.Context, req *warp.CompletionRequest) (warp.Stream, error) {
	// Transform request to Ollama format (streaming)
	ollamaReq := transformToOllamaRequest(req, true)

	// Marshal to JSON
	body, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers (no auth needed for Ollama)
	httpReq.Header.Set("Content-Type", "application/json")

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		defer httpResp.Body.Close()
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("ollama", httpResp.StatusCode, body, nil)
	}

	// Create Ollama stream (newline-delimited JSON, not SSE)
	return newOllamaStream(ctx, httpResp.Body), nil
}

// ollamaStream implements warp.Stream for Ollama's streaming format.
//
// Ollama uses newline-delimited JSON, not Server-Sent Events.
// Each line is a complete JSON object with a delta.
//
// Thread Safety: ollamaStream is NOT safe for concurrent use.
// Only one goroutine should call Recv() at a time.
type ollamaStream struct {
	reader *bufio.Reader
	closer io.Closer
	ctx    context.Context
	err    error // Cached error for subsequent Recv calls
}

// ollamaStreamChunk represents a chunk in Ollama's streaming response.
type ollamaStreamChunk struct {
	Model     string        `json:"model"`
	CreatedAt string        `json:"created_at"`
	Message   ollamaMessage `json:"message"`
	Done      bool          `json:"done"`
}

// newOllamaStream creates a new Ollama stream from an HTTP response body.
//
// The stream will parse newline-delimited JSON and return them as
// CompletionChunk objects.
func newOllamaStream(ctx context.Context, body io.ReadCloser) warp.Stream {
	return &ollamaStream{
		reader: bufio.NewReader(body),
		closer: body,
		ctx:    ctx,
	}
}

// Recv receives the next chunk from the stream.
//
// Returns io.EOF when the stream is complete (after receiving done=true).
// Returns other errors for failure conditions.
//
// After receiving io.EOF or any error, subsequent calls will return the same error.
func (s *ollamaStream) Recv() (*warp.CompletionChunk, error) {
	// Return cached error if we've already failed or completed
	if s.err != nil {
		return nil, s.err
	}

	for {
		// Check context cancellation
		select {
		case <-s.ctx.Done():
			s.err = s.ctx.Err()
			return nil, s.err
		default:
		}

		// Read line
		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				s.err = io.EOF
				return nil, io.EOF
			}
			s.err = fmt.Errorf("failed to read line: %w", err)
			return nil, s.err
		}

		// Trim whitespace
		line = bytes.TrimSpace(line)

		// Skip empty lines
		if len(line) == 0 {
			continue
		}

		// Parse JSON chunk
		var ollamaChunk ollamaStreamChunk
		if err := json.Unmarshal(line, &ollamaChunk); err != nil {
			s.err = fmt.Errorf("failed to parse chunk: %w", err)
			return nil, s.err
		}

		// Convert to Warp format
		chunk := &warp.CompletionChunk{
			ID:      "ollama-" + ollamaChunk.CreatedAt,
			Object:  "chat.completion.chunk",
			Created: 0,
			Model:   ollamaChunk.Model,
			Choices: []warp.ChunkChoice{
				{
					Index: 0,
					Delta: warp.MessageDelta{
						Role:    ollamaChunk.Message.Role,
						Content: ollamaChunk.Message.Content,
					},
				},
			},
		}

		// Set finish reason if done
		if ollamaChunk.Done {
			finishReason := "stop"
			chunk.Choices[0].FinishReason = &finishReason
			// Return this final chunk, then EOF on next call
			s.err = io.EOF
		}

		return chunk, nil
	}
}

// Close closes the stream and releases resources.
//
// It is safe to call Close multiple times.
// Close must be called even if Recv returns an error.
func (s *ollamaStream) Close() error {
	return s.closer.Close()
}
