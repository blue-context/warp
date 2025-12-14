package vllm

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

// CompletionStream sends a streaming chat completion request to vLLM.
//
// This method returns a Stream that delivers response chunks incrementally
// as they become available from the server.
//
// The caller must close the returned stream to release resources.
//
// Uses vLLM's native /inference/v1/generate endpoint with streaming enabled.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "meta-llama/Llama-2-7b-hf",
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
	// Check context cancellation before starting
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Transform request to vLLM format (streaming)
	vllmReq := transformToVLLMRequest(req, true)

	// Marshal to JSON
	body, err := json.Marshal(vllmReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request to vLLM's native generate endpoint
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/inference/v1/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	// Add optional API key if configured
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		defer httpResp.Body.Close()
		body, err := io.ReadAll(httpResp.Body)
		if err != nil {
			body = []byte("failed to read error response")
		}
		return nil, warp.ParseProviderError("vllm", httpResp.StatusCode, body, nil)
	}

	// Create vLLM stream (Server-Sent Events format)
	return newVLLMStream(ctx, httpResp.Body), nil
}

// vllmStream implements warp.Stream for vLLM's streaming format.
//
// vLLM uses Server-Sent Events (SSE) format for streaming.
// Each event contains a JSON chunk with incremental content.
//
// Thread Safety: vllmStream is NOT safe for concurrent use.
// Only one goroutine should call Recv() at a time.
type vllmStream struct {
	reader *bufio.Reader
	closer io.Closer
	ctx    context.Context
	err    error // Cached error for subsequent Recv calls
}

// vllmStreamChunk represents a chunk in vLLM's streaming response.
//
// vLLM returns OpenAI-compatible streaming chunks.
type vllmStreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []vllmStreamChoice  `json:"choices"`
	Usage   *vllmUsage          `json:"usage,omitempty"`
}

// vllmStreamChoice represents a single choice in a streaming chunk.
type vllmStreamChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

// newVLLMStream creates a new vLLM stream from an HTTP response body.
//
// The stream will parse Server-Sent Events and return them as
// CompletionChunk objects.
func newVLLMStream(ctx context.Context, body io.ReadCloser) warp.Stream {
	return &vllmStream{
		reader: bufio.NewReader(body),
		closer: body,
		ctx:    ctx,
	}
}

// Recv receives the next chunk from the stream.
//
// Returns io.EOF when the stream is complete (after receiving [DONE] message).
// Returns other errors for failure conditions.
//
// After receiving io.EOF or any error, subsequent calls will return the same error.
func (s *vllmStream) Recv() (*warp.CompletionChunk, error) {
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

		// Check for SSE data prefix
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Extract data after "data: " prefix
		data := bytes.TrimPrefix(line, []byte("data: "))

		// Check for [DONE] message
		if bytes.Equal(data, []byte("[DONE]")) {
			s.err = io.EOF
			return nil, io.EOF
		}

		// Parse JSON chunk
		var vllmChunk vllmStreamChunk
		if err := json.Unmarshal(data, &vllmChunk); err != nil {
			s.err = fmt.Errorf("failed to parse chunk: %w", err)
			return nil, s.err
		}

		// Convert to Warp format
		chunk := transformVLLMStreamChunk(&vllmChunk)

		return chunk, nil
	}
}

// transformVLLMStreamChunk transforms a vLLM stream chunk to Warp format.
//
// vLLM uses text completion format, so we convert to chat completion chunk format.
func transformVLLMStreamChunk(vllmChunk *vllmStreamChunk) *warp.CompletionChunk {
	// Transform choices
	choices := make([]warp.ChunkChoice, len(vllmChunk.Choices))
	for i, vllmChoice := range vllmChunk.Choices {
		choice := warp.ChunkChoice{
			Index: vllmChoice.Index,
			Delta: warp.MessageDelta{
				Content: vllmChoice.Text,
			},
			FinishReason: vllmChoice.FinishReason,
		}

		// Set role in first chunk
		if vllmChoice.Text != "" && i == 0 {
			choice.Delta.Role = "assistant"
		}

		choices[i] = choice
	}

	// Transform usage (only present in final chunk)
	var usage *warp.Usage
	if vllmChunk.Usage != nil {
		usage = &warp.Usage{
			PromptTokens:     vllmChunk.Usage.PromptTokens,
			CompletionTokens: vllmChunk.Usage.CompletionTokens,
			TotalTokens:      vllmChunk.Usage.TotalTokens,
		}
	}

	return &warp.CompletionChunk{
		ID:      vllmChunk.ID,
		Object:  "chat.completion.chunk",
		Created: vllmChunk.Created,
		Model:   vllmChunk.Model,
		Choices: choices,
		Usage:   usage,
	}
}

// Close closes the stream and releases resources.
//
// It is safe to call Close multiple times.
// Close must be called even if Recv returns an error.
func (s *vllmStream) Close() error {
	if s.closer == nil {
		return nil
	}
	return s.closer.Close()
}
