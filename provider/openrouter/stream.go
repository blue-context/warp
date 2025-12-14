package openrouter

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

// CompletionStream sends a streaming chat completion request to OpenRouter.
//
// This method returns a Stream that delivers response chunks incrementally
// as they become available from the server. OpenRouter uses the same SSE
// (Server-Sent Events) format as OpenAI.
//
// The caller must close the returned stream to release resources.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "anthropic/claude-opus-4",
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
	// Validate model name
	if req.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	// Transform request to OpenRouter format (OpenAI-compatible)
	openrouterReq := transformRequest(req)

	// Enable streaming for this request
	openrouterReq["stream"] = true

	// Marshal to JSON
	body, err := json.Marshal(openrouterReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal stream request for model %s: %w", req.Model, err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create stream request for model %s: %w", req.Model, err)
	}

	// Set standard headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	// Set optional custom headers for rankings and analytics
	if p.httpReferer != "" {
		httpReq.Header.Set("HTTP-Referer", p.httpReferer)
	}
	if p.appTitle != "" {
		httpReq.Header.Set("X-Title", p.appTitle)
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send stream request to model %s: %w", req.Model, err)
	}

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		defer httpResp.Body.Close()
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("openrouter", httpResp.StatusCode, body, nil)
	}

	// Create SSE stream (OpenAI-compatible)
	return newSSEStream(ctx, httpResp.Body), nil
}

// sseStream implements warp.Stream for Server-Sent Events.
//
// This type parses SSE formatted responses from OpenRouter's streaming API
// and converts them into CompletionChunk objects. OpenRouter uses the same
// SSE format as OpenAI.
//
// Thread Safety: sseStream is NOT safe for concurrent use.
// Only one goroutine should call Recv() at a time.
type sseStream struct {
	reader *bufio.Reader
	closer io.Closer
	ctx    context.Context
	err    error // Cached error for subsequent Recv calls
}

// newSSEStream creates a new SSE stream from an HTTP response body.
//
// The stream will parse Server-Sent Events in OpenAI-compatible format
// and return them as CompletionChunk objects.
func newSSEStream(ctx context.Context, body io.ReadCloser) warp.Stream {
	return &sseStream{
		reader: bufio.NewReader(body),
		closer: body,
		ctx:    ctx,
	}
}

// Recv receives the next chunk from the stream.
//
// Returns io.EOF when the stream is complete (after receiving [DONE] marker).
// Returns other errors for failure conditions.
//
// After receiving io.EOF or any error, subsequent calls will return the same error.
func (s *sseStream) Recv() (*warp.CompletionChunk, error) {
	// Return cached error if we've already failed or completed
	if s.err != nil {
		return nil, s.err
	}

	for {
		// Check context cancellation BEFORE blocking read
		select {
		case <-s.ctx.Done():
			s.err = s.ctx.Err()
			return nil, s.err
		default:
		}

		// Read line (blocking operation)
		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				s.err = io.EOF
				return nil, io.EOF
			}
			s.err = fmt.Errorf("failed to read line: %w", err)
			return nil, s.err
		}

		// Check context again after blocking read
		select {
		case <-s.ctx.Done():
			s.err = s.ctx.Err()
			return nil, s.err
		default:
		}

		// Trim whitespace
		line = bytes.TrimSpace(line)

		// Skip empty lines
		if len(line) == 0 {
			continue
		}

		// Parse SSE field - must have "data: " prefix
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Extract data after "data: " prefix
		data := bytes.TrimPrefix(line, []byte("data: "))

		// Check for [DONE] marker (OpenAI-compatible)
		if bytes.Equal(data, []byte("[DONE]")) {
			s.err = io.EOF
			return nil, io.EOF
		}

		// Parse JSON chunk
		var chunk warp.CompletionChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			s.err = fmt.Errorf("failed to parse chunk: %w", err)
			return nil, s.err
		}

		return &chunk, nil
	}
}

// Close closes the stream and releases resources.
//
// It is safe to call Close multiple times.
// Close must be called even if Recv returns an error.
func (s *sseStream) Close() error {
	if s.closer == nil {
		return nil
	}
	return s.closer.Close()
}
