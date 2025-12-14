package azure

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

// CompletionStream sends a streaming chat completion request to Azure OpenAI.
//
// This method returns a Stream that delivers response chunks incrementally
// as they become available from the server. Azure uses the same SSE format
// as OpenAI for streaming responses.
//
// The caller must close the returned stream to release resources.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "gpt-4", // Model is ignored; deployment name is used instead
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
	// Build Azure URL with deployment name and API version
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		p.apiBase, p.deployment, p.apiVersion)

	// Transform request to Azure format
	azureReq := transformRequest(req)

	// Enable streaming for this request
	azureReq["stream"] = true

	// Marshal to JSON
	body, err := json.Marshal(azureReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set Azure-specific headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", p.apiKey) // Azure uses api-key, not Bearer token
	httpReq.Header.Set("Accept", "text/event-stream")

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	// Check status code
	if httpResp.StatusCode != http.StatusOK {
		defer httpResp.Body.Close()
		body, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("azure", httpResp.StatusCode, body, nil)
	}

	// Create SSE stream (Azure uses same format as OpenAI)
	return newSSEStream(ctx, httpResp.Body), nil
}

// sseStream implements warp.Stream for Server-Sent Events.
//
// This type parses SSE formatted responses from Azure OpenAI's streaming API
// and converts them into CompletionChunk objects. Azure uses the same SSE
// format as OpenAI.
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
// The stream will parse Server-Sent Events and return them as
// CompletionChunk objects.
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

		// Parse SSE field - must have "data: " prefix
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Extract data after "data: " prefix
		data := bytes.TrimPrefix(line, []byte("data: "))

		// Check for [DONE] marker
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
	return s.closer.Close()
}
