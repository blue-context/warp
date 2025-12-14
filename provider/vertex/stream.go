package vertex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/blue-context/warp"
)

// CompletionStream sends a streaming chat completion request to Vertex AI.
//
// This method implements the provider.Provider interface for streaming completions.
// Vertex AI uses server-sent events (SSE) for streaming responses.
//
// The returned Stream must be closed by the caller to release resources.
//
// Thread Safety: This method is safe for concurrent use.
// However, the returned Stream is NOT safe for concurrent use.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "gemini-pro",
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
	if req == nil {
		return nil, &warp.WarpError{
			Message:  "completion request cannot be nil",
			Provider: "vertex",
		}
	}

	// Get OAuth2 access token
	token, err := p.tokenProvider.GetToken()
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to get access token: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Build Vertex AI streaming endpoint URL
	url := p.buildEndpoint(req.Model, true)

	// Transform request to Vertex AI format
	vertexReq, err := transformRequest(req)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to transform request: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Marshal request body
	body, err := json.Marshal(vertexReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to marshal request: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to create HTTP request: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("HTTP request failed: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Check HTTP status before starting stream
	if httpResp.StatusCode != http.StatusOK {
		defer httpResp.Body.Close()
		respBody, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("vertex", httpResp.StatusCode, respBody, nil)
	}

	// Create and return stream
	return &vertexStream{
		model:    req.Model,
		response: httpResp,
		reader:   bufio.NewReader(httpResp.Body),
	}, nil
}

// vertexStream implements the warp.Stream interface for Vertex AI.
//
// Vertex AI uses server-sent events (SSE) format for streaming:
//
//	data: {"candidates": [...], "usageMetadata": {...}}
//	data: {"candidates": [...]}
//	...
//
// Thread Safety: vertexStream is NOT safe for concurrent use.
// Only one goroutine should call Recv() at a time.
type vertexStream struct {
	model    string
	response *http.Response
	reader   *bufio.Reader
	err      error
	closed   bool
}

// Recv receives the next chunk from the stream.
//
// Returns io.EOF when the stream is complete.
// After returning io.EOF or any error, subsequent calls will return the same error.
//
// Thread Safety: NOT safe for concurrent use.
func (s *vertexStream) Recv() (*warp.CompletionChunk, error) {
	// Return cached error if already failed
	if s.err != nil {
		return nil, s.err
	}

	for {
		// Read next line
		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				s.err = io.EOF
				return nil, io.EOF
			}
			s.err = fmt.Errorf("failed to read stream: %w", err)
			return nil, s.err
		}

		// Trim whitespace
		line = bytes.TrimSpace(line)

		// Skip empty lines
		if len(line) == 0 {
			continue
		}

		// Parse SSE line
		// Expected format: "data: {...}"
		if !bytes.HasPrefix(line, []byte("data: ")) {
			// Skip non-data lines (e.g., event: message)
			continue
		}

		// Extract JSON payload (after "data: " prefix)
		jsonData := bytes.TrimPrefix(line, []byte("data: "))

		// Skip special SSE messages
		if bytes.Equal(jsonData, []byte("[DONE]")) {
			s.err = io.EOF
			return nil, io.EOF
		}

		// Parse Vertex AI response chunk
		var vertexResp vertexResponse
		if err := json.Unmarshal(jsonData, &vertexResp); err != nil {
			s.err = fmt.Errorf("failed to parse stream chunk: %w", err)
			return nil, s.err
		}

		// Check for prompt feedback (blocking/safety issues)
		if vertexResp.PromptFeedback != nil && vertexResp.PromptFeedback.BlockReason != "" {
			s.err = warp.NewContentPolicyViolationError(
				fmt.Sprintf("prompt blocked: %s", vertexResp.PromptFeedback.BlockReason),
				"vertex",
				nil,
			)
			return nil, s.err
		}

		// Skip empty chunks (no candidates)
		if len(vertexResp.Candidates) == 0 {
			continue
		}

		// Transform to CompletionChunk
		chunk := transformStreamChunk(&vertexResp, s.model)
		return chunk, nil
	}
}

// Close closes the stream and releases resources.
//
// It is safe to call Close multiple times.
// Close must be called even if Recv returns an error.
//
// Thread Safety: Safe for concurrent use.
func (s *vertexStream) Close() error {
	if s.closed {
		return nil
	}

	s.closed = true

	if s.response != nil && s.response.Body != nil {
		return s.response.Body.Close()
	}

	return nil
}

// transformStreamChunk converts a Vertex AI streaming response to a CompletionChunk.
func transformStreamChunk(vResp *vertexResponse, model string) *warp.CompletionChunk {
	chunk := &warp.CompletionChunk{
		ID:      generateResponseID(),
		Object:  "chat.completion.chunk",
		Created: currentUnixTime(),
		Model:   model,
		Choices: make([]warp.ChunkChoice, 0, len(vResp.Candidates)),
	}

	// Transform candidates to chunk choices
	for _, candidate := range vResp.Candidates {
		choice := warp.ChunkChoice{
			Index: candidate.Index,
			Delta: transformContentDelta(candidate.Content),
		}

		// Set finish reason if present
		if candidate.FinishReason != "" {
			reason := transformFinishReason(candidate.FinishReason)
			choice.FinishReason = &reason
		}

		chunk.Choices = append(chunk.Choices, choice)
	}

	// Add usage metadata in final chunk (if present)
	if vResp.UsageMetadata != nil {
		chunk.Usage = &warp.Usage{
			PromptTokens:     vResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: vResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      vResp.UsageMetadata.TotalTokenCount,
		}
	}

	return chunk
}

// transformContentDelta converts Vertex content to message delta.
func transformContentDelta(content vertexContent) warp.MessageDelta {
	delta := warp.MessageDelta{
		Role: inverseTransformRole(content.Role),
	}

	// Extract text from parts
	var textParts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}

		// Handle function calls in streaming
		if part.FunctionCall != nil {
			argsJSON, _ := json.Marshal(part.FunctionCall.Args)
			delta.ToolCalls = append(delta.ToolCalls, warp.ToolCall{
				ID:   generateToolCallID(),
				Type: "function",
				Function: warp.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Set content delta
	if len(textParts) > 0 {
		delta.Content = strings.Join(textParts, "\n")
	}

	return delta
}
