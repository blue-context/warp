package bedrock

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/blue-context/warp"
)

// CompletionStream sends a streaming completion request to AWS Bedrock.
//
// Bedrock uses event stream format for streaming responses. The stream delivers
// response chunks incrementally as they become available.
//
// The returned stream must be closed by the caller when done.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "claude-3-opus",
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
			Message:  "request cannot be nil",
			Provider: "bedrock",
		}
	}

	if req.Model == "" {
		return nil, &warp.WarpError{
			Message:  "model is required",
			Provider: "bedrock",
		}
	}

	// Get Bedrock model ID
	modelID, err := getModelID(req.Model)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("invalid model: %v", err),
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Determine model family
	family := detectModelFamily(modelID)
	if family == familyUnknown {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("unsupported model family for model: %s", modelID),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}

	// Check if model supports streaming
	if !supportsStreaming(family) {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("model family %s does not support streaming", family),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}

	// Transform request based on model family
	var bedrockReq map[string]interface{}
	switch family {
	case familyClaude:
		bedrockReq = transformClaudeRequest(req)
	case familyLlama:
		bedrockReq = transformLlamaRequest(req)
	case familyTitan:
		bedrockReq = transformTitanRequest(req)
	case familyCohere:
		bedrockReq = transformCohereRequest(req)
	default:
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("unsupported model family: %s", family),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}

	// Marshal request body
	body, err := json.Marshal(bedrockReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to marshal request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Build streaming endpoint URL
	endpoint := p.buildEndpoint(modelID, true)

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to create HTTP request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/vnd.amazon.eventstream")

	// Sign request with AWS Signature V4 (session token is included by signer)
	if err := p.signer.SignRequest(httpReq, body); err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to sign request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to send request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Check for errors
	if httpResp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, warp.ParseProviderError("bedrock", httpResp.StatusCode, bodyBytes, nil)
	}

	// Create stream based on model family
	switch family {
	case familyClaude:
		return newClaudeStream(httpResp, req.Model), nil
	default:
		httpResp.Body.Close()
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("streaming not yet implemented for model family: %s", family),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}
}

// claudeStream implements warp.Stream for Claude model streaming responses.
//
// Thread Safety: claudeStream is safe for concurrent Close() calls but Recv()
// should only be called from a single goroutine (standard practice for streams).
type claudeStream struct {
	resp    *http.Response
	scanner *bufio.Scanner
	model   string

	mu     sync.Mutex // Protects err and closed fields
	err    error
	closed bool
}

// newClaudeStream creates a new Claude stream.
func newClaudeStream(resp *http.Response, model string) *claudeStream {
	return &claudeStream{
		resp:    resp,
		scanner: bufio.NewScanner(resp.Body),
		model:   model,
	}
}

// Recv receives the next chunk from the Claude stream.
//
// Bedrock Claude streaming uses event stream format with different event types:
//   - messageStart: Start of message
//   - contentBlockStart: Start of content block
//   - contentBlockDelta: Incremental content
//   - contentBlockStop: End of content block
//   - messageStop: End of message
//   - metadata: Token usage information
func (s *claudeStream) Recv() (*warp.CompletionChunk, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, io.EOF
	}

	if s.err != nil {
		err := s.err
		s.mu.Unlock()
		return nil, err
	}
	s.mu.Unlock()

	// Read lines until we get a content delta or reach end
	for s.scanner.Scan() {
		line := s.scanner.Text()

		// Skip empty lines
		if line == "" {
			continue
		}

		// Parse event stream line
		chunk, err := s.parseEventStreamLine(line)
		if err != nil {
			s.mu.Lock()
			s.err = err
			s.mu.Unlock()
			return nil, err
		}

		// Return chunk if it has content
		if chunk != nil {
			return chunk, nil
		}
	}

	// Check for scanner errors
	if err := s.scanner.Err(); err != nil {
		s.mu.Lock()
		s.err = err
		s.mu.Unlock()
		return nil, &warp.WarpError{
			Message:       "failed to read stream",
			Provider:      "bedrock",
			Model:         s.model,
			OriginalError: err,
		}
	}

	// Stream ended
	return nil, io.EOF
}

// Close closes the stream and releases resources.
//
// Close is safe to call multiple times and from multiple goroutines.
func (s *claudeStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.closed = true
	return s.resp.Body.Close()
}

// parseEventStreamLine parses a single event stream line.
//
// Bedrock event stream format (simplified):
//
//	data: {"type": "contentBlockDelta", "delta": {"type": "text_delta", "text": "Hello"}}
func (s *claudeStream) parseEventStreamLine(line string) (*warp.CompletionChunk, error) {
	// Check for data prefix
	if !bytes.HasPrefix([]byte(line), []byte("data:")) {
		// Skip non-data lines (event type, etc.)
		return nil, nil
	}

	// Extract JSON data (everything after "data: ")
	jsonData := line[5:]
	if jsonData == "" {
		return nil, nil
	}

	// Parse event
	var event struct {
		Type  string `json:"type"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"delta"`
		Message struct {
			StopReason string `json:"stop_reason"`
		} `json:"message"`
	}

	if err := json.Unmarshal([]byte(jsonData), &event); err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to parse event stream",
			Provider:      "bedrock",
			Model:         s.model,
			OriginalError: err,
		}
	}

	// Handle different event types
	switch event.Type {
	case "contentBlockDelta":
		// Content delta - return chunk
		if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
			return &warp.CompletionChunk{
				ID:      "",
				Object:  "chat.completion.chunk",
				Created: 0,
				Model:   s.model,
				Choices: []warp.ChunkChoice{
					{
						Index: 0,
						Delta: warp.MessageDelta{
							Content: event.Delta.Text,
						},
					},
				},
			}, nil
		}

	case "messageStop":
		// End of message - return final chunk with finish reason
		finishReason := "stop"
		if event.Message.StopReason != "" {
			switch event.Message.StopReason {
			case "end_turn":
				finishReason = "stop"
			case "max_tokens":
				finishReason = "length"
			case "stop_sequence":
				finishReason = "stop"
			}
		}

		return &warp.CompletionChunk{
			ID:      "",
			Object:  "chat.completion.chunk",
			Created: 0,
			Model:   s.model,
			Choices: []warp.ChunkChoice{
				{
					Index:        0,
					Delta:        warp.MessageDelta{},
					FinishReason: &finishReason,
				},
			},
		}, nil

	case "messageStart", "contentBlockStart", "contentBlockStop", "metadata":
		// Skip these events
		return nil, nil
	}

	// Unknown event type - skip
	return nil, nil
}
