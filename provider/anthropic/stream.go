package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/blue-context/warp"
)

// CompletionStream sends a streaming chat completion request to Anthropic.
//
// This method returns a Stream that delivers response chunks incrementally
// as they become available from the server.
//
// The caller must close the returned stream to release resources.
//
// Example:
//
//	stream, err := provider.CompletionStream(ctx, &warp.CompletionRequest{
//	    Model: "claude-3-opus-20240229",
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
	// Transform request to Anthropic format
	anthropicReq, err := transformRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to transform request: %w", err)
	}

	// Enable streaming for this request
	anthropicReq.Stream = true

	// Marshal to JSON
	body, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set Anthropic-specific headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.apiVersion)
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
		return nil, warp.ParseProviderError("anthropic", httpResp.StatusCode, body, nil)
	}

	// Create SSE stream
	return newAnthropicStream(ctx, httpResp.Body, req.Model), nil
}

// anthropicStreamEvent represents an Anthropic streaming event.
type anthropicStreamEvent struct {
	Type         string                 `json:"type"`
	Message      *anthropicResponse     `json:"message,omitempty"`
	Index        int                    `json:"index,omitempty"`
	ContentBlock *anthropicContentBlock `json:"content_block,omitempty"`
	Delta        *anthropicDelta        `json:"delta,omitempty"`
	Usage        *anthropicUsage        `json:"usage,omitempty"`
}

// anthropicDelta represents incremental content updates.
type anthropicDelta struct {
	Type         string  `json:"type"`
	Text         string  `json:"text,omitempty"`
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
}

// anthropicStream implements warp.Stream for Anthropic's Server-Sent Events.
//
// This type parses Anthropic's SSE formatted responses and converts them
// into CompletionChunk objects compatible with Warp's streaming interface.
//
// Anthropic uses a multi-event streaming protocol:
// - message_start: Initial message metadata
// - content_block_start: Start of a content block
// - content_block_delta: Incremental text updates
// - message_delta: Message-level updates
// - message_stop: End of stream
//
// Thread Safety: anthropicStream is NOT safe for concurrent use.
// Only one goroutine should call Recv() at a time.
type anthropicStream struct {
	reader       *bufio.Reader
	closer       io.Closer
	ctx          context.Context
	err          error // Cached error for subsequent Recv calls
	model        string
	messageID    string
	currentIndex int
	created      int64
}

// newAnthropicStream creates a new Anthropic SSE stream from an HTTP response body.
//
// The stream will parse Server-Sent Events and return them as
// CompletionChunk objects.
func newAnthropicStream(ctx context.Context, body io.ReadCloser, model string) warp.Stream {
	return &anthropicStream{
		reader:  bufio.NewReader(body),
		closer:  body,
		ctx:     ctx,
		model:   model,
		created: time.Now().Unix(),
	}
}

// Recv receives the next chunk from the stream.
//
// Returns io.EOF when the stream is complete (after receiving message_stop event).
// Returns other errors for failure conditions.
//
// After receiving io.EOF or any error, subsequent calls will return the same error.
func (s *anthropicStream) Recv() (*warp.CompletionChunk, error) {
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

		// Parse SSE field
		if !bytes.HasPrefix(line, []byte("event: ")) && !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		// Handle event type line
		if bytes.HasPrefix(line, []byte("event: ")) {
			// Store event type but continue reading for data
			continue
		}

		// Extract data after "data: " prefix
		data := bytes.TrimPrefix(line, []byte("data: "))

		// Parse JSON event
		var event anthropicStreamEvent
		if err := json.Unmarshal(data, &event); err != nil {
			// Skip malformed events
			continue
		}

		// Process event based on type
		chunk := s.processEvent(&event)
		if chunk != nil {
			return chunk, nil
		}

		// Check for stream completion
		if event.Type == "message_stop" {
			s.err = io.EOF
			return nil, io.EOF
		}
	}
}

// processEvent processes an Anthropic stream event and converts it to a CompletionChunk.
//
// Returns nil for events that don't produce chunks (like message_start).
func (s *anthropicStream) processEvent(event *anthropicStreamEvent) *warp.CompletionChunk {
	switch event.Type {
	case "message_start":
		// Store message metadata
		if event.Message != nil {
			s.messageID = event.Message.ID
		}
		// Return initial chunk with role
		return &warp.CompletionChunk{
			ID:      s.messageID,
			Object:  "chat.completion.chunk",
			Created: s.created,
			Model:   s.model,
			Choices: []warp.ChunkChoice{
				{
					Index: 0,
					Delta: warp.MessageDelta{
						Role: "assistant",
					},
				},
			},
		}

	case "content_block_start":
		// Track the content block index
		s.currentIndex = event.Index
		return nil

	case "content_block_delta":
		// Return incremental text content
		if event.Delta != nil && event.Delta.Type == "text_delta" {
			return &warp.CompletionChunk{
				ID:      s.messageID,
				Object:  "chat.completion.chunk",
				Created: s.created,
				Model:   s.model,
				Choices: []warp.ChunkChoice{
					{
						Index: s.currentIndex,
						Delta: warp.MessageDelta{
							Content: event.Delta.Text,
						},
					},
				},
			}
		}
		return nil

	case "message_delta":
		// Return final chunk with finish reason
		if event.Delta != nil && event.Delta.StopReason != "" {
			finishReason := mapStopReason(event.Delta.StopReason)
			return &warp.CompletionChunk{
				ID:      s.messageID,
				Object:  "chat.completion.chunk",
				Created: s.created,
				Model:   s.model,
				Choices: []warp.ChunkChoice{
					{
						Index:        0,
						Delta:        warp.MessageDelta{},
						FinishReason: &finishReason,
					},
				},
				Usage: s.buildUsage(event.Usage),
			}
		}
		return nil

	case "content_block_stop":
		// Content block ended, no chunk to return
		return nil

	default:
		// Unknown event type, skip
		return nil
	}
}

// buildUsage converts Anthropic usage to Warp usage format.
func (s *anthropicStream) buildUsage(usage *anthropicUsage) *warp.Usage {
	if usage == nil {
		return nil
	}
	return &warp.Usage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.InputTokens + usage.OutputTokens,
	}
}

// Close closes the stream and releases resources.
//
// It is safe to call Close multiple times.
// Close must be called even if Recv returns an error.
func (s *anthropicStream) Close() error {
	return s.closer.Close()
}
