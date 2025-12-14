package bedrock

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"

	"github.com/blue-context/warp"
)

func TestCompletionStream(t *testing.T) {
	tests := []struct {
		name     string
		req      *warp.CompletionRequest
		mockResp *http.Response
		mockErr  error
		wantErr  bool
		errMsg   string
	}{
		{
			name: "successful claude stream",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewBufferString("data: {\"type\":\"messageStart\"}\n")),
			},
			wantErr: false,
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
			errMsg:  "request cannot be nil",
		},
		{
			name: "empty model",
			req: &warp.CompletionRequest{
				Model:  "",
			},
			wantErr: true,
			errMsg:  "model is required",
		},
		{
			name: "invalid model",
			req: &warp.CompletionRequest{
				Model:  "invalid-model",
			},
			wantErr: true,
		},
		{
			name: "http error",
			req: &warp.CompletionRequest{
				Model: "claude-3-opus",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: &http.Response{
				StatusCode: 401,
				Body:       io.NopCloser(bytes.NewBufferString(`{"error":{"message":"Invalid credentials"}}`)),
			},
			wantErr: true,
		},
		{
			name: "unsupported streaming model",
			req: &warp.CompletionRequest{
				Model: "stability.stable-diffusion-xl-v1",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			mockResp: &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewBufferString("")),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockHTTPClient{
				doFunc: func(req *http.Request) (*http.Response, error) {
					if tt.mockErr != nil {
						return nil, tt.mockErr
					}
					return tt.mockResp, nil
				},
			}

			provider, err := NewProvider(
				WithCredentials("test-key", "test-secret"),
				WithRegion("us-east-1"),
				WithHTTPClient(mockClient),
			)
			if err != nil {
				t.Fatalf("NewProvider() error = %v", err)
			}

			ctx := context.Background()
			stream, err := provider.CompletionStream(ctx, tt.req)

			if (err != nil) != tt.wantErr {
				t.Errorf("CompletionStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				if tt.errMsg != "" {
					liteLLMErr, ok := err.(*warp.WarpError)
					if ok && liteLLMErr.Message != tt.errMsg {
						t.Errorf("error message = %q, want %q", liteLLMErr.Message, tt.errMsg)
					}
				}
				return
			}

			if stream == nil {
				t.Error("stream is nil")
				return
			}

			// Clean up
			stream.Close()
		})
	}
}

func TestClaudeStreamRecv(t *testing.T) {
	tests := []struct {
		name        string
		streamData  string
		wantChunks  int
		wantContent string
		wantErr     bool
	}{
		{
			name: "content delta event",
			streamData: `data: {"type":"contentBlockDelta","delta":{"type":"text_delta","text":"Hello"}}

`,
			wantChunks:  1,
			wantContent: "Hello",
			wantErr:     false,
		},
		{
			name: "message stop event",
			streamData: `data: {"type":"messageStop","message":{"stop_reason":"end_turn"}}

`,
			wantChunks: 1,
			wantErr:    false,
		},
		{
			name: "multiple events",
			streamData: `data: {"type":"messageStart"}
data: {"type":"contentBlockStart"}
data: {"type":"contentBlockDelta","delta":{"type":"text_delta","text":"Hi"}}
data: {"type":"contentBlockDelta","delta":{"type":"text_delta","text":" there"}}
data: {"type":"messageStop","message":{"stop_reason":"end_turn"}}

`,
			wantChunks:  3,
			wantContent: "Hi",
			wantErr:     false,
		},
		{
			name:       "empty stream",
			streamData: "",
			wantChunks: 0,
			wantErr:    false,
		},
		{
			name:       "skip non-data lines",
			streamData: "event: contentBlockDelta\ndata: {\"type\":\"contentBlockDelta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Test\"}}\n\n",
			wantChunks: 1,
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewBufferString(tt.streamData)),
			}

			stream := newClaudeStream(resp, "claude-3-opus")

			chunks := 0
			var firstContent string

			for {
				chunk, err := stream.Recv()
				if err == io.EOF {
					break
				}

				if err != nil {
					if !tt.wantErr {
						t.Errorf("unexpected error: %v", err)
					}
					break
				}

				chunks++

				// Capture first content
				if firstContent == "" && len(chunk.Choices) > 0 {
					firstContent = chunk.Choices[0].Delta.Content
				}
			}

			if !tt.wantErr {
				if chunks != tt.wantChunks {
					t.Errorf("received %d chunks, want %d", chunks, tt.wantChunks)
				}

				if tt.wantContent != "" && firstContent != tt.wantContent {
					t.Errorf("first content = %q, want %q", firstContent, tt.wantContent)
				}
			}

			stream.Close()
		})
	}
}

func TestClaudeStreamClose(t *testing.T) {
	resp := &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewBufferString("data: test\n")),
	}

	stream := newClaudeStream(resp, "claude-3-opus")

	// Close once
	err := stream.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// Close again (should be safe)
	err = stream.Close()
	if err != nil {
		t.Errorf("second Close() error = %v", err)
	}

	// Recv after close should return EOF
	_, err = stream.Recv()
	if err != io.EOF {
		t.Errorf("Recv() after close should return EOF, got %v", err)
	}
}

func TestParseEventStreamLine(t *testing.T) {
	stream := &claudeStream{
		model: "claude-3-opus",
	}

	tests := []struct {
		name      string
		line      string
		wantChunk bool
		wantNil   bool
		wantErr   bool
	}{
		{
			name:      "non-data line",
			line:      "event: test",
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:      "empty data",
			line:      "data:",
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:      "content delta",
			line:      `data: {"type":"contentBlockDelta","delta":{"type":"text_delta","text":"test"}}`,
			wantChunk: true,
		},
		{
			name:      "message stop",
			line:      `data: {"type":"messageStop","message":{"stop_reason":"end_turn"}}`,
			wantChunk: true,
		},
		{
			name:      "message start (skipped)",
			line:      `data: {"type":"messageStart"}`,
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:      "content block start (skipped)",
			line:      `data: {"type":"contentBlockStart"}`,
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:      "content block stop (skipped)",
			line:      `data: {"type":"contentBlockStop"}`,
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:      "metadata (skipped)",
			line:      `data: {"type":"metadata"}`,
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:      "unknown event type",
			line:      `data: {"type":"unknown"}`,
			wantChunk: false,
			wantNil:   true,
		},
		{
			name:    "invalid json",
			line:    `data: {invalid}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunk, err := stream.parseEventStreamLine(tt.line)

			if (err != nil) != tt.wantErr {
				t.Errorf("parseEventStreamLine() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantNil {
				if chunk != nil {
					t.Error("expected nil chunk")
				}
			} else if tt.wantChunk {
				if chunk == nil {
					t.Error("expected non-nil chunk")
				}
			}
		})
	}
}

func TestClaudeStreamFinishReasons(t *testing.T) {
	tests := []struct {
		name       string
		stopReason string
		want       string
	}{
		{
			name:       "end_turn",
			stopReason: "end_turn",
			want:       "stop",
		},
		{
			name:       "max_tokens",
			stopReason: "max_tokens",
			want:       "length",
		},
		{
			name:       "stop_sequence",
			stopReason: "stop_sequence",
			want:       "stop",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			streamData := `data: {"type":"messageStop","message":{"stop_reason":"` + tt.stopReason + `"}}

`
			resp := &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewBufferString(streamData)),
			}

			stream := newClaudeStream(resp, "claude-3-opus")

			chunk, err := stream.Recv()
			if err != nil {
				t.Fatalf("Recv() error = %v", err)
			}

			if chunk == nil || len(chunk.Choices) == 0 {
				t.Fatal("no chunk received")
			}

			if chunk.Choices[0].FinishReason == nil {
				t.Fatal("finish_reason is nil")
			}

			got := *chunk.Choices[0].FinishReason
			if got != tt.want {
				t.Errorf("finish_reason = %q, want %q", got, tt.want)
			}

			stream.Close()
		})
	}
}

func TestCompletionStreamUnsupportedFamily(t *testing.T) {
	mockClient := &mockHTTPClient{
		doFunc: func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewBufferString("")),
			}, nil
		},
	}

	provider, err := NewProvider(
		WithCredentials("test-key", "test-secret"),
		WithRegion("us-east-1"),
		WithHTTPClient(mockClient),
	)
	if err != nil {
		t.Fatalf("NewProvider() error = %v", err)
	}

	ctx := context.Background()

	// Test Llama streaming (not yet implemented)
	req := &warp.CompletionRequest{
		Model: "llama3-70b",
		Messages: []warp.Message{
			{Role: "user", Content: "Hello"},
		},
	}

	stream, err := provider.CompletionStream(ctx, req)
	if err == nil {
		stream.Close()
		t.Error("expected error for unsupported streaming family")
	}
}
