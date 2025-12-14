package testutil

import (
	"io"

	"github.com/blue-context/warp"
)

// MockStream is a mock stream for testing streaming responses.
//
// It returns pre-configured chunks in order, then io.EOF when exhausted.
// This is useful for testing streaming completion logic without making
// actual API calls.
//
// Example:
//
//	stream := testutil.NewMockStream(
//	    &warp.CompletionChunk{
//	        Choices: []warp.ChunkChoice{
//	            {Delta: warp.MessageDelta{Content: "Hello"}},
//	        },
//	    },
//	    &warp.CompletionChunk{
//	        Choices: []warp.ChunkChoice{
//	            {Delta: warp.MessageDelta{Content: " world"}},
//	        },
//	    },
//	)
//	defer stream.Close()
//
//	for {
//	    chunk, err := stream.Recv()
//	    if err == io.EOF {
//	        break
//	    }
//	    // Process chunk...
//	}
type MockStream struct {
	chunks []*warp.CompletionChunk
	index  int
	closed bool
}

// NewMockStream creates a new mock stream with the given chunks.
//
// The chunks will be returned in order by successive calls to Recv().
// After all chunks are exhausted, Recv() returns io.EOF.
func NewMockStream(chunks ...*warp.CompletionChunk) *MockStream {
	return &MockStream{
		chunks: chunks,
		index:  0,
	}
}

// NewMockStreamWithChunks is an alias for NewMockStream for convenience.
//
// This function provides a more explicit name when creating test streams.
func NewMockStreamWithChunks(chunks []*warp.CompletionChunk) *MockStream {
	return &MockStream{
		chunks: chunks,
		index:  0,
	}
}

// Recv returns the next chunk or io.EOF when exhausted.
//
// Once the stream is closed or all chunks are consumed, returns io.EOF.
// Subsequent calls after io.EOF continue to return io.EOF.
func (m *MockStream) Recv() (*warp.CompletionChunk, error) {
	if m.closed {
		return nil, io.EOF
	}
	if m.index >= len(m.chunks) {
		return nil, io.EOF
	}
	chunk := m.chunks[m.index]
	m.index++
	return chunk, nil
}

// Close closes the stream.
//
// After Close is called, Recv will return io.EOF.
// It is safe to call Close multiple times.
func (m *MockStream) Close() error {
	m.closed = true
	return nil
}
