package token

import (
	"testing"

	"github.com/blue-context/warp"
)

func TestCountText(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		minToken int // Minimum expected tokens
		maxToken int // Maximum expected tokens (for approximation range)
	}{
		{
			name:     "empty string",
			text:     "",
			minToken: 0,
			maxToken: 0,
		},
		{
			name:     "single word",
			text:     "Hello",
			minToken: 1,
			maxToken: 2,
		},
		{
			name:     "simple sentence",
			text:     "Hello, world!",
			minToken: 3,
			maxToken: 5,
		},
		{
			name:     "sentence with punctuation",
			text:     "Hello, how are you? I'm fine!",
			minToken: 6,
			maxToken: 10,
		},
		{
			name:     "longer text",
			text:     "The quick brown fox jumps over the lazy dog. This is a test sentence.",
			minToken: 15,
			maxToken: 20,
		},
		{
			name:     "text with special characters",
			text:     "Code: func main() { fmt.Println(\"Hello\") }",
			minToken: 9,
			maxToken: 13,
		},
		{
			name:     "single character",
			text:     "a",
			minToken: 1,
			maxToken: 2,
		},
		{
			name:     "whitespace only",
			text:     "   ",
			minToken: 0,
			maxToken: 1,
		},
	}

	counter := NewCounter()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := counter.CountText(tt.text)

			if got < tt.minToken || got > tt.maxToken {
				t.Errorf("CountText(%q) = %d, want between %d and %d",
					tt.text, got, tt.minToken, tt.maxToken)
			}
		})
	}
}

func TestCountMessages(t *testing.T) {
	tests := []struct {
		name     string
		messages []warp.Message
		minToken int
		maxToken int
	}{
		{
			name:     "empty messages",
			messages: []warp.Message{},
			minToken: 3,
			maxToken: 3,
		},
		{
			name: "single user message",
			messages: []warp.Message{
				{Role: "user", Content: "Hello"},
			},
			minToken: 5,
			maxToken: 8,
		},
		{
			name: "conversation",
			messages: []warp.Message{
				{Role: "system", Content: "You are a helpful assistant."},
				{Role: "user", Content: "Hello, how are you?"},
				{Role: "assistant", Content: "I'm doing well, thank you!"},
			},
			minToken: 20,
			maxToken: 35,
		},
		{
			name: "message with name",
			messages: []warp.Message{
				{Role: "user", Name: "Alice", Content: "Hello"},
			},
			minToken: 6,
			maxToken: 10,
		},
		{
			name: "message with tool call",
			messages: []warp.Message{
				{
					Role:    "assistant",
					Content: "",
					ToolCalls: []warp.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: warp.FunctionCall{
								Name:      "get_weather",
								Arguments: `{"location": "San Francisco"}`,
							},
						},
					},
				},
			},
			minToken: 10,
			maxToken: 25,
		},
		{
			name: "tool response message",
			messages: []warp.Message{
				{
					Role:       "tool",
					Content:    "The weather is sunny, 72°F",
					ToolCallID: "call_123",
				},
			},
			minToken: 8,
			maxToken: 15,
		},
	}

	counter := NewCounter()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := counter.CountMessages(tt.messages)

			if got < tt.minToken || got > tt.maxToken {
				t.Errorf("CountMessages() = %d, want between %d and %d",
					got, tt.minToken, tt.maxToken)
			}
		})
	}
}

func TestCountMessagesMultimodal(t *testing.T) {
	tests := []struct {
		name     string
		messages []warp.Message
		minToken int
		maxToken int
	}{
		{
			name: "text only multimodal",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{Type: "text", Text: "What is in this image?"},
					},
				},
			},
			minToken: 8,
			maxToken: 12,
		},
		{
			name: "image with low detail",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{Type: "text", Text: "What is this?"},
						{
							Type: "image_url",
							ImageURL: &warp.ImageURL{
								URL:    "https://example.com/image.jpg",
								Detail: "low",
							},
						},
					},
				},
			},
			minToken: 90,
			maxToken: 100,
		},
		{
			name: "image with high detail",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{Type: "text", Text: "Describe this image in detail."},
						{
							Type: "image_url",
							ImageURL: &warp.ImageURL{
								URL:    "https://example.com/image.jpg",
								Detail: "high",
							},
						},
					},
				},
			},
			minToken: 175,
			maxToken: 185,
		},
		{
			name: "image with auto detail",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{Type: "text", Text: "What do you see?"},
						{
							Type: "image_url",
							ImageURL: &warp.ImageURL{
								URL:    "https://example.com/image.jpg",
								Detail: "auto",
							},
						},
					},
				},
			},
			minToken: 90,
			maxToken: 100,
		},
		{
			name: "multiple images",
			messages: []warp.Message{
				{
					Role: "user",
					Content: []warp.ContentPart{
						{Type: "text", Text: "Compare these images."},
						{
							Type: "image_url",
							ImageURL: &warp.ImageURL{
								URL:    "https://example.com/image1.jpg",
								Detail: "low",
							},
						},
						{
							Type: "image_url",
							ImageURL: &warp.ImageURL{
								URL:    "https://example.com/image2.jpg",
								Detail: "low",
							},
						},
					},
				},
			},
			minToken: 175,
			maxToken: 185,
		},
	}

	counter := NewCounter()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := counter.CountMessages(tt.messages)

			if got < tt.minToken || got > tt.maxToken {
				t.Errorf("CountMessages() = %d, want between %d and %d",
					got, tt.minToken, tt.maxToken)
			}
		})
	}
}

func TestCountRequest(t *testing.T) {
	tests := []struct {
		name     string
		req      *warp.CompletionRequest
		minToken int
		maxToken int
	}{
		{
			name:     "nil request",
			req:      nil,
			minToken: 0,
			maxToken: 0,
		},
		{
			name: "simple request",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Hello"},
				},
			},
			minToken: 5,
			maxToken: 10,
		},
		{
			name: "request with tools",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "What's the weather?"},
				},
				Tools: []warp.Tool{
					{
						Type: "function",
						Function: warp.Function{
							Name:        "get_weather",
							Description: "Get the current weather for a location",
							Parameters: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"location": map[string]any{
										"type":        "string",
										"description": "City name",
									},
									"unit": map[string]any{
										"type": "string",
										"enum": []any{"celsius", "fahrenheit"},
									},
								},
								"required": []any{"location"},
							},
						},
					},
				},
			},
			minToken: 50,
			maxToken: 100,
		},
		{
			name: "request with json response format",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4",
				Messages: []warp.Message{
					{Role: "user", Content: "Return a JSON object"},
				},
				ResponseFormat: &warp.ResponseFormat{
					Type: "json_object",
				},
			},
			minToken: 15,
			maxToken: 25,
		},
		{
			name: "complex request",
			req: &warp.CompletionRequest{
				Model: "openai/gpt-4",
				Messages: []warp.Message{
					{Role: "system", Content: "You are a helpful assistant."},
					{Role: "user", Content: "What's the weather in San Francisco?"},
				},
				Tools: []warp.Tool{
					{
						Type: "function",
						Function: warp.Function{
							Name:        "get_weather",
							Description: "Get the current weather for a location",
							Parameters: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"location": map[string]any{
										"type":        "string",
										"description": "City name",
									},
								},
								"required": []any{"location"},
							},
						},
					},
				},
				ResponseFormat: &warp.ResponseFormat{
					Type: "json_object",
				},
			},
			minToken: 70,
			maxToken: 120,
		},
	}

	counter := NewCounter()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := counter.CountRequest(tt.req)

			if got < tt.minToken || got > tt.maxToken {
				t.Errorf("CountRequest() = %d, want between %d and %d",
					got, tt.minToken, tt.maxToken)
			}
		})
	}
}

func TestCountAccuracy(t *testing.T) {
	tests := []struct {
		name         string
		text         string
		actualTokens int // Known token count from actual tokenizer
		maxDeviation float64
	}{
		{
			name:         "short greeting",
			text:         "Hello, world!",
			actualTokens: 4,
			maxDeviation: 0.30, // Allow 30% deviation for short texts
		},
		{
			name:         "medium sentence",
			text:         "The quick brown fox jumps over the lazy dog.",
			actualTokens: 10,
			maxDeviation: 0.20, // 20% for medium texts
		},
		{
			name:         "longer text",
			text:         "This is a longer piece of text that contains multiple sentences. It should help us test the accuracy of our token counting algorithm.",
			actualTokens: 28,
			maxDeviation: 0.15, // Better accuracy on longer texts
		},
		{
			name:         "code snippet",
			text:         "func main() { fmt.Println(\"Hello, world!\") }",
			actualTokens: 14,
			maxDeviation: 0.30, // Code may have higher deviation
		},
	}

	counter := NewCounter()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := counter.CountText(tt.text)

			// Calculate deviation
			deviation := float64(abs(got-tt.actualTokens)) / float64(tt.actualTokens)

			if deviation > tt.maxDeviation {
				t.Errorf("CountText(%q) = %d (actual: %d), deviation %.2f%% exceeds max %.2f%%",
					tt.text, got, tt.actualTokens, deviation*100, tt.maxDeviation*100)
			}
		})
	}
}

func TestProviderCounter(t *testing.T) {
	tests := []struct {
		name     string
		provider string
	}{
		{name: "openai", provider: "openai"},
		{name: "azure", provider: "azure"},
		{name: "anthropic", provider: "anthropic"},
		{name: "google", provider: "google"},
		{name: "vertex_ai", provider: "vertex_ai"},
		{name: "aws", provider: "aws"},
		{name: "bedrock", provider: "bedrock"},
		{name: "unknown", provider: "unknown_provider"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			counter := ProviderCounter(tt.provider)

			if counter == nil {
				t.Error("ProviderCounter() returned nil")
				return
			}

			// Verify it's functional
			tokens := counter.CountText("Hello, world!")
			if tokens == 0 {
				t.Error("ProviderCounter() returned non-functional counter")
			}
		})
	}
}

func TestCounterThreadSafety(t *testing.T) {
	counter := NewCounter()
	text := "This is a test message for concurrent counting."

	// Run concurrent operations
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				counter.CountText(text)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// If we get here without panicking, thread safety is working
}

// Benchmarks

func BenchmarkCountText(b *testing.B) {
	counter := NewCounter()
	text := "The quick brown fox jumps over the lazy dog. This is a test sentence."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counter.CountText(text)
	}
}

func BenchmarkCountTextLong(b *testing.B) {
	counter := NewCounter()
	text := `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
	tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
	quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
	Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu
	fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
	culpa qui officia deserunt mollit anim id est laborum.`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counter.CountText(text)
	}
}

func BenchmarkCountMessages(b *testing.B) {
	counter := NewCounter()
	messages := []warp.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello, how are you?"},
		{Role: "assistant", Content: "I'm doing well, thank you for asking!"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counter.CountMessages(messages)
	}
}

func BenchmarkCountMessagesMultimodal(b *testing.B) {
	counter := NewCounter()
	messages := []warp.Message{
		{
			Role: "user",
			Content: []warp.ContentPart{
				{Type: "text", Text: "What is in this image?"},
				{
					Type: "image_url",
					ImageURL: &warp.ImageURL{
						URL:    "https://example.com/image.jpg",
						Detail: "high",
					},
				},
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counter.CountMessages(messages)
	}
}

func BenchmarkCountRequest(b *testing.B) {
	counter := NewCounter()
	req := &warp.CompletionRequest{
		Model: "openai/gpt-4",
		Messages: []warp.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "What's the weather in San Francisco?"},
		},
		Tools: []warp.Tool{
			{
				Type: "function",
				Function: warp.Function{
					Name:        "get_weather",
					Description: "Get the current weather for a location",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "City name",
							},
						},
						"required": []any{"location"},
					},
				},
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counter.CountRequest(req)
	}
}

// Helper function
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
