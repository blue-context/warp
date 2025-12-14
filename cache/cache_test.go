package cache

import (
	"testing"
)

func TestKey(t *testing.T) {
	tests := []struct {
		name        string
		model       string
		messages    []byte
		temperature any
		maxTokens   any
		topP        any
		want        string
	}{
		{
			name:     "basic key",
			model:    "gpt-4",
			messages: []byte(`[{"role":"user","content":"hello"}]`),
			want:     "warp:v1:",
		},
		{
			name:        "with temperature",
			model:       "gpt-4",
			messages:    []byte(`[{"role":"user","content":"hello"}]`),
			temperature: float64(0.7),
			want:        "warp:v1:",
		},
		{
			name:        "with temperature pointer",
			model:       "gpt-4",
			messages:    []byte(`[{"role":"user","content":"hello"}]`),
			temperature: func() *float64 { v := 0.7; return &v }(),
			want:        "warp:v1:",
		},
		{
			name:      "with maxTokens",
			model:     "gpt-4",
			messages:  []byte(`[{"role":"user","content":"hello"}]`),
			maxTokens: 100,
			want:      "warp:v1:",
		},
		{
			name:      "with maxTokens pointer",
			model:     "gpt-4",
			messages:  []byte(`[{"role":"user","content":"hello"}]`),
			maxTokens: func() *int { v := 100; return &v }(),
			want:      "warp:v1:",
		},
		{
			name:     "with topP",
			model:    "gpt-4",
			messages: []byte(`[{"role":"user","content":"hello"}]`),
			topP:     float64(0.9),
			want:     "warp:v1:",
		},
		{
			name:     "with topP pointer",
			model:    "gpt-4",
			messages: []byte(`[{"role":"user","content":"hello"}]`),
			topP:     func() *float64 { v := 0.9; return &v }(),
			want:     "warp:v1:",
		},
		{
			name:        "all parameters",
			model:       "gpt-4",
			messages:    []byte(`[{"role":"user","content":"hello"}]`),
			temperature: float64(0.7),
			maxTokens:   100,
			topP:        float64(0.9),
			want:        "warp:v1:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Key(tt.model, tt.messages, tt.temperature, tt.maxTokens, tt.topP)

			// Check prefix
			if len(got) < len(tt.want) || got[:len(tt.want)] != tt.want {
				t.Errorf("Key() = %v, want prefix %v", got, tt.want)
			}

			// Check it's a valid SHA256 hash (64 hex chars after prefix)
			if len(got) != len("warp:v1:")+64 {
				t.Errorf("Key() length = %d, want %d", len(got), len("warp:v1:")+64)
			}
		})
	}
}

func TestKeyDeterministic(t *testing.T) {
	model := "gpt-4"
	messages := []byte(`[{"role":"user","content":"hello"}]`)
	temp := float64(0.7)
	maxTokens := 100
	topP := float64(0.9)

	key1 := Key(model, messages, temp, maxTokens, topP)
	key2 := Key(model, messages, temp, maxTokens, topP)

	if key1 != key2 {
		t.Errorf("Key() is not deterministic: %s != %s", key1, key2)
	}
}

func TestKeyDifferentInputs(t *testing.T) {
	messages := []byte(`[{"role":"user","content":"hello"}]`)

	key1 := Key("gpt-4", messages, nil, nil, nil)
	key2 := Key("gpt-3.5-turbo", messages, nil, nil, nil)

	if key1 == key2 {
		t.Error("Different models produced same cache key")
	}

	key3 := Key("gpt-4", []byte(`[{"role":"user","content":"goodbye"}]`), nil, nil, nil)
	if key1 == key3 {
		t.Error("Different messages produced same cache key")
	}

	key4 := Key("gpt-4", messages, float64(0.7), nil, nil)
	if key1 == key4 {
		t.Error("Different temperatures produced same cache key")
	}
}

func TestKeyNilParameters(t *testing.T) {
	messages := []byte(`[{"role":"user","content":"hello"}]`)

	// Should handle nil pointers without panicking
	var tempPtr *float64 = nil
	var maxTokensPtr *int = nil
	var topPPtr *float64 = nil

	key := Key("gpt-4", messages, tempPtr, maxTokensPtr, topPPtr)
	if key == "" {
		t.Error("Key() returned empty string")
	}
}
