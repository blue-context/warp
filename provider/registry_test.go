package provider

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/blue-context/warp"
)

func TestNewRegistry(t *testing.T) {
	registry := NewRegistry()

	if registry == nil {
		t.Fatal("NewRegistry() returned nil")
	}

	if registry.Count() != 0 {
		t.Errorf("NewRegistry() Count() = %d, want 0", registry.Count())
	}
}

func TestRegistryRegister(t *testing.T) {
	tests := []struct {
		name      string
		provider  Provider
		wantErr   bool
		errMsg    string
		wantCount int
	}{
		{
			name: "valid provider",
			provider: &MockProvider{
				name:         "openai",
				capabilities: AllSupported(),
			},
			wantErr:   false,
			wantCount: 1,
		},
		{
			name:      "nil provider",
			provider:  nil,
			wantErr:   true,
			errMsg:    "provider cannot be nil",
			wantCount: 0,
		},
		{
			name: "empty name",
			provider: &MockProvider{
				name: "",
			},
			wantErr:   true,
			errMsg:    "provider name cannot be empty",
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := NewRegistry()
			err := registry.Register(tt.provider)

			if (err != nil) != tt.wantErr {
				t.Errorf("Register() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("Register() error = %q, want %q", err.Error(), tt.errMsg)
			}

			if registry.Count() != tt.wantCount {
				t.Errorf("Count() = %d, want %d", registry.Count(), tt.wantCount)
			}
		})
	}
}

func TestRegistryRegisterDuplicate(t *testing.T) {
	registry := NewRegistry()

	provider1 := &MockProvider{name: "openai"}
	provider2 := &MockProvider{name: "openai"}

	err := registry.Register(provider1)
	if err != nil {
		t.Fatalf("Register() first provider error = %v", err)
	}

	err = registry.Register(provider2)
	if err == nil {
		t.Error("Register() duplicate provider error = nil, want error")
	}

	expectedErr := `provider "openai" already registered`
	if err.Error() != expectedErr {
		t.Errorf("Register() error = %q, want %q", err.Error(), expectedErr)
	}

	if registry.Count() != 1 {
		t.Errorf("Count() = %d, want 1", registry.Count())
	}
}

func TestRegistryGet(t *testing.T) {
	registry := NewRegistry()

	provider := &MockProvider{
		name:         "openai",
		capabilities: AllSupported(),
	}

	err := registry.Register(provider)
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	t.Run("existing provider", func(t *testing.T) {
		pInterface, err := registry.Get("openai")
		if err != nil {
			t.Errorf("Get() error = %v", err)
		}
		if pInterface == nil {
			t.Error("Get() returned nil provider")
		}
		p := pInterface.(Provider)
		if p.Name() != "openai" {
			t.Errorf("Get() provider name = %q, want %q", p.Name(), "openai")
		}
	})

	t.Run("non-existing provider", func(t *testing.T) {
		p, err := registry.Get("anthropic")
		if err == nil {
			t.Error("Get() error = nil, want error")
		}
		if p != nil {
			t.Errorf("Get() provider = %v, want nil", p)
		}

		expectedErr := `provider "anthropic" not found`
		if err.Error() != expectedErr {
			t.Errorf("Get() error = %q, want %q", err.Error(), expectedErr)
		}
	})
}

func TestRegistryUnregister(t *testing.T) {
	registry := NewRegistry()

	provider := &MockProvider{name: "openai"}
	err := registry.Register(provider)
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	t.Run("existing provider", func(t *testing.T) {
		err := registry.Unregister("openai")
		if err != nil {
			t.Errorf("Unregister() error = %v", err)
		}

		if registry.Count() != 0 {
			t.Errorf("Count() after Unregister() = %d, want 0", registry.Count())
		}

		if registry.Has("openai") {
			t.Error("Has() after Unregister() = true, want false")
		}
	})

	t.Run("non-existing provider", func(t *testing.T) {
		err := registry.Unregister("anthropic")
		if err == nil {
			t.Error("Unregister() error = nil, want error")
		}

		expectedErr := `provider "anthropic" not found`
		if err.Error() != expectedErr {
			t.Errorf("Unregister() error = %q, want %q", err.Error(), expectedErr)
		}
	})
}

func TestRegistryHas(t *testing.T) {
	registry := NewRegistry()

	provider := &MockProvider{name: "openai"}
	err := registry.Register(provider)
	if err != nil {
		t.Fatalf("Register() error = %v", err)
	}

	tests := []struct {
		name     string
		provider string
		want     bool
	}{
		{
			name:     "existing provider",
			provider: "openai",
			want:     true,
		},
		{
			name:     "non-existing provider",
			provider: "anthropic",
			want:     false,
		},
		{
			name:     "empty name",
			provider: "",
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := registry.Has(tt.provider); got != tt.want {
				t.Errorf("Has(%q) = %v, want %v", tt.provider, got, tt.want)
			}
		})
	}
}

func TestRegistryList(t *testing.T) {
	t.Run("empty registry", func(t *testing.T) {
		registry := NewRegistry()
		list := registry.List()

		if len(list) != 0 {
			t.Errorf("List() len = %d, want 0", len(list))
		}
	})

	t.Run("single provider", func(t *testing.T) {
		registry := NewRegistry()
		registry.Register(&MockProvider{name: "openai"})

		list := registry.List()

		if len(list) != 1 {
			t.Fatalf("List() len = %d, want 1", len(list))
		}
		if list[0] != "openai" {
			t.Errorf("List()[0] = %q, want %q", list[0], "openai")
		}
	})

	t.Run("multiple providers sorted", func(t *testing.T) {
		registry := NewRegistry()
		registry.Register(&MockProvider{name: "openai"})
		registry.Register(&MockProvider{name: "anthropic"})
		registry.Register(&MockProvider{name: "azure"})

		list := registry.List()

		if len(list) != 3 {
			t.Fatalf("List() len = %d, want 3", len(list))
		}

		// Check sorted order
		expected := []string{"anthropic", "azure", "openai"}
		for i, name := range expected {
			if list[i] != name {
				t.Errorf("List()[%d] = %q, want %q", i, list[i], name)
			}
		}
	})

	t.Run("list returns copy", func(t *testing.T) {
		registry := NewRegistry()
		registry.Register(&MockProvider{name: "openai"})

		list1 := registry.List()
		list2 := registry.List()

		// Modify list1
		list1[0] = "modified"

		// list2 should be unchanged
		if list2[0] != "openai" {
			t.Errorf("List() returned shared slice, want independent copies")
		}
	})
}

func TestRegistryCount(t *testing.T) {
	registry := NewRegistry()

	tests := []struct {
		name     string
		action   func()
		want     int
		describe string
	}{
		{
			name:     "initial count",
			action:   func() {},
			want:     0,
			describe: "empty registry",
		},
		{
			name: "after first register",
			action: func() {
				registry.Register(&MockProvider{name: "openai"})
			},
			want:     1,
			describe: "one provider registered",
		},
		{
			name: "after second register",
			action: func() {
				registry.Register(&MockProvider{name: "anthropic"})
			},
			want:     2,
			describe: "two providers registered",
		},
		{
			name: "after unregister",
			action: func() {
				registry.Unregister("openai")
			},
			want:     1,
			describe: "one provider unregistered",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.action()
			if got := registry.Count(); got != tt.want {
				t.Errorf("Count() = %d, want %d (%s)", got, tt.want, tt.describe)
			}
		})
	}
}

func TestRegistryClear(t *testing.T) {
	registry := NewRegistry()

	// Register multiple providers
	registry.Register(&MockProvider{name: "openai"})
	registry.Register(&MockProvider{name: "anthropic"})
	registry.Register(&MockProvider{name: "azure"})

	if registry.Count() != 3 {
		t.Fatalf("Count() before Clear() = %d, want 3", registry.Count())
	}

	registry.Clear()

	if registry.Count() != 0 {
		t.Errorf("Count() after Clear() = %d, want 0", registry.Count())
	}

	if registry.Has("openai") {
		t.Error("Has(openai) after Clear() = true, want false")
	}

	list := registry.List()
	if len(list) != 0 {
		t.Errorf("List() len after Clear() = %d, want 0", len(list))
	}
}

func TestRegistryConcurrentAccess(t *testing.T) {
	registry := NewRegistry()

	// Pre-register some providers
	for i := 0; i < 10; i++ {
		registry.Register(&MockProvider{name: fmt.Sprintf("provider-%d", i)})
	}

	const numGoroutines = 50
	const numOperations = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Run concurrent operations
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				// Mix of read operations
				switch j % 4 {
				case 0:
					// Get
					registry.Get(fmt.Sprintf("provider-%d", j%10))
				case 1:
					// Has
					registry.Has(fmt.Sprintf("provider-%d", j%10))
				case 2:
					// List
					registry.List()
				case 3:
					// Count
					registry.Count()
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify registry is still consistent
	if registry.Count() != 10 {
		t.Errorf("Count() after concurrent reads = %d, want 10", registry.Count())
	}
}

func TestRegistryConcurrentRegister(t *testing.T) {
	registry := NewRegistry()

	const numGoroutines = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Attempt to register same provider concurrently
	successCount := 0
	var mu sync.Mutex

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			err := registry.Register(&MockProvider{name: "concurrent-test"})
			if err == nil {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	// Only one registration should succeed
	if successCount != 1 {
		t.Errorf("successful registrations = %d, want 1", successCount)
	}

	if registry.Count() != 1 {
		t.Errorf("Count() after concurrent register = %d, want 1", registry.Count())
	}
}

func TestRegistryConcurrentRegisterUnregister(t *testing.T) {
	registry := NewRegistry()

	const numGoroutines = 25
	const numOperations = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	// Half goroutines register
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				providerName := fmt.Sprintf("provider-%d", id)
				registry.Register(&MockProvider{name: providerName})
			}
		}(i)
	}

	// Half goroutines unregister
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()

			for j := 0; j < numOperations; j++ {
				providerName := fmt.Sprintf("provider-%d", id)
				registry.Unregister(providerName)
			}
		}(i)
	}

	wg.Wait()

	// Registry should be in a consistent state (no crashes)
	count := registry.Count()
	if count < 0 || count > numGoroutines {
		t.Errorf("Count() after concurrent register/unregister = %d, want 0-%d", count, numGoroutines)
	}
}

func TestRegistryConcurrentClear(t *testing.T) {
	registry := NewRegistry()

	// Register initial providers
	for i := 0; i < 20; i++ {
		registry.Register(&MockProvider{name: fmt.Sprintf("provider-%d", i)})
	}

	const numGoroutines = 10

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Half goroutines clear, half read
	for i := 0; i < numGoroutines/2; i++ {
		go func() {
			defer wg.Done()
			registry.Clear()
		}()
	}

	for i := 0; i < numGoroutines/2; i++ {
		go func() {
			defer wg.Done()
			registry.List()
			registry.Count()
		}()
	}

	wg.Wait()

	// After all clears, registry should be empty
	if registry.Count() != 0 {
		t.Errorf("Count() after Clear() = %d, want 0", registry.Count())
	}
}

func TestRegistryIntegration(t *testing.T) {
	registry := NewRegistry()

	// Create providers with different capabilities
	openai := &MockProvider{
		name:         "openai",
		capabilities: AllSupported(),
	}

	anthropic := &MockProvider{
		name: "anthropic",
		capabilities: Capabilities{
			Completion:      true,
			Streaming:       true,
			FunctionCalling: true,
			Vision:          true,
		},
	}

	azure := &MockProvider{
		name: "azure",
		capabilities: Capabilities{
			Completion: true,
			Streaming:  true,
		},
	}

	// Register all providers
	if err := registry.Register(openai); err != nil {
		t.Fatalf("Register(openai) error = %v", err)
	}
	if err := registry.Register(anthropic); err != nil {
		t.Fatalf("Register(anthropic) error = %v", err)
	}
	if err := registry.Register(azure); err != nil {
		t.Fatalf("Register(azure) error = %v", err)
	}

	// Verify count
	if registry.Count() != 3 {
		t.Errorf("Count() = %d, want 3", registry.Count())
	}

	// Verify list is sorted
	list := registry.List()
	expected := []string{"anthropic", "azure", "openai"}
	for i, name := range expected {
		if list[i] != name {
			t.Errorf("List()[%d] = %q, want %q", i, list[i], name)
		}
	}

	// Get and use a provider
	pInterface, err := registry.Get("openai")
	if err != nil {
		t.Fatalf("Get(openai) error = %v", err)
	}
	p := pInterface.(Provider)

	resp, err := p.Completion(context.Background(), &warp.CompletionRequest{
		Model: "gpt-4",
		Messages: []warp.Message{
			{Role: "user", Content: "test"},
		},
	})
	if err != nil {
		t.Fatalf("Completion() error = %v", err)
	}
	if resp == nil {
		t.Error("Completion() returned nil response")
	}

	// Check capabilities
	capsInterface := p.Supports()
	caps, ok := capsInterface.(Capabilities)
	if !ok {
		t.Fatalf("Supports() returned unexpected type: %T", capsInterface)
	}
	if !caps.Completion {
		t.Error("Provider doesn't support Completion")
	}

	// Unregister a provider
	if err := registry.Unregister("azure"); err != nil {
		t.Fatalf("Unregister(azure) error = %v", err)
	}

	if registry.Count() != 2 {
		t.Errorf("Count() after Unregister() = %d, want 2", registry.Count())
	}

	// Clear all
	registry.Clear()

	if registry.Count() != 0 {
		t.Errorf("Count() after Clear() = %d, want 0", registry.Count())
	}
}
