package provider

import (
	"fmt"
	"sort"
	"sync"
)

// Registry manages provider implementations.
//
// Registry is thread-safe and can be used concurrently from multiple goroutines.
// It uses a read-write mutex to protect internal state while allowing concurrent reads.
//
// Example:
//
//	registry := provider.NewRegistry()
//	registry.Register(openaiProvider)
//	registry.Register(anthropicProvider)
//
//	p, err := registry.Get("openai")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Use provider...
//	resp, err := p.Completion(ctx, req)
type Registry struct {
	providers map[string]Provider
	mu        sync.RWMutex
}

// NewRegistry creates a new provider registry.
//
// The registry starts empty. Use Register to add providers.
//
// Example:
//
//	registry := provider.NewRegistry()
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
	}
}

// Register registers a provider.
//
// Returns an error if:
//   - The provider is nil
//   - The provider name is empty
//   - A provider with the same name is already registered
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	err := registry.Register(openaiProvider)
//	if err != nil {
//	    log.Fatal(err)
//	}
func (r *Registry) Register(p Provider) error {
	if p == nil {
		return fmt.Errorf("provider cannot be nil")
	}

	name := p.Name()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.providers[name]; exists {
		return fmt.Errorf("provider %q already registered", name)
	}

	r.providers[name] = p
	return nil
}

// Unregister removes a provider from the registry.
//
// Returns an error if the provider is not found.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	err := registry.Unregister("openai")
//	if err != nil {
//	    log.Printf("Failed to unregister: %v", err)
//	}
func (r *Registry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.providers[name]; !exists {
		return fmt.Errorf("provider %q not found", name)
	}

	delete(r.providers, name)
	return nil
}

// Get retrieves a provider by name.
//
// Returns an error if the provider is not found.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	provider, err := registry.Get("openai")
//	if err != nil {
//	    return err
//	}
//	resp, err := provider.Completion(ctx, req)
func (r *Registry) Get(name string) (interface{}, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	p, exists := r.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider %q not found", name)
	}

	return p, nil
}

// GetProvider retrieves a provider by name with concrete type.
//
// This is a type-safe wrapper around Get().
//
// Thread Safety: Safe for concurrent use.
func (r *Registry) GetProvider(name string) (Provider, error) {
	pInterface, err := r.Get(name)
	if err != nil {
		return nil, err
	}
	return pInterface.(Provider), nil
}

// Has checks if a provider is registered.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	if registry.Has("openai") {
//	    provider, _ := registry.Get("openai")
//	    // Use provider...
//	}
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	_, exists := r.providers[name]
	return exists
}

// List returns all registered provider names.
//
// The returned slice is sorted alphabetically.
// Returns a new slice on each call - modifications will not affect the registry.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	providers := registry.List()
//	fmt.Println("Available providers:", providers)
//	// Output: Available providers: [anthropic azure openai]
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}

	sort.Strings(names)
	return names
}

// Count returns the number of registered providers.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	count := registry.Count()
//	fmt.Printf("Registry has %d providers\n", count)
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return len(r.providers)
}

// Clear removes all providers from the registry.
//
// After calling Clear, the registry will be empty.
//
// Thread Safety: Safe for concurrent use.
//
// Example:
//
//	registry.Clear()
//	fmt.Println("Providers cleared")
func (r *Registry) Clear() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.providers = make(map[string]Provider)
}
