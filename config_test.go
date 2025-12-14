package warp

import (
	"net/http"
	"os"
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	config := defaultConfig()

	if config.APIKeys == nil {
		t.Error("APIKeys map should be initialized")
	}
	if config.APIBases == nil {
		t.Error("APIBases map should be initialized")
	}
	if config.DefaultTimeout != 60*time.Second {
		t.Errorf("DefaultTimeout = %v, want %v", config.DefaultTimeout, 60*time.Second)
	}
	if config.MaxRetries != 3 {
		t.Errorf("MaxRetries = %d, want 3", config.MaxRetries)
	}
	if config.RetryDelay != 1*time.Second {
		t.Errorf("RetryDelay = %v, want %v", config.RetryDelay, 1*time.Second)
	}
	if config.RetryMultiplier != 2.0 {
		t.Errorf("RetryMultiplier = %f, want 2.0", config.RetryMultiplier)
	}
	if config.Debug != false {
		t.Errorf("Debug = %v, want false", config.Debug)
	}
	if config.TrackCost != false {
		t.Errorf("TrackCost = %v, want false", config.TrackCost)
	}
	if config.MaxBudget != 0 {
		t.Errorf("MaxBudget = %f, want 0", config.MaxBudget)
	}
	if config.HTTPClient == nil {
		t.Error("HTTPClient should be initialized")
	}
}

func TestWithAPIKey(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		key      string
		wantErr  bool
	}{
		{
			name:     "valid API key",
			provider: "openai",
			key:      "sk-test123",
			wantErr:  false,
		},
		{
			name:     "empty provider",
			provider: "",
			key:      "sk-test123",
			wantErr:  true,
		},
		{
			name:     "empty key",
			provider: "openai",
			key:      "",
			wantErr:  true,
		},
		{
			name:     "both empty",
			provider: "",
			key:      "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithAPIKey(tt.provider, tt.key)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithAPIKey() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if got := config.APIKeys[tt.provider]; got != tt.key {
					t.Errorf("APIKeys[%s] = %s, want %s", tt.provider, got, tt.key)
				}
			}
		})
	}
}

func TestWithAPIBase(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		base     string
		wantErr  bool
	}{
		{
			name:     "valid API base",
			provider: "openai",
			base:     "https://api.custom.com/v1",
			wantErr:  false,
		},
		{
			name:     "empty provider",
			provider: "",
			base:     "https://api.custom.com/v1",
			wantErr:  true,
		},
		{
			name:     "empty base",
			provider: "openai",
			base:     "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithAPIBase(tt.provider, tt.base)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithAPIBase() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if got := config.APIBases[tt.provider]; got != tt.base {
					t.Errorf("APIBases[%s] = %s, want %s", tt.provider, got, tt.base)
				}
			}
		})
	}
}

func TestWithTimeout(t *testing.T) {
	tests := []struct {
		name    string
		timeout time.Duration
		wantErr bool
	}{
		{
			name:    "valid timeout",
			timeout: 30 * time.Second,
			wantErr: false,
		},
		{
			name:    "zero timeout",
			timeout: 0,
			wantErr: true,
		},
		{
			name:    "negative timeout",
			timeout: -1 * time.Second,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithTimeout(tt.timeout)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithTimeout() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if config.DefaultTimeout != tt.timeout {
					t.Errorf("DefaultTimeout = %v, want %v", config.DefaultTimeout, tt.timeout)
				}
			}
		})
	}
}

func TestWithRetries(t *testing.T) {
	tests := []struct {
		name         string
		maxRetries   int
		initialDelay time.Duration
		multiplier   float64
		wantErr      bool
	}{
		{
			name:         "valid retry config",
			maxRetries:   5,
			initialDelay: 2 * time.Second,
			multiplier:   2.5,
			wantErr:      false,
		},
		{
			name:         "zero retries",
			maxRetries:   0,
			initialDelay: 1 * time.Second,
			multiplier:   2.0,
			wantErr:      false,
		},
		{
			name:         "negative retries",
			maxRetries:   -1,
			initialDelay: 1 * time.Second,
			multiplier:   2.0,
			wantErr:      true,
		},
		{
			name:         "negative delay",
			maxRetries:   3,
			initialDelay: -1 * time.Second,
			multiplier:   2.0,
			wantErr:      true,
		},
		{
			name:         "zero multiplier",
			maxRetries:   3,
			initialDelay: 1 * time.Second,
			multiplier:   0,
			wantErr:      true,
		},
		{
			name:         "negative multiplier",
			maxRetries:   3,
			initialDelay: 1 * time.Second,
			multiplier:   -1.0,
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithRetries(tt.maxRetries, tt.initialDelay, tt.multiplier)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithRetries() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if config.MaxRetries != tt.maxRetries {
					t.Errorf("MaxRetries = %d, want %d", config.MaxRetries, tt.maxRetries)
				}
				if config.RetryDelay != tt.initialDelay {
					t.Errorf("RetryDelay = %v, want %v", config.RetryDelay, tt.initialDelay)
				}
				if config.RetryMultiplier != tt.multiplier {
					t.Errorf("RetryMultiplier = %f, want %f", config.RetryMultiplier, tt.multiplier)
				}
			}
		})
	}
}

func TestWithMaxRetries(t *testing.T) {
	tests := []struct {
		name    string
		max     int
		wantErr bool
	}{
		{
			name:    "valid max retries",
			max:     5,
			wantErr: false,
		},
		{
			name:    "zero retries",
			max:     0,
			wantErr: false,
		},
		{
			name:    "negative retries",
			max:     -1,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithMaxRetries(tt.max)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithMaxRetries() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if config.MaxRetries != tt.max {
					t.Errorf("MaxRetries = %d, want %d", config.MaxRetries, tt.max)
				}
			}
		})
	}
}

func TestWithFallbacks(t *testing.T) {
	tests := []struct {
		name    string
		models  []string
		wantErr bool
	}{
		{
			name:    "single fallback",
			models:  []string{"gpt-3.5-turbo"},
			wantErr: false,
		},
		{
			name:    "multiple fallbacks",
			models:  []string{"gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"},
			wantErr: false,
		},
		{
			name:    "no fallbacks",
			models:  []string{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithFallbacks(tt.models...)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithFallbacks() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if len(config.FallbackModels) != len(tt.models) {
					t.Errorf("FallbackModels length = %d, want %d", len(config.FallbackModels), len(tt.models))
				}
				for i, model := range tt.models {
					if config.FallbackModels[i] != model {
						t.Errorf("FallbackModels[%d] = %s, want %s", i, config.FallbackModels[i], model)
					}
				}
			}
		})
	}
}

func TestWithDebug(t *testing.T) {
	tests := []struct {
		name  string
		debug bool
	}{
		{
			name:  "enable debug",
			debug: true,
		},
		{
			name:  "disable debug",
			debug: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithDebug(tt.debug)
			err := opt(config)

			if err != nil {
				t.Errorf("WithDebug() unexpected error: %v", err)
				return
			}

			if config.Debug != tt.debug {
				t.Errorf("Debug = %v, want %v", config.Debug, tt.debug)
			}
		})
	}
}

func TestWithCostTracking(t *testing.T) {
	tests := []struct {
		name  string
		track bool
	}{
		{
			name:  "enable cost tracking",
			track: true,
		},
		{
			name:  "disable cost tracking",
			track: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithCostTracking(tt.track)
			err := opt(config)

			if err != nil {
				t.Errorf("WithCostTracking() unexpected error: %v", err)
				return
			}

			if config.TrackCost != tt.track {
				t.Errorf("TrackCost = %v, want %v", config.TrackCost, tt.track)
			}
		})
	}
}

func TestWithMaxBudget(t *testing.T) {
	tests := []struct {
		name    string
		budget  float64
		wantErr bool
	}{
		{
			name:    "valid budget",
			budget:  10.0,
			wantErr: false,
		},
		{
			name:    "zero budget",
			budget:  0,
			wantErr: false,
		},
		{
			name:    "negative budget",
			budget:  -1.0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithMaxBudget(tt.budget)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithMaxBudget() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if config.MaxBudget != tt.budget {
					t.Errorf("MaxBudget = %f, want %f", config.MaxBudget, tt.budget)
				}
			}
		})
	}
}

func TestWithHTTPClient(t *testing.T) {
	tests := []struct {
		name    string
		client  HTTPClient
		wantErr bool
	}{
		{
			name:    "valid HTTP client",
			client:  &http.Client{Timeout: 30 * time.Second},
			wantErr: false,
		},
		{
			name:    "nil HTTP client",
			client:  nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			opt := WithHTTPClient(tt.client)
			err := opt(config)

			if (err != nil) != tt.wantErr {
				t.Errorf("WithHTTPClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if config.HTTPClient != tt.client {
					t.Errorf("HTTPClient not set correctly")
				}
			}
		})
	}
}

func TestLoadConfigFromEnv(t *testing.T) {
	// Save original env vars
	originalEnv := make(map[string]string)
	envVars := []string{
		"OPENAI_API_KEY",
		"OPENAI_API_BASE",
		"ANTHROPIC_API_KEY",
		"AWS_ACCESS_KEY_ID",
		"AWS_SECRET_ACCESS_KEY",
		"AWS_REGION_NAME",
		"VERTEX_PROJECT",
		"VERTEX_LOCATION",
	}

	for _, key := range envVars {
		originalEnv[key] = os.Getenv(key)
	}

	// Restore env vars after test
	defer func() {
		for key, val := range originalEnv {
			if val == "" {
				os.Unsetenv(key)
			} else {
				os.Setenv(key, val)
			}
		}
	}()

	tests := []struct {
		name     string
		envSetup map[string]string
		wantKeys map[string]string
	}{
		{
			name: "OpenAI key and base",
			envSetup: map[string]string{
				"OPENAI_API_KEY":  "sk-test123",
				"OPENAI_API_BASE": "https://api.custom.com/v1",
			},
			wantKeys: map[string]string{
				"openai": "sk-test123",
			},
		},
		{
			name: "multiple providers",
			envSetup: map[string]string{
				"OPENAI_API_KEY":    "sk-openai",
				"ANTHROPIC_API_KEY": "sk-anthropic",
			},
			wantKeys: map[string]string{
				"openai":    "sk-openai",
				"anthropic": "sk-anthropic",
			},
		},
		{
			name: "AWS credentials",
			envSetup: map[string]string{
				"AWS_ACCESS_KEY_ID":     "AKIATEST",
				"AWS_SECRET_ACCESS_KEY": "secret123",
				"AWS_REGION_NAME":       "us-west-2",
			},
			wantKeys: map[string]string{
				"aws_access_key_id":     "AKIATEST",
				"aws_secret_access_key": "secret123",
				"aws_region_name":       "us-west-2",
			},
		},
		{
			name: "Vertex AI",
			envSetup: map[string]string{
				"VERTEX_PROJECT":  "my-project",
				"VERTEX_LOCATION": "us-central1",
			},
			wantKeys: map[string]string{
				"vertex_project":  "my-project",
				"vertex_location": "us-central1",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear all env vars first
			for _, key := range envVars {
				os.Unsetenv(key)
			}

			// Set test env vars
			for key, val := range tt.envSetup {
				os.Setenv(key, val)
			}

			opts, err := LoadConfigFromEnv()
			if err != nil {
				t.Errorf("LoadConfigFromEnv() unexpected error: %v", err)
				return
			}

			// Apply options to config
			config := defaultConfig()
			for _, opt := range opts {
				if err := opt(config); err != nil {
					t.Errorf("Failed to apply option: %v", err)
				}
			}

			// Verify expected keys are set
			for key, expectedVal := range tt.wantKeys {
				if got := config.APIKeys[key]; got != expectedVal {
					t.Errorf("APIKeys[%s] = %s, want %s", key, got, expectedVal)
				}
			}
		})
	}
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(*ClientConfig)
		wantErr bool
	}{
		{
			name:    "valid config",
			modify:  func(c *ClientConfig) {},
			wantErr: false,
		},
		{
			name: "invalid timeout",
			modify: func(c *ClientConfig) {
				c.DefaultTimeout = 0
			},
			wantErr: true,
		},
		{
			name: "negative retries",
			modify: func(c *ClientConfig) {
				c.MaxRetries = -1
			},
			wantErr: true,
		},
		{
			name: "negative retry delay",
			modify: func(c *ClientConfig) {
				c.RetryDelay = -1 * time.Second
			},
			wantErr: true,
		},
		{
			name: "invalid retry multiplier",
			modify: func(c *ClientConfig) {
				c.RetryMultiplier = 0
			},
			wantErr: true,
		},
		{
			name: "negative budget",
			modify: func(c *ClientConfig) {
				c.MaxBudget = -1.0
			},
			wantErr: true,
		},
		{
			name: "nil HTTP client",
			modify: func(c *ClientConfig) {
				c.HTTPClient = nil
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := defaultConfig()
			tt.modify(config)
			err := config.Validate()

			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestMultipleOptions(t *testing.T) {
	config := defaultConfig()

	opts := []ClientOption{
		WithAPIKey("openai", "sk-test123"),
		WithAPIKey("anthropic", "sk-ant123"),
		WithTimeout(30 * time.Second),
		WithMaxRetries(5),
		WithDebug(true),
		WithCostTracking(true),
		WithMaxBudget(100.0),
	}

	for _, opt := range opts {
		if err := opt(config); err != nil {
			t.Errorf("Failed to apply option: %v", err)
		}
	}

	// Verify all options were applied
	if config.APIKeys["openai"] != "sk-test123" {
		t.Errorf("OpenAI key not set")
	}
	if config.APIKeys["anthropic"] != "sk-ant123" {
		t.Errorf("Anthropic key not set")
	}
	if config.DefaultTimeout != 30*time.Second {
		t.Errorf("Timeout not set")
	}
	if config.MaxRetries != 5 {
		t.Errorf("MaxRetries not set")
	}
	if !config.Debug {
		t.Errorf("Debug not enabled")
	}
	if !config.TrackCost {
		t.Errorf("CostTracking not enabled")
	}
	if config.MaxBudget != 100.0 {
		t.Errorf("MaxBudget not set")
	}

	// Verify config is still valid
	if err := config.Validate(); err != nil {
		t.Errorf("Config validation failed: %v", err)
	}
}
