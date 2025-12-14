package warp

import (
	"errors"
	"fmt"
	"testing"
	"time"
)

func TestErrorHierarchy(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		wantType string
		isWarp   bool
	}{
		{
			name:     "APIError is WarpError",
			err:      NewAPIError("test error", 500, "openai", nil),
			wantType: "*warp.APIError",
			isWarp:   true,
		},
		{
			name:     "AuthenticationError is WarpError",
			err:      NewAuthenticationError("invalid key", "openai", nil),
			wantType: "*warp.AuthenticationError",
			isWarp:   true,
		},
		{
			name:     "RateLimitError is WarpError",
			err:      NewRateLimitError("too many requests", "openai", 5*time.Second, nil),
			wantType: "*warp.RateLimitError",
			isWarp:   true,
		},
		{
			name:     "TimeoutError is WarpError",
			err:      NewTimeoutError("request timed out", "openai", nil),
			wantType: "*warp.TimeoutError",
			isWarp:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify error has the embedded WarpError
			// Since the error types embed WarpError, we check by attempting type assertion
			// to each concrete type and verifying the embedded field exists
			switch e := tt.err.(type) {
			case *APIError:
				if e.WarpError.Message == "" {
					t.Error("embedded WarpError not accessible")
				}
			case *AuthenticationError:
				if e.WarpError.Message == "" {
					t.Error("embedded WarpError not accessible")
				}
			case *RateLimitError:
				if e.WarpError.Message == "" {
					t.Error("embedded WarpError not accessible")
				}
			case *TimeoutError:
				if e.WarpError.Message == "" {
					t.Error("embedded WarpError not accessible")
				}
			}

			// Verify Error() returns a message
			if tt.err.Error() == "" {
				t.Error("Error() returned empty string")
			}
		})
	}
}

func TestErrorUnwrap(t *testing.T) {
	tests := []struct {
		name        string
		createError func(error) error
		wantUnwrap  bool
	}{
		{
			name: "APIError unwraps original error",
			createError: func(orig error) error {
				return NewAPIError("test", 500, "openai", orig)
			},
			wantUnwrap: true,
		},
		{
			name: "AuthenticationError unwraps original error",
			createError: func(orig error) error {
				return NewAuthenticationError("test", "openai", orig)
			},
			wantUnwrap: true,
		},
		{
			name: "RateLimitError unwraps original error",
			createError: func(orig error) error {
				return NewRateLimitError("test", "openai", 0, orig)
			},
			wantUnwrap: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalErr := errors.New("original error")
			wrappedErr := tt.createError(originalErr)

			// Use errors.Is to check unwrapping
			if tt.wantUnwrap {
				if !errors.Is(wrappedErr, originalErr) {
					t.Error("error does not unwrap to original error")
				}
			}

			// Test Unwrap method directly
			if unwrapper, ok := wrappedErr.(interface{ Unwrap() error }); ok {
				unwrapped := unwrapper.Unwrap()
				if tt.wantUnwrap && unwrapped != originalErr {
					t.Errorf("Unwrap() = %v, want %v", unwrapped, originalErr)
				}
			}
		})
	}
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name          string
		err           error
		wantRetryable bool
	}{
		{
			name:          "RateLimitError is retryable",
			err:           NewRateLimitError("rate limit", "openai", 5*time.Second, nil),
			wantRetryable: true,
		},
		{
			name:          "TimeoutError is retryable",
			err:           NewTimeoutError("timeout", "openai", nil),
			wantRetryable: true,
		},
		{
			name:          "ServiceUnavailableError is retryable",
			err:           NewServiceUnavailableError("service down", "openai", nil),
			wantRetryable: true,
		},
		{
			name:          "AuthenticationError is not retryable",
			err:           NewAuthenticationError("invalid key", "openai", nil),
			wantRetryable: false,
		},
		{
			name:          "PermissionError is not retryable",
			err:           NewPermissionError("forbidden", "openai", nil),
			wantRetryable: false,
		},
		{
			name:          "InvalidRequestError is not retryable",
			err:           NewInvalidRequestError("bad request", "openai", nil),
			wantRetryable: false,
		},
		{
			name:          "ContextWindowExceededError is not retryable",
			err:           NewContextWindowExceededError("too many tokens", "openai", 4096, 5000, nil),
			wantRetryable: false,
		},
		{
			name:          "ContentPolicyViolationError is not retryable",
			err:           NewContentPolicyViolationError("policy violation", "openai", nil),
			wantRetryable: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Check if error implements IsRetryable
			if retryable, ok := tt.err.(interface{ IsRetryable() bool }); ok {
				got := retryable.IsRetryable()
				if got != tt.wantRetryable {
					t.Errorf("IsRetryable() = %v, want %v", got, tt.wantRetryable)
				}
			} else {
				t.Error("error does not implement IsRetryable()")
			}
		})
	}
}

func TestParseProviderError(t *testing.T) {
	tests := []struct {
		name       string
		provider   string
		statusCode int
		body       []byte
		wantType   string
	}{
		{
			name:       "401 returns AuthenticationError",
			provider:   "openai",
			statusCode: 401,
			body:       []byte(`{"error":{"message":"Invalid API key"}}`),
			wantType:   "*warp.AuthenticationError",
		},
		{
			name:       "403 returns PermissionError",
			provider:   "openai",
			statusCode: 403,
			body:       []byte(`{"error":{"message":"Forbidden"}}`),
			wantType:   "*warp.PermissionError",
		},
		{
			name:       "429 returns RateLimitError",
			provider:   "openai",
			statusCode: 429,
			body:       []byte(`{"error":{"message":"Rate limit exceeded"}}`),
			wantType:   "*warp.RateLimitError",
		},
		{
			name:       "503 returns ServiceUnavailableError",
			provider:   "openai",
			statusCode: 503,
			body:       []byte(`{"error":{"message":"Service unavailable"}}`),
			wantType:   "*warp.ServiceUnavailableError",
		},
		{
			name:       "500 returns ServiceUnavailableError",
			provider:   "openai",
			statusCode: 500,
			body:       []byte(`{"error":{"message":"Internal server error"}}`),
			wantType:   "*warp.ServiceUnavailableError",
		},
		{
			name:       "400 with context message returns ContextWindowExceededError",
			provider:   "openai",
			statusCode: 400,
			body:       []byte(`{"error":{"message":"Maximum context length exceeded"}}`),
			wantType:   "*warp.ContextWindowExceededError",
		},
		{
			name:       "400 with token limit message returns ContextWindowExceededError",
			provider:   "openai",
			statusCode: 400,
			body:       []byte(`{"error":{"message":"Too many tokens in request"}}`),
			wantType:   "*warp.ContextWindowExceededError",
		},
		{
			name:       "400 with content policy returns ContentPolicyViolationError",
			provider:   "openai",
			statusCode: 400,
			body:       []byte(`{"error":{"message":"Content policy violation detected"}}`),
			wantType:   "*warp.ContentPolicyViolationError",
		},
		{
			name:       "400 with safety filter returns ContentPolicyViolationError",
			provider:   "anthropic",
			statusCode: 400,
			body:       []byte(`{"error":{"message":"Content blocked by safety filter"}}`),
			wantType:   "*warp.ContentPolicyViolationError",
		},
		{
			name:       "400 with invalid message returns InvalidRequestError",
			provider:   "openai",
			statusCode: 400,
			body:       []byte(`{"error":{"message":"Invalid request: missing field"}}`),
			wantType:   "*warp.InvalidRequestError",
		},
		{
			name:       "400 generic returns BadRequestError",
			provider:   "openai",
			statusCode: 400,
			body:       []byte(`{"error":{"message":"Bad request"}}`),
			wantType:   "*warp.BadRequestError",
		},
		{
			name:       "unknown status code returns APIError",
			provider:   "openai",
			statusCode: 418,
			body:       []byte(`{"error":{"message":"I'm a teapot"}}`),
			wantType:   "*warp.APIError",
		},
		{
			name:       "non-JSON body returns ServiceUnavailableError for 500",
			provider:   "openai",
			statusCode: 500,
			body:       []byte("Internal Server Error"),
			wantType:   "*warp.ServiceUnavailableError",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ParseProviderError(tt.provider, tt.statusCode, tt.body, nil)

			if err == nil {
				t.Fatal("ParseProviderError() returned nil")
			}

			// Check error type using type assertion
			errType := fmt.Sprintf("%T", err)
			if errType != tt.wantType {
				t.Errorf("ParseProviderError() type = %v, want %v", errType, tt.wantType)
			}

			// Verify error message is not empty
			if err.Error() == "" {
				t.Error("error message is empty")
			}

			// Verify provider is set
			var litellmErr *WarpError
			if errors.As(err, &litellmErr) {
				if litellmErr.Provider != tt.provider {
					t.Errorf("Provider = %v, want %v", litellmErr.Provider, tt.provider)
				}

				if litellmErr.StatusCode != tt.statusCode {
					t.Errorf("StatusCode = %v, want %v", litellmErr.StatusCode, tt.statusCode)
				}
			}
		})
	}
}

func TestErrorConstructors(t *testing.T) {
	t.Run("NewAPIError", func(t *testing.T) {
		err := NewAPIError("test message", 500, "openai", nil)
		if err == nil {
			t.Fatal("NewAPIError() returned nil")
		}

		if err.Message != "test message" {
			t.Errorf("Message = %v, want %v", err.Message, "test message")
		}

		if err.StatusCode != 500 {
			t.Errorf("StatusCode = %v, want %v", err.StatusCode, 500)
		}

		if err.Provider != "openai" {
			t.Errorf("Provider = %v, want %v", err.Provider, "openai")
		}
	})

	t.Run("NewAuthenticationError", func(t *testing.T) {
		err := NewAuthenticationError("invalid key", "anthropic", nil)
		if err == nil {
			t.Fatal("NewAuthenticationError() returned nil")
		}

		if err.StatusCode != 401 {
			t.Errorf("StatusCode = %v, want 401", err.StatusCode)
		}
	})

	t.Run("NewRateLimitError", func(t *testing.T) {
		retryAfter := 10 * time.Second
		err := NewRateLimitError("rate limited", "openai", retryAfter, nil)
		if err == nil {
			t.Fatal("NewRateLimitError() returned nil")
		}

		if err.StatusCode != 429 {
			t.Errorf("StatusCode = %v, want 429", err.StatusCode)
		}

		if err.RetryAfter != retryAfter {
			t.Errorf("RetryAfter = %v, want %v", err.RetryAfter, retryAfter)
		}

		if !err.IsRetryable() {
			t.Error("RateLimitError should be retryable")
		}
	})

	t.Run("NewContextWindowExceededError", func(t *testing.T) {
		err := NewContextWindowExceededError("too long", "openai", 4096, 5000, nil)
		if err == nil {
			t.Fatal("NewContextWindowExceededError() returned nil")
		}

		if err.MaxTokens != 4096 {
			t.Errorf("MaxTokens = %v, want 4096", err.MaxTokens)
		}

		if err.Tokens != 5000 {
			t.Errorf("Tokens = %v, want 5000", err.Tokens)
		}
	})
}

func TestErrorMessage(t *testing.T) {
	tests := []struct {
		name         string
		err          error
		wantContains string
	}{
		{
			name:         "error with provider includes provider in message",
			err:          NewAPIError("test error", 500, "openai", nil),
			wantContains: "[openai]",
		},
		{
			name:         "error message includes original message",
			err:          NewAPIError("specific error", 500, "anthropic", nil),
			wantContains: "specific error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := tt.err.Error()
			if msg == "" {
				t.Error("Error() returned empty string")
			}

			if tt.wantContains != "" && !contains(msg, tt.wantContains) {
				t.Errorf("Error() = %q, want to contain %q", msg, tt.wantContains)
			}
		})
	}
}

func TestRateLimitErrorRetryable(t *testing.T) {
	err := NewRateLimitError("rate limit", "openai", 5*time.Second, nil)

	if !err.IsRetryable() {
		t.Error("RateLimitError should be retryable")
	}

	// Verify it's also retryable through the interface
	var retryable interface{ IsRetryable() bool }
	if !errors.As(err, &retryable) {
		t.Fatal("error does not implement IsRetryable")
	}

	if !retryable.IsRetryable() {
		t.Error("RateLimitError IsRetryable() = false, want true")
	}
}

func TestTimeoutErrorRetryable(t *testing.T) {
	err := NewTimeoutError("timeout", "openai", nil)

	if !err.IsRetryable() {
		t.Error("TimeoutError should be retryable")
	}
}

func TestServiceUnavailableErrorRetryable(t *testing.T) {
	err := NewServiceUnavailableError("service down", "openai", nil)

	if !err.IsRetryable() {
		t.Error("ServiceUnavailableError should be retryable")
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
