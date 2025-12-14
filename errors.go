package warp

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// WarpError is the base error type for all Warp SDK errors.
// It provides context about the error including provider, model, and status code.
// All specific error types embed this base type.
type WarpError struct {
	// Message is the human-readable error message.
	Message string

	// StatusCode is the HTTP status code (if applicable).
	StatusCode int

	// Provider is the LLM provider where the error occurred.
	// Examples: "openai", "anthropic", "azure"
	Provider string

	// Model is the model that was being used when the error occurred.
	Model string

	// LLMProvider is the underlying LLM provider (may differ from Provider for proxies).
	LLMProvider string

	// OriginalError is the underlying error that caused this error.
	OriginalError error
}

// Error implements the error interface.
// Returns a formatted error message with provider context when available.
func (e *WarpError) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("[%s] %s", e.Provider, e.Message)
	}
	return e.Message
}

// Unwrap returns the original underlying error.
// This enables error chain traversal using errors.Is() and errors.As().
func (e *WarpError) Unwrap() error {
	return e.OriginalError
}

// IsRetryable returns true if this error represents a retryable condition.
// Base implementation returns false; specific error types override this.
func (e *WarpError) IsRetryable() bool {
	return false
}

// APIError represents a general API error from the LLM provider.
// This is used for errors that don't fit into more specific categories.
type APIError struct {
	WarpError
}

// NewAPIError creates a new API error with the given details.
func NewAPIError(message string, statusCode int, provider string, err error) *APIError {
	return &APIError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    statusCode,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// AuthenticationError represents an authentication failure (401).
// This occurs when API keys are invalid, missing, or expired.
type AuthenticationError struct {
	WarpError
}

// NewAuthenticationError creates a new authentication error.
func NewAuthenticationError(message string, provider string, err error) *AuthenticationError {
	return &AuthenticationError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    401,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// PermissionError represents a permission denied error (403).
// This occurs when the API key is valid but lacks necessary permissions.
type PermissionError struct {
	WarpError
}

// NewPermissionError creates a new permission error.
func NewPermissionError(message string, provider string, err error) *PermissionError {
	return &PermissionError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    403,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// RateLimitError represents a rate limit exceeded error (429).
// This is retryable after the specified retry-after duration.
type RateLimitError struct {
	WarpError

	// RetryAfter specifies how long to wait before retrying.
	RetryAfter time.Duration
}

// NewRateLimitError creates a new rate limit error.
func NewRateLimitError(message string, provider string, retryAfter time.Duration, err error) *RateLimitError {
	return &RateLimitError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    429,
			Provider:      provider,
			OriginalError: err,
		},
		RetryAfter: retryAfter,
	}
}

// IsRetryable returns true for rate limit errors.
// Clients should wait for RetryAfter duration before retrying.
func (e *RateLimitError) IsRetryable() bool {
	return true
}

// ContextWindowExceededError represents a context window exceeded error.
// This occurs when the input tokens exceed the model's maximum context length.
type ContextWindowExceededError struct {
	WarpError

	// MaxTokens is the maximum context window size for this model.
	MaxTokens int

	// Tokens is the actual number of tokens in the request.
	Tokens int
}

// NewContextWindowExceededError creates a new context window exceeded error.
func NewContextWindowExceededError(message string, provider string, maxTokens, tokens int, err error) *ContextWindowExceededError {
	return &ContextWindowExceededError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    400,
			Provider:      provider,
			OriginalError: err,
		},
		MaxTokens: maxTokens,
		Tokens:    tokens,
	}
}

// ContentPolicyViolationError represents a content policy violation error.
// This occurs when the input or output violates the provider's content policy.
type ContentPolicyViolationError struct {
	WarpError
}

// NewContentPolicyViolationError creates a new content policy violation error.
func NewContentPolicyViolationError(message string, provider string, err error) *ContentPolicyViolationError {
	return &ContentPolicyViolationError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    400,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// InvalidRequestError represents an invalid request error (400).
// This occurs when the request parameters are malformed or invalid.
type InvalidRequestError struct {
	WarpError
}

// NewInvalidRequestError creates a new invalid request error.
func NewInvalidRequestError(message string, provider string, err error) *InvalidRequestError {
	return &InvalidRequestError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    400,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// TimeoutError represents a request timeout error.
// This is retryable as it may be due to temporary network issues.
type TimeoutError struct {
	WarpError
}

// NewTimeoutError creates a new timeout error.
func NewTimeoutError(message string, provider string, err error) *TimeoutError {
	return &TimeoutError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    0, // No HTTP status for timeouts
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// IsRetryable returns true for timeout errors.
// Timeouts are often transient and worth retrying.
func (e *TimeoutError) IsRetryable() bool {
	return true
}

// ServiceUnavailableError represents a service unavailable error (503).
// This is retryable as the service may recover shortly.
type ServiceUnavailableError struct {
	WarpError
}

// NewServiceUnavailableError creates a new service unavailable error.
func NewServiceUnavailableError(message string, provider string, err error) *ServiceUnavailableError {
	return &ServiceUnavailableError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    503,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// IsRetryable returns true for service unavailable errors.
// These are typically temporary outages that may resolve quickly.
func (e *ServiceUnavailableError) IsRetryable() bool {
	return true
}

// BadRequestError represents a bad request error (400).
// This is a general 400 error that doesn't fit other categories.
type BadRequestError struct {
	WarpError
}

// NewBadRequestError creates a new bad request error.
func NewBadRequestError(message string, provider string, err error) *BadRequestError {
	return &BadRequestError{
		WarpError: WarpError{
			Message:       message,
			StatusCode:    400,
			Provider:      provider,
			OriginalError: err,
		},
	}
}

// ParseProviderError parses a provider-specific error response into a typed Warp error.
// This function attempts to parse JSON error responses and maps HTTP status codes
// to appropriate error types.
//
// The function handles various error response formats from different providers and
// performs intelligent error classification based on status codes and message content.
//
// Parameters:
//   - provider: The LLM provider name (e.g., "openai", "anthropic")
//   - statusCode: The HTTP status code from the error response
//   - body: The raw response body bytes
//   - err: The original error (if any)
//
// Returns an appropriate error type based on the status code and message content.
func ParseProviderError(provider string, statusCode int, body []byte, err error) error {
	// Attempt to parse JSON error response
	var errorResp struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}

	message := ""
	if jsonErr := json.Unmarshal(body, &errorResp); jsonErr == nil && errorResp.Error.Message != "" {
		message = errorResp.Error.Message
	} else {
		// Fall back to raw body if JSON parsing fails
		message = string(body)
	}

	// Use a descriptive message if body is empty
	if message == "" {
		message = fmt.Sprintf("HTTP %d error", statusCode)
	}

	// Map status codes to specific error types
	switch statusCode {
	case 401:
		return NewAuthenticationError(message, provider, err)

	case 403:
		return NewPermissionError(message, provider, err)

	case 429:
		// Rate limit errors are retryable
		return NewRateLimitError(message, provider, 0, err)

	case 400:
		// 400 errors require content analysis to categorize properly
		lowerMsg := strings.ToLower(message)

		// Check for context window errors
		if strings.Contains(lowerMsg, "context") ||
			strings.Contains(lowerMsg, "token limit") ||
			strings.Contains(lowerMsg, "maximum context length") ||
			strings.Contains(lowerMsg, "too many tokens") {
			return NewContextWindowExceededError(message, provider, 0, 0, err)
		}

		// Check for content policy violations
		if strings.Contains(lowerMsg, "content") && (strings.Contains(lowerMsg, "policy") ||
			strings.Contains(lowerMsg, "filter") ||
			strings.Contains(lowerMsg, "safety")) {
			return NewContentPolicyViolationError(message, provider, err)
		}

		// Check for invalid request
		if strings.Contains(lowerMsg, "invalid") ||
			strings.Contains(lowerMsg, "malformed") ||
			strings.Contains(lowerMsg, "missing required") {
			return NewInvalidRequestError(message, provider, err)
		}

		// Generic bad request if no specific pattern matches
		return NewBadRequestError(message, provider, err)

	case 503:
		// Service unavailable is retryable
		return NewServiceUnavailableError(message, provider, err)

	case 500, 502, 504:
		// Server errors are generally retryable
		// Note: We preserve the actual status code for these server errors
		serviceErr := NewServiceUnavailableError(message, provider, err)
		serviceErr.StatusCode = statusCode
		return serviceErr

	default:
		// Generic API error for unhandled status codes
		return NewAPIError(message, statusCode, provider, err)
	}
}
