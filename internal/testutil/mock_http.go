package testutil

import (
	"bytes"
	"io"
	"net/http"
	"strings"
)

// MockHTTPClient is a mock HTTP client for testing.
//
// It allows you to define custom responses for HTTP requests and tracks
// all requests made for verification in tests.
//
// Example:
//
//	mock := &testutil.MockHTTPClient{
//	    DoFunc: func(req *http.Request) (*http.Response, error) {
//	        return &http.Response{
//	            StatusCode: 200,
//	            Body:       io.NopCloser(strings.NewReader(`{"result": "ok"}`)),
//	        }, nil
//	    },
//	}
//	client := &http.Client{Transport: mock}
type MockHTTPClient struct {
	// DoFunc is the function to call when Do is invoked.
	// If nil, a default 200 OK response with empty JSON body is returned.
	DoFunc func(req *http.Request) (*http.Response, error)

	// RequestsMade tracks all requests made to this client.
	// Useful for verifying that the correct requests were made in tests.
	RequestsMade []*http.Request
}

// Do executes the mock function and tracks the request.
//
// This method implements the http.RoundTripper interface, making MockHTTPClient
// usable as an http.Client transport.
func (m *MockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	m.RequestsMade = append(m.RequestsMade, req)
	if m.DoFunc != nil {
		return m.DoFunc(req)
	}
	// Default: return 200 OK with empty JSON body
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(bytes.NewReader([]byte("{}"))),
		Header:     make(http.Header),
	}, nil
}

// RoundTrip implements http.RoundTripper interface.
//
// This allows MockHTTPClient to be used as http.Client.Transport.
func (m *MockHTTPClient) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.Do(req)
}

// MockResponse is a helper to create mock HTTP responses.
//
// Example:
//
//	resp := testutil.MockResponse(200, `{"result": "ok"}`)
func MockResponse(statusCode int, body string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

// MockErrorResponse creates a mock error response.
//
// This is a convenience helper for creating error responses with the
// appropriate status code and error body.
//
// Example:
//
//	resp := testutil.MockErrorResponse(401, `{"error": {"message": "Unauthorized"}}`)
func MockErrorResponse(statusCode int, errorBody string) *http.Response {
	return &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(strings.NewReader(errorBody)),
		Header:     make(http.Header),
	}
}
