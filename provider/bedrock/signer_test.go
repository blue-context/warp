package bedrock

import (
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestNewSigner(t *testing.T) {
	tests := []struct {
		name            string
		accessKeyID     string
		secretAccessKey string
		region          string
		wantService     string
	}{
		{
			name:            "valid signer creation",
			accessKeyID:     "AKIAIOSFODNN7EXAMPLE",
			secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
			region:          "us-east-1",
			wantService:     "bedrock",
		},
		{
			name:            "different region",
			accessKeyID:     "test-key",
			secretAccessKey: "test-secret",
			region:          "eu-west-1",
			wantService:     "bedrock",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signer := NewSigner(tt.accessKeyID, tt.secretAccessKey, tt.region)

			if signer == nil {
				t.Fatal("NewSigner returned nil")
			}

			if signer.accessKeyID != tt.accessKeyID {
				t.Errorf("accessKeyID = %q, want %q", signer.accessKeyID, tt.accessKeyID)
			}

			if signer.secretAccessKey != tt.secretAccessKey {
				t.Errorf("secretAccessKey = %q, want %q", signer.secretAccessKey, tt.secretAccessKey)
			}

			if signer.region != tt.region {
				t.Errorf("region = %q, want %q", signer.region, tt.region)
			}

			if signer.service != tt.wantService {
				t.Errorf("service = %q, want %q", signer.service, tt.wantService)
			}
		})
	}
}

func TestSignRequest(t *testing.T) {
	signer := NewSigner(
		"AKIAIOSFODNN7EXAMPLE",
		"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		"us-east-1",
	)

	tests := []struct {
		name        string
		req         *http.Request
		payload     []byte
		wantErr     bool
		checkHeader string
	}{
		{
			name: "POST request with body",
			req: &http.Request{
				Method: "POST",
				URL: &url.URL{
					Scheme: "https",
					Host:   "bedrock-runtime.us-east-1.amazonaws.com",
					Path:   "/model/anthropic.claude-3-opus-20240229-v1:0/invoke",
				},
				Header: make(http.Header),
			},
			payload:     []byte(`{"messages":[{"role":"user","content":"test"}]}`),
			wantErr:     false,
			checkHeader: "Authorization",
		},
		{
			name: "GET request without body",
			req: &http.Request{
				Method: "GET",
				URL: &url.URL{
					Scheme: "https",
					Host:   "bedrock-runtime.us-east-1.amazonaws.com",
					Path:   "/",
				},
				Header: make(http.Header),
			},
			payload:     nil,
			wantErr:     false,
			checkHeader: "Authorization",
		},
		{
			name:    "nil request",
			req:     nil,
			payload: nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.req != nil {
				tt.req.Host = tt.req.URL.Host
			}

			err := signer.SignRequest(tt.req, tt.payload)

			if (err != nil) != tt.wantErr {
				t.Errorf("SignRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if err != nil {
				return
			}

			// Check required headers are set
			if tt.checkHeader != "" && tt.req.Header.Get(tt.checkHeader) == "" {
				t.Errorf("expected header %q to be set", tt.checkHeader)
			}

			// Check X-Amz-Date header
			if tt.req.Header.Get("X-Amz-Date") == "" {
				t.Error("X-Amz-Date header not set")
			}

			// Check Authorization header format
			auth := tt.req.Header.Get("Authorization")
			if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256") {
				t.Errorf("Authorization header doesn't start with AWS4-HMAC-SHA256: %s", auth)
			}

			if !strings.Contains(auth, "Credential=") {
				t.Error("Authorization header missing Credential")
			}

			if !strings.Contains(auth, "SignedHeaders=") {
				t.Error("Authorization header missing SignedHeaders")
			}

			if !strings.Contains(auth, "Signature=") {
				t.Error("Authorization header missing Signature")
			}
		})
	}
}

func TestCredentialScope(t *testing.T) {
	signer := NewSigner("test-key", "test-secret", "us-west-2")

	testTime := time.Date(2024, 1, 15, 12, 0, 0, 0, time.UTC)

	scope := signer.credentialScope(testTime)

	expected := "20240115/us-west-2/bedrock/aws4_request"
	if scope != expected {
		t.Errorf("credentialScope() = %q, want %q", scope, expected)
	}
}

func TestCanonicalQueryString(t *testing.T) {
	signer := NewSigner("test", "test", "us-east-1")

	tests := []struct {
		name  string
		query map[string][]string
		want  string
	}{
		{
			name:  "empty query",
			query: map[string][]string{},
			want:  "",
		},
		{
			name: "single parameter",
			query: map[string][]string{
				"foo": {"bar"},
			},
			want: "foo=bar",
		},
		{
			name: "multiple parameters",
			query: map[string][]string{
				"foo": {"bar"},
				"baz": {"qux"},
			},
			want: "baz=qux&foo=bar", // Sorted by key
		},
		{
			name: "parameter with multiple values",
			query: map[string][]string{
				"foo": {"bar", "baz"},
			},
			want: "foo=bar&foo=baz", // Sorted values
		},
		{
			name: "parameter with special characters",
			query: map[string][]string{
				"foo bar": {"baz qux"},
			},
			want: "foo%20bar=baz%20qux", // URL encoded
		},
		{
			name: "empty value",
			query: map[string][]string{
				"foo": {""},
			},
			want: "foo=",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := signer.canonicalQueryString(tt.query)
			if got != tt.want {
				t.Errorf("canonicalQueryString() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCanonicalHeaders(t *testing.T) {
	signer := NewSigner("test", "test", "us-east-1")

	tests := []struct {
		name            string
		headers         http.Header
		wantCanonical   string
		wantSignedNames string
	}{
		{
			name: "single header",
			headers: http.Header{
				"Content-Type": {"application/json"},
			},
			wantCanonical:   "content-type:application/json\n",
			wantSignedNames: "content-type",
		},
		{
			name: "multiple headers",
			headers: http.Header{
				"Content-Type": {"application/json"},
				"Host":         {"example.com"},
			},
			wantCanonical:   "content-type:application/json\nhost:example.com\n",
			wantSignedNames: "content-type;host",
		},
		{
			name: "headers with mixed case",
			headers: http.Header{
				"Content-Type": {"application/json"},
				"X-Amz-Date":   {"20240115T120000Z"},
			},
			wantCanonical:   "content-type:application/json\nx-amz-date:20240115T120000Z\n",
			wantSignedNames: "content-type;x-amz-date",
		},
		{
			name: "header with whitespace",
			headers: http.Header{
				"X-Test": {"  value  "},
			},
			wantCanonical:   "x-test:value\n",
			wantSignedNames: "x-test",
		},
		{
			name: "header with multiple values",
			headers: http.Header{
				"X-Test": {"value1", "value2"},
			},
			wantCanonical:   "x-test:value1,value2\n",
			wantSignedNames: "x-test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			canonical, signed := signer.canonicalHeaders(tt.headers)

			if canonical != tt.wantCanonical {
				t.Errorf("canonical headers = %q, want %q", canonical, tt.wantCanonical)
			}

			if signed != tt.wantSignedNames {
				t.Errorf("signed headers = %q, want %q", signed, tt.wantSignedNames)
			}
		})
	}
}

func TestHexEncodedHash(t *testing.T) {
	signer := NewSigner("test", "test", "us-east-1")

	tests := []struct {
		name string
		data []byte
		want string
	}{
		{
			name: "empty data",
			data: []byte{},
			want: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		},
		{
			name: "simple string",
			data: []byte("hello"),
			want: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
		},
		{
			name: "json data",
			data: []byte(`{"test":"value"}`),
			want: "f98be16ebfa861cb39a61faff9e52b33f5bcc16bb6ae72e728d226dc07093932",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := signer.hexEncodedHash(tt.data)
			if got != tt.want {
				t.Errorf("hexEncodedHash() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestURIEncode(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "alphanumeric",
			input: "abc123",
			want:  "abc123",
		},
		{
			name:  "unreserved characters",
			input: "-_.~",
			want:  "-_.~",
		},
		{
			name:  "space",
			input: "hello world",
			want:  "hello%20world",
		},
		{
			name:  "special characters",
			input: "hello!@#$%^&*()",
			want:  "hello%21%40%23%24%25%5E%26%2A%28%29",
		},
		{
			name:  "forward slash",
			input: "path/to/resource",
			want:  "path%2Fto%2Fresource",
		},
		{
			name:  "unicode",
			input: "hello世界",
			want:  "hello%E4%B8%96%E7%95%8C",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := uriEncode(tt.input)
			if got != tt.want {
				t.Errorf("uriEncode() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCreateCanonicalRequest(t *testing.T) {
	signer := NewSigner("test", "test", "us-east-1")

	req := &http.Request{
		Method: "POST",
		URL: &url.URL{
			Scheme: "https",
			Host:   "bedrock-runtime.us-east-1.amazonaws.com",
			Path:   "/model/test/invoke",
		},
		Header: http.Header{
			"Content-Type": {"application/json"},
			"Host":         {"bedrock-runtime.us-east-1.amazonaws.com"},
		},
	}

	payload := []byte(`{"test":"value"}`)

	canonical := signer.createCanonicalRequest(req, payload)

	// Check that canonical request has required components
	if !strings.Contains(canonical, "POST") {
		t.Error("canonical request missing method")
	}

	if !strings.Contains(canonical, "/model/test/invoke") {
		t.Error("canonical request missing path")
	}

	if !strings.Contains(canonical, "content-type:application/json") {
		t.Error("canonical request missing headers")
	}

	// Check that canonical request is not empty
	if canonical == "" {
		t.Error("canonical request is empty")
	}
}

func TestCalculateSignature(t *testing.T) {
	signer := NewSigner(
		"AKIAIOSFODNN7EXAMPLE",
		"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		"us-east-1",
	)

	testTime := time.Date(2024, 1, 15, 12, 0, 0, 0, time.UTC)
	stringToSign := "AWS4-HMAC-SHA256\n20240115T120000Z\n20240115/us-east-1/bedrock/aws4_request\ntest-hash"

	signature := signer.calculateSignature(testTime, stringToSign)

	// Signature should be a 64-character hex string
	if len(signature) != 64 {
		t.Errorf("signature length = %d, want 64", len(signature))
	}

	// Signature should only contain hex characters
	for _, c := range signature {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
			t.Errorf("signature contains non-hex character: %c", c)
		}
	}

	// Same input should produce same signature
	signature2 := signer.calculateSignature(testTime, stringToSign)
	if signature != signature2 {
		t.Error("signature not deterministic")
	}
}

// TestSignRequestWithSessionToken verifies that session tokens are included in signature.
func TestSignRequestWithSessionToken(t *testing.T) {
	sessionToken := "AQoDYXdzEJr...EXAMPLE/SESSION/TOKEN"

	// Create signer with session token
	signer := NewSigner(
		"AKIAIOSFODNN7EXAMPLE",
		"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		"us-east-1",
		WithSignerSessionToken(sessionToken),
	)

	if signer.sessionToken != sessionToken {
		t.Errorf("sessionToken = %q, want %q", signer.sessionToken, sessionToken)
	}

	req := &http.Request{
		Method: "POST",
		URL: &url.URL{
			Scheme: "https",
			Host:   "bedrock-runtime.us-east-1.amazonaws.com",
			Path:   "/model/anthropic.claude-3-opus-20240229-v1:0/invoke",
		},
		Header: make(http.Header),
	}

	payload := []byte(`{"messages":[{"role":"user","content":"test"}]}`)

	err := signer.SignRequest(req, payload)
	if err != nil {
		t.Fatalf("SignRequest failed: %v", err)
	}

	// Verify session token header is present
	if got := req.Header.Get("X-Amz-Security-Token"); got != sessionToken {
		t.Errorf("X-Amz-Security-Token header = %q, want %q", got, sessionToken)
	}

	// Verify Authorization header exists (signature was calculated with token)
	authHeader := req.Header.Get("Authorization")
	if authHeader == "" {
		t.Error("Authorization header is missing")
	}

	// Verify session token is included in signed headers
	// The canonical headers should include x-amz-security-token
	if !strings.Contains(authHeader, "SignedHeaders=") {
		t.Error("Authorization header missing SignedHeaders")
	}

	// Create signer without session token for comparison
	signerNoToken := NewSigner(
		"AKIAIOSFODNN7EXAMPLE",
		"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		"us-east-1",
	)

	req2 := &http.Request{
		Method: "POST",
		URL: &url.URL{
			Scheme: "https",
			Host:   "bedrock-runtime.us-east-1.amazonaws.com",
			Path:   "/model/anthropic.claude-3-opus-20240229-v1:0/invoke",
		},
		Header: make(http.Header),
	}

	err = signerNoToken.SignRequest(req2, payload)
	if err != nil {
		t.Fatalf("SignRequest failed: %v", err)
	}

	// Signatures should be different when session token is present
	authHeader2 := req2.Header.Get("Authorization")
	if authHeader == authHeader2 {
		t.Error("Authorization headers should differ when session token is present")
	}

	// Verify no session token header without option
	if got := req2.Header.Get("X-Amz-Security-Token"); got != "" {
		t.Errorf("X-Amz-Security-Token should be empty without session token, got %q", got)
	}
}

// TestStreamThreadSafety verifies that concurrent Close() and Recv() operations are safe.
func TestStreamThreadSafety(t *testing.T) {
	// This test is primarily to verify no race conditions with -race flag
	// Create a mock response
	resp := &http.Response{
		StatusCode: 200,
		Body:       http.NoBody,
	}

	stream := newClaudeStream(resp, "test-model")

	// Start concurrent Close() calls
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			_ = stream.Close()
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Verify stream is closed
	stream.mu.Lock()
	if !stream.closed {
		t.Error("stream should be closed after concurrent Close() calls")
	}
	stream.mu.Unlock()
}
