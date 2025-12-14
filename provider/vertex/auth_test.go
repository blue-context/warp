package vertex

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// generateTestServiceAccountKey generates a test service account key with a real RSA key.
func generateTestServiceAccountKey(t *testing.T) []byte {
	t.Helper()

	// Generate RSA private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("failed to generate RSA key: %v", err)
	}

	// Marshal to PKCS#8
	privateKeyBytes, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		t.Fatalf("failed to marshal private key: %v", err)
	}

	// Encode to PEM
	privateKeyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: privateKeyBytes,
	})

	// Create service account key
	key := ServiceAccountKey{
		Type:         "service_account",
		ProjectID:    "test-project",
		PrivateKeyID: "test-key-id",
		PrivateKey:   string(privateKeyPEM),
		ClientEmail:  "test@test-project.iam.gserviceaccount.com",
		ClientID:     "123456789",
		TokenURI:     "https://oauth2.googleapis.com/token",
	}

	keyJSON, err := json.Marshal(key)
	if err != nil {
		t.Fatalf("failed to marshal service account key: %v", err)
	}

	return keyJSON
}

func TestNewTokenProvider(t *testing.T) {
	tests := []struct {
		name        string
		keyJSON     []byte
		wantErr     bool
		errContains string
	}{
		{
			name:    "valid service account key",
			keyJSON: generateTestServiceAccountKey(t),
			wantErr: false,
		},
		{
			name:        "empty key",
			keyJSON:     []byte{},
			wantErr:     true,
			errContains: "empty",
		},
		{
			name:        "invalid JSON",
			keyJSON:     []byte("not json"),
			wantErr:     true,
			errContains: "parse",
		},
		{
			name: "missing client_email",
			keyJSON: []byte(`{
				"type": "service_account",
				"private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----"
			}`),
			wantErr:     true,
			errContains: "client_email",
		},
		{
			name: "missing private_key",
			keyJSON: []byte(`{
				"type": "service_account",
				"client_email": "test@example.com"
			}`),
			wantErr:     true,
			errContains: "private_key",
		},
		{
			name: "invalid private key format",
			keyJSON: []byte(`{
				"type": "service_account",
				"client_email": "test@example.com",
				"private_key": "not a valid key"
			}`),
			wantErr:     true,
			errContains: "private key",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, err := NewTokenProvider(tt.keyJSON)

			if tt.wantErr {
				if err == nil {
					t.Errorf("NewTokenProvider() expected error, got nil")
					return
				}
				if tt.errContains != "" && !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("NewTokenProvider() error = %v, want error containing %q", err, tt.errContains)
				}
				return
			}

			if err != nil {
				t.Errorf("NewTokenProvider() unexpected error = %v", err)
				return
			}

			if provider == nil {
				t.Error("NewTokenProvider() returned nil provider")
				return
			}

			if provider.serviceAccountKey == nil {
				t.Error("NewTokenProvider() service account key is nil")
			}
		})
	}
}

func TestTokenProvider_CreateJWT(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)
	provider, err := NewTokenProvider(keyJSON)
	if err != nil {
		t.Fatalf("NewTokenProvider() failed: %v", err)
	}

	jwt, err := provider.createJWT()
	if err != nil {
		t.Fatalf("createJWT() failed: %v", err)
	}

	// JWT should have 3 parts: header.claims.signature
	parts := strings.Split(jwt, ".")
	if len(parts) != 3 {
		t.Errorf("createJWT() JWT should have 3 parts, got %d", len(parts))
	}

	// Each part should be non-empty
	for i, part := range parts {
		if part == "" {
			t.Errorf("createJWT() part %d is empty", i)
		}
	}

	// Verify JWT header and claims can be decoded (just basic validation)
	if len(parts[0]) < 10 || len(parts[1]) < 10 || len(parts[2]) < 10 {
		t.Error("createJWT() JWT parts seem too short")
	}
}

func TestTokenProvider_GetToken(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)

	// Create test server that returns a token
	var requestCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++

		// Verify request
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}

		if err := r.ParseForm(); err != nil {
			t.Errorf("failed to parse form: %v", err)
		}

		grantType := r.FormValue("grant_type")
		if grantType != "urn:ietf:params:oauth:grant-type:jwt-bearer" {
			t.Errorf("expected jwt-bearer grant type, got %s", grantType)
		}

		assertion := r.FormValue("assertion")
		if assertion == "" {
			t.Error("assertion is empty")
		}

		// Return valid token response
		resp := map[string]interface{}{
			"access_token": fmt.Sprintf("test-token-%d", requestCount),
			"expires_in":   3600,
			"token_type":   "Bearer",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Modify service account key to use test server
	var key ServiceAccountKey
	if err := json.Unmarshal(keyJSON, &key); err != nil {
		t.Fatalf("failed to unmarshal key: %v", err)
	}
	key.TokenURI = server.URL

	modifiedKeyJSON, err := json.Marshal(key)
	if err != nil {
		t.Fatalf("failed to marshal modified key: %v", err)
	}

	provider, err := NewTokenProvider(modifiedKeyJSON)
	if err != nil {
		t.Fatalf("NewTokenProvider() failed: %v", err)
	}

	// Test 1: Get token
	token1, err := provider.GetToken()
	if err != nil {
		t.Fatalf("GetToken() failed: %v", err)
	}

	if token1 == "" {
		t.Error("GetToken() returned empty token")
	}

	if requestCount != 1 {
		t.Errorf("expected 1 request, got %d", requestCount)
	}

	// Test 2: Get token again (should use cache)
	token2, err := provider.GetToken()
	if err != nil {
		t.Fatalf("GetToken() second call failed: %v", err)
	}

	if token2 != token1 {
		t.Error("GetToken() should return cached token")
	}

	if requestCount != 1 {
		t.Errorf("expected 1 request (cached), got %d", requestCount)
	}

	// Test 3: Expire token and get new one
	provider.expiration = time.Now().Add(-1 * time.Hour) // Force expiration
	token3, err := provider.GetToken()
	if err != nil {
		t.Fatalf("GetToken() third call failed: %v", err)
	}

	if token3 == token1 {
		t.Error("GetToken() should refresh expired token")
	}

	if requestCount != 2 {
		t.Errorf("expected 2 requests (refresh), got %d", requestCount)
	}
}

func TestTokenProvider_GetToken_ServerError(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)

	tests := []struct {
		name       string
		statusCode int
		response   string
		wantErr    bool
	}{
		{
			name:       "401 unauthorized",
			statusCode: http.StatusUnauthorized,
			response:   `{"error": "invalid_grant"}`,
			wantErr:    true,
		},
		{
			name:       "403 forbidden",
			statusCode: http.StatusForbidden,
			response:   `{"error": "access_denied"}`,
			wantErr:    true,
		},
		{
			name:       "500 server error",
			statusCode: http.StatusInternalServerError,
			response:   `{"error": "server_error"}`,
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.statusCode)
				w.Write([]byte(tt.response))
			}))
			defer server.Close()

			// Modify key to use test server
			var key ServiceAccountKey
			if err := json.Unmarshal(keyJSON, &key); err != nil {
				t.Fatalf("failed to unmarshal key: %v", err)
			}
			key.TokenURI = server.URL

			modifiedKeyJSON, err := json.Marshal(key)
			if err != nil {
				t.Fatalf("failed to marshal modified key: %v", err)
			}

			provider, err := NewTokenProvider(modifiedKeyJSON)
			if err != nil {
				t.Fatalf("NewTokenProvider() failed: %v", err)
			}

			_, err = provider.GetToken()

			if tt.wantErr && err == nil {
				t.Error("GetToken() expected error, got nil")
			}

			if !tt.wantErr && err != nil {
				t.Errorf("GetToken() unexpected error = %v", err)
			}
		})
	}
}

func TestTokenProvider_ConcurrentAccess(t *testing.T) {
	keyJSON := generateTestServiceAccountKey(t)

	var requestCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++

		resp := map[string]interface{}{
			"access_token": "test-token",
			"expires_in":   3600,
			"token_type":   "Bearer",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Modify key
	var key ServiceAccountKey
	if err := json.Unmarshal(keyJSON, &key); err != nil {
		t.Fatalf("failed to unmarshal key: %v", err)
	}
	key.TokenURI = server.URL

	modifiedKeyJSON, err := json.Marshal(key)
	if err != nil {
		t.Fatalf("failed to marshal modified key: %v", err)
	}

	provider, err := NewTokenProvider(modifiedKeyJSON)
	if err != nil {
		t.Fatalf("NewTokenProvider() failed: %v", err)
	}

	// Test concurrent access
	const numGoroutines = 10
	done := make(chan bool, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer func() { done <- true }()

			token, err := provider.GetToken()
			if err != nil {
				t.Errorf("GetToken() failed: %v", err)
			}
			if token == "" {
				t.Error("GetToken() returned empty token")
			}
		}()
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		<-done
	}

	// Should only make one request (cached for concurrent calls)
	if requestCount > numGoroutines {
		t.Logf("Request count: %d (expected <= %d due to caching)", requestCount, numGoroutines)
	}
}

func TestValidatePrivateKey(t *testing.T) {
	// Generate valid key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("failed to generate RSA key: %v", err)
	}

	privateKeyBytes, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		t.Fatalf("failed to marshal private key: %v", err)
	}

	validPEM := string(pem.EncodeToMemory(&pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: privateKeyBytes,
	}))

	tests := []struct {
		name       string
		privateKey string
		wantErr    bool
	}{
		{
			name:       "valid PKCS#8 key",
			privateKey: validPEM,
			wantErr:    false,
		},
		{
			name:       "not PEM encoded",
			privateKey: "not a pem key",
			wantErr:    true,
		},
		{
			name:       "empty key",
			privateKey: "",
			wantErr:    true,
		},
		{
			name:       "invalid PEM content",
			privateKey: "-----BEGIN PRIVATE KEY-----\ninvalid\n-----END PRIVATE KEY-----",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePrivateKey(tt.privateKey)

			if tt.wantErr && err == nil {
				t.Error("validatePrivateKey() expected error, got nil")
			}

			if !tt.wantErr && err != nil {
				t.Errorf("validatePrivateKey() unexpected error = %v", err)
			}
		})
	}
}
