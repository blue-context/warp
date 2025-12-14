// Package vertex implements the Google Vertex AI provider for Warp.
//
// This package provides Vertex AI support with zero dependencies by implementing
// OAuth2 service account authentication using only the Go standard library.
package vertex

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"
)

const (
	// OAuth2 token endpoint for GCP
	defaultTokenURI = "https://oauth2.googleapis.com/token"

	// OAuth2 scopes required for Vertex AI
	vertexAIScope = "https://www.googleapis.com/auth/cloud-platform"

	// JWT validity duration
	jwtExpiration = 1 * time.Hour

	// Token refresh buffer - refresh token this much before expiration
	tokenRefreshBuffer = 5 * time.Minute
)

// TokenProvider manages OAuth2 access tokens for GCP service accounts.
//
// This implementation creates and signs JWT assertions using RS256 (RSA + SHA256)
// and exchanges them for OAuth2 access tokens via Google's token endpoint.
//
// Token caching is implemented to avoid unnecessary token exchanges. Tokens are
// refreshed automatically when they approach expiration.
//
// Thread Safety: TokenProvider is safe for concurrent use.
// Multiple goroutines may call GetToken simultaneously.
type TokenProvider struct {
	serviceAccountKey *ServiceAccountKey
	httpClient        *http.Client

	// Token cache with mutex protection
	mu         sync.RWMutex
	token      string
	expiration time.Time
}

// ServiceAccountKey represents a GCP service account key file.
//
// This structure matches the JSON format of GCP service account keys
// downloaded from the Google Cloud Console.
//
// The PrivateKey field contains a PEM-encoded PKCS#8 RSA private key.
type ServiceAccountKey struct {
	Type                    string `json:"type"`
	ProjectID               string `json:"project_id"`
	PrivateKeyID            string `json:"private_key_id"`
	PrivateKey              string `json:"private_key"`
	ClientEmail             string `json:"client_email"`
	ClientID                string `json:"client_id"`
	AuthURI                 string `json:"auth_uri"`
	TokenURI                string `json:"token_uri"`
	AuthProviderX509CertURL string `json:"auth_provider_x509_cert_url"`
	ClientX509CertURL       string `json:"client_x509_cert_url"`
}

// NewTokenProvider creates a new OAuth2 token provider from service account JSON.
//
// The serviceAccountJSON parameter should contain the contents of a GCP service
// account key file in JSON format. This file can be created in the Google Cloud
// Console under IAM & Admin > Service Accounts.
//
// Example:
//
//	keyJSON, err := os.ReadFile("service-account-key.json")
//	if err != nil {
//	    return err
//	}
//	provider, err := vertex.NewTokenProvider(keyJSON)
//	if err != nil {
//	    return err
//	}
func NewTokenProvider(serviceAccountJSON []byte) (*TokenProvider, error) {
	if len(serviceAccountJSON) == 0 {
		return nil, fmt.Errorf("service account JSON is empty")
	}

	var key ServiceAccountKey
	if err := json.Unmarshal(serviceAccountJSON, &key); err != nil {
		return nil, fmt.Errorf("failed to parse service account key: %w", err)
	}

	// Validate required fields
	if key.ClientEmail == "" {
		return nil, fmt.Errorf("service account key missing client_email")
	}
	if key.PrivateKey == "" {
		return nil, fmt.Errorf("service account key missing private_key")
	}

	// Set default token URI if not specified
	if key.TokenURI == "" {
		key.TokenURI = defaultTokenURI
	}

	// Validate private key format
	if err := validatePrivateKey(key.PrivateKey); err != nil {
		return nil, fmt.Errorf("invalid private key: %w", err)
	}

	return &TokenProvider{
		serviceAccountKey: &key,
		httpClient:        &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// GetToken returns a valid OAuth2 access token, refreshing if needed.
//
// This method implements token caching to avoid unnecessary token exchanges.
// If a cached token exists and is still valid (not expired and not close to
// expiration), it is returned immediately.
//
// Otherwise, a new JWT assertion is created, signed with the service account's
// private key, and exchanged for a fresh access token.
//
// Thread Safety: This method is safe for concurrent use.
// Multiple goroutines may call GetToken simultaneously.
//
// Example:
//
//	token, err := provider.GetToken()
//	if err != nil {
//	    return err
//	}
//	req.Header.Set("Authorization", "Bearer "+token)
func (t *TokenProvider) GetToken() (string, error) {
	// Fast path: return cached token if still valid
	t.mu.RLock()
	if t.token != "" && time.Now().Add(tokenRefreshBuffer).Before(t.expiration) {
		token := t.token
		t.mu.RUnlock()
		return token, nil
	}
	t.mu.RUnlock()

	// Slow path: refresh token
	t.mu.Lock()
	defer t.mu.Unlock()

	// Double-check after acquiring write lock (another goroutine may have refreshed)
	if t.token != "" && time.Now().Add(tokenRefreshBuffer).Before(t.expiration) {
		return t.token, nil
	}

	// Create JWT assertion
	jwt, err := t.createJWT()
	if err != nil {
		return "", fmt.Errorf("failed to create JWT: %w", err)
	}

	// Exchange JWT for access token
	token, expiration, err := t.exchangeJWT(jwt)
	if err != nil {
		return "", fmt.Errorf("failed to exchange JWT for token: %w", err)
	}

	// Cache token
	t.token = token
	t.expiration = expiration

	return token, nil
}

// createJWT creates a signed JWT assertion for OAuth2 token exchange.
//
// JWT structure:
//  1. Header: {"alg": "RS256", "typ": "JWT"}
//  2. Claims: {iss, scope, aud, exp, iat}
//  3. Signature: RS256 signature of header.claims
//
// The JWT is base64url-encoded as: header.claims.signature
//
// Reference: https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
func (t *TokenProvider) createJWT() (string, error) {
	now := time.Now()

	// JWT header
	header := map[string]string{
		"alg": "RS256",
		"typ": "JWT",
	}

	// JWT claims
	claims := map[string]interface{}{
		"iss":   t.serviceAccountKey.ClientEmail,
		"scope": vertexAIScope,
		"aud":   t.serviceAccountKey.TokenURI,
		"exp":   now.Add(jwtExpiration).Unix(),
		"iat":   now.Unix(),
	}

	// Encode header and claims
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return "", fmt.Errorf("failed to marshal header: %w", err)
	}

	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		return "", fmt.Errorf("failed to marshal claims: %w", err)
	}

	// Base64url encode (without padding)
	headerB64 := base64.RawURLEncoding.EncodeToString(headerJSON)
	claimsB64 := base64.RawURLEncoding.EncodeToString(claimsJSON)

	// Create signing input
	signInput := headerB64 + "." + claimsB64

	// Sign with RS256
	signature, err := t.signJWT(signInput)
	if err != nil {
		return "", fmt.Errorf("failed to sign JWT: %w", err)
	}

	// Base64url encode signature
	signatureB64 := base64.RawURLEncoding.EncodeToString(signature)

	// Build complete JWT
	return signInput + "." + signatureB64, nil
}

// signJWT signs the JWT input using RS256 (RSA + SHA256).
//
// Process:
//  1. Parse PEM-encoded private key from service account
//  2. Decode PKCS#8 private key
//  3. Compute SHA256 hash of input
//  4. Sign hash with RSA private key using PKCS#1 v1.5
//
// The signature is returned as raw bytes (caller will base64url encode).
func (t *TokenProvider) signJWT(input string) ([]byte, error) {
	// Parse PEM block
	block, _ := pem.Decode([]byte(t.serviceAccountKey.PrivateKey))
	if block == nil {
		return nil, fmt.Errorf("failed to decode PEM block from private key")
	}

	// Parse PKCS#8 private key
	privateKey, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse PKCS#8 private key: %w", err)
	}

	// Ensure it's an RSA private key
	rsaKey, ok := privateKey.(*rsa.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("private key is not RSA (got %T)", privateKey)
	}

	// Compute SHA256 hash
	hash := sha256.Sum256([]byte(input))

	// Sign with RSA-SHA256 (PKCS#1 v1.5)
	signature, err := rsa.SignPKCS1v15(rand.Reader, rsaKey, crypto.SHA256, hash[:])
	if err != nil {
		return nil, fmt.Errorf("failed to sign with RSA: %w", err)
	}

	return signature, nil
}

// exchangeJWT exchanges a JWT assertion for an OAuth2 access token.
//
// This method POSTs the JWT to Google's token endpoint with:
//
//	grant_type: urn:ietf:params:oauth:grant-type:jwt-bearer
//	assertion: <signed JWT>
//
// The response contains:
//
//	access_token: The OAuth2 access token
//	expires_in: Token lifetime in seconds (typically 3600)
//	token_type: "Bearer"
//
// Reference: https://developers.google.com/identity/protocols/oauth2/service-account#httprest
func (t *TokenProvider) exchangeJWT(jwt string) (token string, expiration time.Time, err error) {
	// Build form data
	data := url.Values{
		"grant_type": {"urn:ietf:params:oauth:grant-type:jwt-bearer"},
		"assertion":  {jwt},
	}

	// POST to token endpoint
	resp, err := t.httpClient.PostForm(t.serviceAccountKey.TokenURI, data)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", time.Time{}, fmt.Errorf("token exchange failed (HTTP %d): %s",
			resp.StatusCode, string(body))
	}

	// Parse response
	var tokenResp struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
		TokenType   string `json:"token_type"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return "", time.Time{}, fmt.Errorf("failed to decode token response: %w", err)
	}

	// Validate response
	if tokenResp.AccessToken == "" {
		return "", time.Time{}, fmt.Errorf("token response missing access_token")
	}

	// Calculate expiration time
	expiration = time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second)

	return tokenResp.AccessToken, expiration, nil
}

// validatePrivateKey validates that the private key is properly formatted.
//
// This performs a quick check to ensure the key is PEM-encoded and can be decoded.
// It does not verify the key is valid for signing (that happens during signing).
func validatePrivateKey(privateKey string) error {
	block, _ := pem.Decode([]byte(privateKey))
	if block == nil {
		return fmt.Errorf("private key is not PEM-encoded")
	}

	// Try to parse as PKCS#8
	_, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return fmt.Errorf("private key is not valid PKCS#8: %w", err)
	}

	return nil
}
