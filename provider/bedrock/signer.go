// Package bedrock implements the AWS Bedrock provider for Warp.
//
// This package provides AWS Bedrock support with zero dependencies by implementing
// AWS Signature Version 4 signing using only the Go standard library.
package bedrock

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	// AWS v4 signature constants
	algorithm       = "AWS4-HMAC-SHA256"
	timeFormat      = "20060102T150405Z"
	shortTimeFormat = "20060102"
	serviceName     = "bedrock"
)

// Signer signs AWS requests with Signature Version 4.
//
// AWS Signature Version 4 is a cryptographic signing process that authenticates
// requests to AWS services. This implementation uses only crypto/sha256 and crypto/hmac
// from the standard library, with zero external dependencies.
//
// Thread Safety: Signer is safe for concurrent use.
// Multiple goroutines may call SignRequest simultaneously.
type Signer struct {
	accessKeyID     string
	secretAccessKey string
	sessionToken    string
	region          string
	service         string
}

// SignerOption is a functional option for configuring the Signer.
type SignerOption func(*Signer)

// WithSignerSessionToken sets the session token for temporary AWS credentials.
//
// Session tokens are required when using AWS STS temporary credentials.
// The token must be included in the signature calculation.
func WithSignerSessionToken(token string) SignerOption {
	return func(s *Signer) {
		s.sessionToken = token
	}
}

// NewSigner creates a new AWS request signer.
//
// The signer implements AWS Signature Version 4 signing for Bedrock requests.
// It requires valid AWS credentials and a region.
//
// Example:
//
//	signer := bedrock.NewSigner(
//	    os.Getenv("AWS_ACCESS_KEY_ID"),
//	    os.Getenv("AWS_SECRET_ACCESS_KEY"),
//	    "us-east-1",
//	)
//
// With session token for temporary credentials:
//
//	signer := bedrock.NewSigner(
//	    accessKeyID,
//	    secretAccessKey,
//	    "us-east-1",
//	    bedrock.WithSignerSessionToken(sessionToken),
//	)
func NewSigner(accessKeyID, secretAccessKey, region string, opts ...SignerOption) *Signer {
	s := &Signer{
		accessKeyID:     accessKeyID,
		secretAccessKey: secretAccessKey,
		region:          region,
		service:         serviceName,
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

// SignRequest signs an HTTP request with AWS Signature V4.
//
// This method modifies the request by adding the following headers:
//   - Authorization: Contains the AWS v4 signature
//   - X-Amz-Date: The request timestamp
//   - X-Amz-Security-Token: The session token (if using temporary credentials)
//   - Host: The request host
//
// The payload parameter should contain the request body bytes (or nil for GET requests).
//
// AWS Signature V4 Process:
//  1. Create canonical request (standardized representation)
//  2. Create string to sign (includes timestamp and scope)
//  3. Calculate signing key (derived from secret key)
//  4. Calculate signature (HMAC-SHA256 of string to sign)
//  5. Add Authorization header with signature
//
// CRITICAL: When using temporary credentials (STS), the session token MUST be
// added to the request BEFORE signing and included in the canonical headers.
//
// Reference: https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
func (s *Signer) SignRequest(req *http.Request, payload []byte) error {
	if req == nil {
		return fmt.Errorf("request cannot be nil")
	}

	// Use current time for signing
	now := time.Now().UTC()

	// Ensure Host header is set
	if req.Host == "" && req.URL != nil {
		req.Host = req.URL.Host
	}

	// Add required AWS headers before signing
	req.Header.Set("X-Amz-Date", now.Format(timeFormat))
	if req.Host != "" {
		req.Header.Set("Host", req.Host)
	}

	// CRITICAL: Add session token BEFORE signing if using temporary credentials
	// The X-Amz-Security-Token header must be included in the canonical headers
	// for the signature calculation to be valid with STS credentials
	if s.sessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", s.sessionToken)
	}

	// Step 1: Create canonical request
	canonicalRequest := s.createCanonicalRequest(req, payload)

	// Step 2: Create string to sign
	credentialScope := s.credentialScope(now)
	stringToSign := s.createStringToSign(now, credentialScope, canonicalRequest)

	// Step 3: Calculate signature
	signature := s.calculateSignature(now, stringToSign)

	// Step 4: Build and add authorization header
	authorization := s.authorizationHeader(now, credentialScope, req.Header, signature)
	req.Header.Set("Authorization", authorization)

	return nil
}

// createCanonicalRequest builds the canonical request string.
//
// Canonical request format:
//
//	HTTP_METHOD + "\n" +
//	CANONICAL_URI + "\n" +
//	CANONICAL_QUERY_STRING + "\n" +
//	CANONICAL_HEADERS + "\n" +
//	SIGNED_HEADERS + "\n" +
//	HASHED_PAYLOAD
func (s *Signer) createCanonicalRequest(req *http.Request, payload []byte) string {
	// HTTP method (uppercase)
	method := req.Method

	// Canonical URI (path)
	uri := req.URL.Path
	if uri == "" {
		uri = "/"
	}

	// Canonical query string (sorted by key)
	query := s.canonicalQueryString(req.URL.Query())

	// Canonical headers (sorted, lowercase, trimmed)
	canonicalHeaders, signedHeaders := s.canonicalHeaders(req.Header)

	// Payload hash
	payloadHash := s.hexEncodedHash(payload)

	// Build canonical request
	canonical := fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s",
		method,
		uri,
		query,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	)

	return canonical
}

// createStringToSign builds the string to sign.
//
// String to sign format:
//
//	ALGORITHM + "\n" +
//	REQUEST_DATETIME + "\n" +
//	CREDENTIAL_SCOPE + "\n" +
//	HASHED_CANONICAL_REQUEST
func (s *Signer) createStringToSign(t time.Time, credentialScope, canonicalRequest string) string {
	hashedCanonicalRequest := s.hexEncodedHash([]byte(canonicalRequest))

	return fmt.Sprintf("%s\n%s\n%s\n%s",
		algorithm,
		t.Format(timeFormat),
		credentialScope,
		hashedCanonicalRequest,
	)
}

// calculateSignature computes the AWS v4 signature.
//
// Signature calculation:
//  1. Derive signing key from secret key, date, region, and service
//  2. HMAC-SHA256 the string to sign with the signing key
//  3. Hex encode the result
func (s *Signer) calculateSignature(t time.Time, stringToSign string) string {
	// Derive signing key
	date := t.Format(shortTimeFormat)
	kDate := s.hmacSHA256([]byte("AWS4"+s.secretAccessKey), []byte(date))
	kRegion := s.hmacSHA256(kDate, []byte(s.region))
	kService := s.hmacSHA256(kRegion, []byte(s.service))
	kSigning := s.hmacSHA256(kService, []byte("aws4_request"))

	// Calculate signature
	signature := s.hmacSHA256(kSigning, []byte(stringToSign))
	return hex.EncodeToString(signature)
}

// authorizationHeader builds the Authorization header value.
//
// Authorization header format:
//
//	ALGORITHM Credential=ACCESS_KEY/CREDENTIAL_SCOPE,
//	SignedHeaders=SIGNED_HEADERS,
//	Signature=SIGNATURE
func (s *Signer) authorizationHeader(t time.Time, credentialScope string, headers http.Header, signature string) string {
	_, signedHeaders := s.canonicalHeaders(headers)

	credential := fmt.Sprintf("%s/%s", s.accessKeyID, credentialScope)

	return fmt.Sprintf("%s Credential=%s, SignedHeaders=%s, Signature=%s",
		algorithm,
		credential,
		signedHeaders,
		signature,
	)
}

// credentialScope builds the credential scope string.
//
// Credential scope format: DATE/REGION/SERVICE/aws4_request
func (s *Signer) credentialScope(t time.Time) string {
	return fmt.Sprintf("%s/%s/%s/aws4_request",
		t.Format(shortTimeFormat),
		s.region,
		s.service,
	)
}

// canonicalQueryString creates the canonical query string.
//
// Query parameters are sorted by key, then by value within each key.
// Parameters are URL-encoded.
func (s *Signer) canonicalQueryString(query map[string][]string) string {
	if len(query) == 0 {
		return ""
	}

	// Sort keys
	keys := make([]string, 0, len(query))
	for k := range query {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Build query string
	var parts []string
	for _, k := range keys {
		values := query[k]
		sort.Strings(values)

		for _, v := range values {
			if v == "" {
				parts = append(parts, uriEncode(k)+"=")
			} else {
				parts = append(parts, uriEncode(k)+"="+uriEncode(v))
			}
		}
	}

	return strings.Join(parts, "&")
}

// canonicalHeaders creates the canonical headers string and signed headers list.
//
// Returns:
//   - canonicalHeaders: Lowercase, sorted headers with trimmed values
//   - signedHeaders: Semicolon-separated list of signed header names
//
// Headers are processed as:
//  1. Convert to lowercase
//  2. Trim values
//  3. Sort by header name
//  4. Format as "name:value\n"
func (s *Signer) canonicalHeaders(headers http.Header) (canonical, signed string) {
	// Collect and normalize headers
	headerMap := make(map[string]string)
	for k, v := range headers {
		if len(v) == 0 {
			continue
		}

		// Lowercase header name
		key := strings.ToLower(k)

		// Join multiple values with comma, trim spaces
		value := strings.TrimSpace(strings.Join(v, ","))

		headerMap[key] = value
	}

	// Sort header names
	keys := make([]string, 0, len(headerMap))
	for k := range headerMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Build canonical headers and signed headers
	var canonicalParts []string
	for _, k := range keys {
		canonicalParts = append(canonicalParts, fmt.Sprintf("%s:%s\n", k, headerMap[k]))
	}

	canonical = strings.Join(canonicalParts, "")
	signed = strings.Join(keys, ";")

	return canonical, signed
}

// hexEncodedHash returns the SHA256 hash of data as a hex string.
func (s *Signer) hexEncodedHash(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// hmacSHA256 computes the HMAC-SHA256 of data with the given key.
func (s *Signer) hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

// uriEncode encodes a string for use in URIs per AWS Signature Version 4 requirements.
//
// AWS URI encoding specification (RFC 3986 compliant):
//   - Unreserved characters (A-Z, a-z, 0-9, -, _, ., ~) are NOT encoded
//   - All other characters are percent-encoded as %HH where HH is uppercase hex
//   - Space is encoded as %20 (not +)
//   - Forward slash (/) in paths is NOT encoded (handled by caller)
//
// This implementation encodes UTF-8 byte-by-byte, which matches AWS behavior.
//
// Reference: https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
func uriEncode(s string) string {
	var result strings.Builder
	result.Grow(len(s))

	for i := 0; i < len(s); i++ {
		c := s[i]

		// Unreserved characters per RFC 3986: A-Z, a-z, 0-9, -, _, ., ~
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
			c == '-' || c == '_' || c == '.' || c == '~' {
			result.WriteByte(c)
			continue
		}

		// Percent-encode all other characters with uppercase hex (AWS requirement)
		result.WriteString(fmt.Sprintf("%%%02X", c))
	}

	return result.String()
}
