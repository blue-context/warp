package vertex

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a chat completion request to Vertex AI.
//
// This method implements the provider.Provider interface for non-streaming completions.
//
// The request is transformed from OpenAI format to Vertex AI format:
//   - messages -> contents (with role mapping)
//   - "assistant" role -> "model"
//   - temperature, maxTokens -> generationConfig
//   - tools -> functionDeclarations
//
// The response is transformed back from Vertex AI format to OpenAI-compatible format.
//
// Thread Safety: This method is safe for concurrent use.
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "gemini-pro",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	if req == nil {
		return nil, &warp.WarpError{
			Message:  "completion request cannot be nil",
			Provider: "vertex",
		}
	}

	// Get OAuth2 access token
	token, err := p.tokenProvider.GetToken()
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to get access token: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Build Vertex AI endpoint URL
	url := p.buildEndpoint(req.Model, false)

	// Transform request to Vertex AI format
	vertexReq, err := transformRequest(req)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to transform request: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Marshal request body
	body, err := json.Marshal(vertexReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to marshal request: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to create HTTP request: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token)

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("HTTP request failed: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}
	defer httpResp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to read response body: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Check HTTP status
	if httpResp.StatusCode != http.StatusOK {
		return nil, warp.ParseProviderError("vertex", httpResp.StatusCode, respBody, nil)
	}

	// Parse Vertex AI response
	var vertexResp vertexResponse
	if err := json.Unmarshal(respBody, &vertexResp); err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to parse response: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	// Check for prompt feedback (content blocked, safety issues, etc.)
	if vertexResp.PromptFeedback != nil && vertexResp.PromptFeedback.BlockReason != "" {
		return nil, warp.NewContentPolicyViolationError(
			fmt.Sprintf("prompt blocked: %s", vertexResp.PromptFeedback.BlockReason),
			"vertex",
			nil,
		)
	}

	// Check if we have any candidates
	if len(vertexResp.Candidates) == 0 {
		return nil, &warp.WarpError{
			Message:  "no completion candidates returned",
			Provider: "vertex",
		}
	}

	// Transform response to OpenAI format
	resp, err := transformResponse(&vertexResp, req.Model)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("failed to transform response: %v", err),
			Provider:      "vertex",
			OriginalError: err,
		}
	}

	return resp, nil
}
