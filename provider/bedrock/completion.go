package bedrock

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/blue-context/warp"
)

// Completion sends a completion request to AWS Bedrock.
//
// The request is automatically transformed based on the model family:
//   - Claude models use Anthropic's message format
//   - Llama models use Meta's prompt format
//   - Titan models use Amazon's format
//   - Cohere models use Cohere's format
//
// The response is automatically parsed and normalized to Warp's standard format.
//
// Example:
//
//	resp, err := provider.Completion(ctx, &warp.CompletionRequest{
//	    Model: "claude-3-opus",
//	    Messages: []warp.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	})
func (p *Provider) Completion(ctx context.Context, req *warp.CompletionRequest) (*warp.CompletionResponse, error) {
	if req == nil {
		return nil, &warp.WarpError{
			Message:  "request cannot be nil",
			Provider: "bedrock",
		}
	}

	if req.Model == "" {
		return nil, &warp.WarpError{
			Message:  "model is required",
			Provider: "bedrock",
		}
	}

	// Get Bedrock model ID
	modelID, err := getModelID(req.Model)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       fmt.Sprintf("invalid model: %v", err),
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Determine model family
	family := detectModelFamily(modelID)
	if family == familyUnknown {
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("unsupported model family for model: %s", modelID),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}

	// Transform request based on model family
	var bedrockReq map[string]interface{}
	switch family {
	case familyClaude:
		bedrockReq = transformClaudeRequest(req)
	case familyLlama:
		bedrockReq = transformLlamaRequest(req)
	case familyTitan:
		bedrockReq = transformTitanRequest(req)
	case familyCohere:
		bedrockReq = transformCohereRequest(req)
	default:
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("unsupported model family: %s", family),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}

	// Marshal request body
	body, err := json.Marshal(bedrockReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to marshal request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Build endpoint URL
	endpoint := p.buildEndpoint(modelID, false)

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to create HTTP request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	// Sign request with AWS Signature V4 (session token is included by signer)
	if err := p.signer.SignRequest(httpReq, body); err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to sign request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Send request
	httpResp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to send request",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}
	defer httpResp.Body.Close()

	// Check for errors
	if httpResp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(httpResp.Body)
		return nil, warp.ParseProviderError("bedrock", httpResp.StatusCode, bodyBytes, nil)
	}

	// Parse response based on model family
	var resp *warp.CompletionResponse
	switch family {
	case familyClaude:
		resp, err = parseClaudeResponse(httpResp.Body)
	case familyLlama:
		resp, err = parseLlamaResponse(httpResp.Body)
	case familyTitan:
		resp, err = parseTitanResponse(httpResp.Body)
	case familyCohere:
		resp, err = parseCohereResponse(httpResp.Body)
	default:
		return nil, &warp.WarpError{
			Message:  fmt.Sprintf("unsupported model family for parsing: %s", family),
			Provider: "bedrock",
			Model:    req.Model,
		}
	}

	if err != nil {
		return nil, &warp.WarpError{
			Message:       "failed to parse response",
			Provider:      "bedrock",
			Model:         req.Model,
			OriginalError: err,
		}
	}

	// Set model in response
	resp.Model = req.Model

	return resp, nil
}
