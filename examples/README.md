# Warp Go SDK Examples

This directory contains comprehensive examples demonstrating the features and capabilities of the Warp Go SDK.

## Prerequisites

Before running any examples, you'll need API keys for the providers you want to use. Set them as environment variables:

```bash
export OPENAI_API_KEY=sk-...
```

For other providers, use their respective environment variables:
- Anthropic: `ANTHROPIC_API_KEY`
- Azure: `AZURE_API_KEY`
- AWS Bedrock: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`
- Google Vertex AI: `VERTEX_PROJECT`, `VERTEX_LOCATION`

## Examples Overview

### 1. Basic Usage (`examples/basic`)

**What it demonstrates:**
- Creating a provider and client
- Registering a provider with the client
- Sending a simple completion request
- Handling errors properly
- Printing response and token usage

**To run:**
```bash
cd examples/basic
go run main.go
```

**Expected output:**
```
Response: The capital of France is Paris.
Tokens used: 25 (prompt: 18, completion: 7)
```

### 2. Streaming (`examples/streaming`)

**What it demonstrates:**
- Streaming completion requests
- Processing chunks as they arrive
- Real-time output display
- Handling stream completion (io.EOF)
- Proper stream cleanup with defer

**To run:**
```bash
cd examples/streaming
go run main.go
```

**Expected output:**
```
Streaming response:
---
Code flows like water,
Concurrency made simple,
Go routines unite.
---
Stream completed
```

### 3. Embeddings (`examples/embedding`)

**What it demonstrates:**
- Generating single text embeddings
- Batch embedding multiple texts
- Accessing embedding dimensions
- Processing embedding vectors
- Token usage for embeddings

**To run:**
```bash
cd examples/embedding
go run main.go
```

**Expected output:**
```
Example 1: Single text embedding
---
Embedding dimensions: 1536
First 5 values: [0.0023, -0.0084, 0.0156, -0.0034, 0.0091]
Tokens used: 10

Example 2: Batch embeddings
---
Generated 3 embeddings
Embedding 1: 1536 dimensions
Embedding 2: 1536 dimensions
Embedding 3: 1536 dimensions
Total tokens used: 12

Embeddings completed successfully
```

### 4. Advanced Features (`examples/advanced`)

**What it demonstrates:**
- Client configuration options (timeout, retries, debug mode)
- Context-based timeout control
- Request-level timeout configuration
- Error type handling and differentiation
- Multimodal message structure (text + images)
- Best practices for error handling

**To run:**
```bash
cd examples/advanced
go run main.go
```

**Error types covered:**
- `RateLimitError` - Rate limiting with retry-after
- `AuthenticationError` - Invalid API keys
- `TimeoutError` - Request timeouts
- `ContextWindowExceededError` - Token limits
- `APIError` - Generic API errors

## Common Patterns

### Error Handling

All examples demonstrate proper error handling patterns:

```go
resp, err := client.Completion(ctx, req)
if err != nil {
    var rateLimitErr *warp.RateLimitError
    if errors.As(err, &rateLimitErr) {
        // Handle rate limit
        fmt.Printf("Rate limited! Retry after: %v\n", rateLimitErr.RetryAfter)
    }
    // Handle other error types...
}
```

### Resource Cleanup

Always use `defer` for cleanup:

```go
client, err := warp.NewClient()
if err != nil {
    log.Fatal(err)
}
defer client.Close()

stream, err := client.CompletionStream(ctx, req)
if err != nil {
    log.Fatal(err)
}
defer stream.Close()
```

### Context Usage

Use context for cancellation and timeouts:

```go
// Create context with timeout
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

// Use context in request
resp, err := client.Completion(ctx, req)
```

## Building Examples

To build all examples without running them:

```bash
# From the repository root
go build ./examples/basic/
go build ./examples/streaming/
go build ./examples/embedding/
go build ./examples/advanced/

# Or build all at once
go build ./examples/...
```

## Running Tests

The examples directory doesn't include tests, but you can test the SDK itself:

```bash
# From the repository root
go test ./...

# With race detection
go test -race ./...

# With verbose output
go test -v ./...
```

## Troubleshooting

### API Key Not Set
```
OPENAI_API_KEY environment variable is required
```
**Solution:** Set the required environment variable before running the example.

### Provider Not Registered
```
provider "openai" not found
```
**Solution:** Ensure you call `client.RegisterProvider(provider)` before using the client.

### Import Errors
```
package github.com/blue-context/warp: cannot find package
```
**Solution:** Run `go mod tidy` to download dependencies.

## Next Steps

After running these examples, you can:

1. **Explore the SDK documentation** - Check the main [README](../README.md) for detailed API documentation
2. **Try different providers** - The SDK supports 100+ LLM providers
3. **Customize configuration** - Experiment with different timeout, retry, and fallback settings
4. **Build your application** - Use these examples as a starting point for your own projects

## Additional Resources

- [Warp Documentation](https://docs.warp.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Go Context Package](https://pkg.go.dev/context)
- [Go Error Handling](https://go.dev/blog/error-handling-and-go)

## Contributing

Found an issue or have a suggestion? Please open an issue or submit a pull request on GitHub.

---

**Happy coding with Warp Go SDK!**
