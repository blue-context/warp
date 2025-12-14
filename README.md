# Warp

[![CI](https://github.com/blue-context/warp/workflows/CI/badge.svg)](https://github.com/blue-context/warp/actions)
[![Go Report Card](https://goreportcard.com/badge/github.com/blue-context/warp)](https://goreportcard.com/report/github.com/blue-context/warp)
[![GoDoc](https://godoc.org/github.com/blue-context/warp?status.svg)](https://godoc.org/github.com/blue-context/warp)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Docs-blue)](https://deepwiki.com/blue-context/warp)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A unified Go SDK for calling multiple LLM providers (OpenAI, Anthropic, Azure, AWS Bedrock, Google Vertex AI, and more) with zero external dependencies.

## Features

- 🔌 **Unified Interface**: Call any LLM with the same API
- 🚀 **Zero Dependencies**: Only Go standard library
- 🔄 **Streaming Support**: Stream responses from any provider
- 💰 **Cost Tracking**: Track costs across different models
- 🔀 **Load Balancing**: Route requests across multiple deployments
- 🔁 **Retry & Fallback**: Automatic retries and fallback models
- 📊 **Observability**: Built-in logging and metrics

## Installation

```bash
go get github.com/blue-context/warp
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/blue-context/warp"
)

func main() {
    client, err := warp.NewClient(
        warp.WithAPIKey("openai", os.Getenv("OPENAI_API_KEY")),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    resp, err := client.Completion(context.Background(), &warp.CompletionRequest{
        Model: "openai/gpt-4",
        Messages: []warp.Message{
            {Role: "user", Content: "Hello, world!"},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Choices[0].Message.Content)
}
```

## Status

✅ **Production Ready** - v1.0 with 10 providers and complete API surface

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

Copyright 2025 Blue Context Inc.
