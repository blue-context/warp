# Contributing to Warp Go SDK

Thank you for your interest in contributing!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/blue-context/warp.git
   cd litellm-go
   ```

2. **Install Go 1.21+**
   ```bash
   go version  # Should be 1.21 or higher
   ```

3. **Install dependencies**
   ```bash
   go mod download
   ```

## Development Workflow

### Running Tests
```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run with race detector
go test -race ./...

# Run specific package
go test ./provider/openai/
```

### Code Style

- Run `gofmt -s -w .` before committing
- Run `go vet ./...` to check for issues
- Follow Go idioms and best practices
- Add GoDoc comments to all exported symbols

### Commit Messages

Use conventional commits:

```
feat(openai): add streaming support
fix(router): handle empty deployment list
docs(readme): update usage examples
test(provider): add integration tests
```

### Pull Request Process

1. Create a feature branch
2. Write tests for your changes
3. Ensure all tests pass
4. Update documentation
5. Submit PR with clear description

## Code Review

All code must be reviewed and approved before merging.

## Questions?

Open an issue or start a discussion!
