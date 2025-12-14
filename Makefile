.PHONY: test
test:
	go test -v -race -coverprofile=coverage.txt ./...

.PHONY: coverage
coverage: test
	go tool cover -html=coverage.txt -o coverage.html

.PHONY: lint
lint:
	gofmt -s -l .
	go vet ./...

.PHONY: fmt
fmt:
	gofmt -s -w .

.PHONY: build
build:
	go build -v ./...

.PHONY: clean
clean:
	rm -f coverage.txt coverage.html
	go clean

.PHONY: deps
deps:
	go mod download
	go mod tidy

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  test      - Run tests with race detector"
	@echo "  coverage  - Generate coverage report"
	@echo "  lint      - Run linters"
	@echo "  fmt       - Format code"
	@echo "  build     - Build all packages"
	@echo "  clean     - Clean build artifacts"
	@echo "  deps      - Download dependencies"
