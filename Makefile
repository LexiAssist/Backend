# LexiAssist Go Backend Makefile

.PHONY: all build test clean docker-up docker-down run-user-service migrate-up

# Default target
all: build

# Build all services
build:
	@echo "Building User Service..."
	go build -o bin/user-service ./services/user/cmd/main.go

# Run tests
test:
	@echo "Running unit tests..."
	go test -v ./services/user/internal/...

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	go test -v -coverprofile=coverage.out ./services/user/internal/...
	go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -f coverage.out coverage.html

# Start infrastructure with Docker Compose
docker-up:
	@echo "Starting infrastructure..."
	docker-compose -f infra/docker-compose.yml up -d

# Stop infrastructure
docker-down:
	@echo "Stopping infrastructure..."
	docker-compose -f infra/docker-compose.yml down

# View logs
docker-logs:
	docker-compose -f infra/docker-compose.yml logs -f

# Run the user service locally (requires local postgres and redis)
run-user-service:
	@echo "Running User Service..."
	go run ./services/user/cmd/main.go

# Database migrations (requires golang-migrate installed)
migrate-up:
	@echo "Running database migrations..."
	migrate -path infra/migrations -database "${DATABASE_URL}" up

migrate-down:
	@echo "Rolling back database migrations..."
	migrate -path infra/migrations -database "${DATABASE_URL}" down

# Create a new migration
migrate-create:
	@read -p "Enter migration name: " name; \
	migrate create -ext sql -dir infra/migrations -seq $$name

# Install dependencies
deps:
	@echo "Installing dependencies..."
	go mod download
	go mod tidy

# Format code
fmt:
	@echo "Formatting code..."
	go fmt ./...

# Run linter
lint:
	@echo "Running linter..."
	golangci-lint run ./...

# Generate mocks (if using mockgen)
mocks:
	@echo "Generating mocks..."
	# go generate ./...

# Help
help:
	@echo "Available targets:"
	@echo "  build              - Build all services"
	@echo "  test               - Run unit tests"
	@echo "  test-coverage      - Run tests with coverage report"
	@echo "  clean              - Clean build artifacts"
	@echo "  docker-up          - Start infrastructure with Docker Compose"
	@echo "  docker-down        - Stop infrastructure"
	@echo "  docker-logs        - View Docker logs"
	@echo "  run-user-service   - Run User Service locally"
	@echo "  migrate-up         - Run database migrations"
	@echo "  migrate-down       - Rollback database migrations"
	@echo "  migrate-create     - Create a new migration"
	@echo "  deps               - Install dependencies"
	@echo "  fmt                - Format code"
	@echo "  lint               - Run linter"
