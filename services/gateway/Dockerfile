# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Install git and ca-certificates
RUN apk add --no-cache git ca-certificates

# Copy go.mod and go.sum
COPY go.mod go.sum ./

# Copy vendor directory
COPY vendor/ ./vendor/

# Copy source code
COPY services/gateway/ ./services/gateway/
COPY shared/ ./shared/

# Build the application using vendor
RUN CGO_ENABLED=0 GOOS=linux go build -mod=vendor -a -installsuffix cgo -o gateway ./services/gateway/cmd/main.go

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

COPY --from=builder /app/gateway .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

CMD ["./gateway"]
