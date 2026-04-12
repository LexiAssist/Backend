# Build stage
FROM golang:1.24-alpine AS builder

WORKDIR /app

# Install git and ca-certificates
RUN apk add --no-cache git ca-certificates

# Copy go.mod, go.sum and vendor directory
COPY go.mod go.sum ./
COPY vendor/ ./vendor/

# Copy shared packages
COPY shared/ ./shared/

# Copy source code
COPY services/sync-service/ ./services/sync-service/

# Build the application using vendor
RUN CGO_ENABLED=0 GOOS=linux go build -mod=mod -a -installsuffix cgo -o sync-service ./services/sync-service/

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates
WORKDIR /root/

# Copy the binary
COPY --from=builder /app/sync-service .

EXPOSE 8085
CMD ["./sync-service"]
