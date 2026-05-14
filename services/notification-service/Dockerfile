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
COPY services/notification-service/ ./services/notification-service/

# Build the application using vendor
RUN CGO_ENABLED=0 GOOS=linux go build -mod=vendor -a -installsuffix cgo -o notification-service ./services/notification-service/

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates
WORKDIR /root/

# Copy the binary
COPY --from=builder /app/notification-service .

EXPOSE 8084
CMD ["./notification-service"]
