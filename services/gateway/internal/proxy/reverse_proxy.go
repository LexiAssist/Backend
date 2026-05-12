package proxy

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	
	"github.com/labstack/echo/v4"
	"go.uber.org/zap"
	
	"lexiassist/shared/pkg/logger"
)

// ReverseProxy handles proxying requests to upstream services.
type ReverseProxy struct {
	client            *http.Client
	circuitBreakers   map[string]*CircuitBreaker
	defaultCBThreshold int
	defaultCBTimeout   time.Duration
	internalAPIKey    string
}

// NewReverseProxy creates a new reverse proxy.
func NewReverseProxy(cbThreshold int, cbTimeout time.Duration, internalAPIKey string) *ReverseProxy {
	return &ReverseProxy{
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		circuitBreakers:    make(map[string]*CircuitBreaker),
		defaultCBThreshold: cbThreshold,
		defaultCBTimeout:   cbTimeout,
		internalAPIKey:     internalAPIKey,
	}
}

// ProxyRequest proxies a request to the target service.
func (p *ReverseProxy) ProxyRequest(c echo.Context, targetURL string, serviceName string, injectUserID bool) error {
	ctx := c.Request().Context()
	
	// Get or create circuit breaker for this service
	cb := p.getCircuitBreaker(serviceName)
	
	// Execute with circuit breaker protection
	err := cb.Execute(func() error {
		return p.doProxyRequest(ctx, c, targetURL, injectUserID)
	})
	
	if err == ErrCircuitOpen {
		logger.Warn("circuit breaker open, rejecting request",
			zap.String("service", serviceName),
		)
		return echo.NewHTTPError(http.StatusServiceUnavailable, "service temporarily unavailable")
	}
	
	return err
}

// doProxyRequest performs the actual proxying.
func (p *ReverseProxy) doProxyRequest(ctx context.Context, c echo.Context, targetURL string, injectUserID bool) error {
	// Parse target URL
	target, err := url.Parse(targetURL)
	if err != nil {
		return fmt.Errorf("invalid target URL: %w", err)
	}
	
	// Create new request - must clear RequestURI as it's not allowed in client requests
	req := c.Request().Clone(ctx)
	req.RequestURI = "" // Clear RequestURI - only valid for incoming server requests
	
	// Update URL to point to target
	req.URL.Scheme = target.Scheme
	req.URL.Host = target.Host
	req.URL.Path = target.Path + req.URL.Path
	req.Host = target.Host
	
	// Remove hop-by-hop headers but PRESERVE Content-Type for multipart/form-data
	contentType := req.Header.Get("Content-Type")
	removeHopByHopHeaders(req.Header)
	
	// Restore Content-Type if it was multipart/form-data (needed for file uploads)
	if strings.Contains(contentType, "multipart/form-data") {
		req.Header.Set("Content-Type", contentType)
	}
	
	// Inject X-User-ID header if authenticated
	if injectUserID {
		if userID := c.Get("user_id"); userID != nil {
			req.Header.Set("X-User-ID", userID.(string))
		}
	}
	
	// Forward correlation ID
	if correlationID := req.Header.Get("X-Correlation-ID"); correlationID != "" {
		req.Header.Set("X-Correlation-ID", correlationID)
	}
	
	// Inject internal API key so upstream services can validate the request came from the gateway
	req.Header.Set("X-Internal-Key", p.internalAPIKey)
	
	// Execute request
	resp, err := p.client.Do(req)
	if err != nil {
		return fmt.Errorf("proxy request failed: %w", err)
	}
	defer resp.Body.Close()
	
	// Copy headers from response
	for key, values := range resp.Header {
		for _, value := range values {
			c.Response().Header().Add(key, value)
		}
	}
	
	// Set status code
	c.Response().WriteHeader(resp.StatusCode)
	
	// Copy body
	_, err = io.Copy(c.Response().Writer, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to copy response body: %w", err)
	}
	
	return nil
}

// getCircuitBreaker gets or creates a circuit breaker for a service.
func (p *ReverseProxy) getCircuitBreaker(serviceName string) *CircuitBreaker {
	if cb, exists := p.circuitBreakers[serviceName]; exists {
		return cb
	}
	
	cb := NewCircuitBreaker(serviceName, p.defaultCBThreshold, p.defaultCBTimeout)
	p.circuitBreakers[serviceName] = cb
	return cb
}

// removeHopByHopHeaders removes headers that should not be forwarded.
func removeHopByHopHeaders(header http.Header) {
	hopByHop := []string{
		"Connection",
		"Keep-Alive",
		"Proxy-Authenticate",
		"Proxy-Authorization",
		"Te",
		"Trailers",
		"Transfer-Encoding",
		"Upgrade",
	}
	
	for _, h := range hopByHop {
		header.Del(h)
	}
}

// HealthCheck checks if a service is healthy.
func (p *ReverseProxy) HealthCheck(serviceURL string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, serviceURL+"/health", nil)
	if err != nil {
		return err
	}
	
	resp, err := p.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}
	
	return nil
}
