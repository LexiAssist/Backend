package middleware

import (
	"net/http"
	"strconv"
	"strings"
	"time"
	
	"github.com/labstack/echo/v4"
	"go.uber.org/zap"
	
	"lexiassist/shared/pkg/logger"
	"lexiassist/shared/pkg/redis"
)

// RateLimitConfig holds rate limiting configuration.
type RateLimitConfig struct {
	RedisClient    *redis.Client
	DefaultRPM     int
	AIRPM          int
	AIPathPrefixes []string
}

// RateLimitMiddleware returns middleware for rate limiting.
func RateLimitMiddleware(config *RateLimitConfig) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			clientIP := c.RealIP()
			path := c.Request().URL.Path
			
			// Determine rate limit based on path
			isAIEndpoint := false
			for _, prefix := range config.AIPathPrefixes {
				if strings.HasPrefix(path, prefix) {
					isAIEndpoint = true
					break
				}
			}
			
			maxRequests := config.DefaultRPM
			if isAIEndpoint {
				maxRequests = config.AIRPM
			}
			
			// Check rate limit
			allowed, remaining, resetAt, err := config.RedisClient.CheckRateLimit(
				c.Request().Context(),
				clientIP,
				&redis.RateLimitConfig{
					Window:      time.Minute,
					MaxRequests: maxRequests,
					KeyPrefix:   "ratelimit:ip",
				},
			)
			
			if err != nil {
				logger.Error("rate limit check failed", zap.Error(err))
				// Allow request on error (fail open)
				return next(c)
			}
			
			// Set rate limit headers
			c.Response().Header().Set("X-RateLimit-Limit", strconv.Itoa(maxRequests))
			c.Response().Header().Set("X-RateLimit-Remaining", strconv.Itoa(remaining))
			c.Response().Header().Set("X-RateLimit-Reset", resetAt.Format(time.RFC3339))
			
			if !allowed {
				logger.Warn("rate limit exceeded",
					zap.String("ip", clientIP),
					zap.String("path", path),
					zap.Int("limit", maxRequests),
				)
				return echo.NewHTTPError(http.StatusTooManyRequests, "rate limit exceeded")
			}
			
			return next(c)
		}
	}
}
