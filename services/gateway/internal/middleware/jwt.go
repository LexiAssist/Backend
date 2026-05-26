// Package middleware provides HTTP middleware for the Gateway.
package middleware

import (
	"crypto/rsa"
	"errors"
	"fmt"
	"net/http"
	"strings"
	
	"github.com/golang-jwt/jwt/v5"
	"github.com/labstack/echo/v4"
	"go.uber.org/zap"
	
	"lexiassist/shared/pkg/logger"
)

// JWTConfig holds JWT middleware configuration.
type JWTConfig struct {
	PublicKey *rsa.PublicKey
	Skipper   func(c echo.Context) bool
}

// JWTValidator validates JWT tokens.
type JWTValidator struct {
	publicKey *rsa.PublicKey
}

// Claims represents JWT claims.
type Claims struct {
	UserID    string `json:"user_id"`
	Email     string `json:"email"`
	TokenType string `json:"token_type"`
	jwt.RegisteredClaims
}

// NewJWTValidator creates a new JWT validator.
func NewJWTValidator(publicKey *rsa.PublicKey) *JWTValidator {
	return &JWTValidator{publicKey: publicKey}
}

// Validate validates a JWT token and returns claims.
func (v *JWTValidator) Validate(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return v.publicKey, nil
	})
	
	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, errors.New("token expired")
		}
		return nil, errors.New("invalid token")
	}
	
	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return nil, errors.New("invalid token")
	}
	
	// Ensure it's an access token
	if claims.TokenType != "access" {
		return nil, errors.New("invalid token type")
	}
	
	return claims, nil
}

// JWTMiddleware returns Echo middleware for JWT validation.
func JWTMiddleware(validator *JWTValidator, skipper func(echo.Context) bool) echo.MiddlewareFunc {
	return func(next echo.HandlerFunc) echo.HandlerFunc {
		return func(c echo.Context) error {
			// Skip if skipper returns true
			if skipper != nil && skipper(c) {
				return next(c)
			}
			
			// Extract token from Authorization header or query param (needed for WebSocket)
			var tokenString string
			authHeader := c.Request().Header.Get("Authorization")
			if authHeader != "" {
				// Parse Bearer token
				parts := strings.SplitN(authHeader, " ", 2)
				if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
					return echo.NewHTTPError(http.StatusUnauthorized, "invalid authorization header format")
				}
				tokenString = parts[1]
			} else {
				// Fallback to query param for WebSocket auth
				tokenString = c.QueryParam("token")
				if tokenString == "" {
					return echo.NewHTTPError(http.StatusUnauthorized, "missing authorization header")
				}
			}
			
			// Validate token
			claims, err := validator.Validate(tokenString)
			if err != nil {
				logger.Warn("token validation failed",
					zap.String("error", err.Error()),
					zap.String("path", c.Request().URL.Path),
				)
				return echo.NewHTTPError(http.StatusUnauthorized, err.Error())
			}
			
			// Store claims in context
			c.Set("user_id", claims.UserID)
			c.Set("email", claims.Email)
			c.Set("claims", claims)
			
			return next(c)
		}
	}
}

// PublicKeySkipper returns a skipper that skips JWT validation for public paths.
func PublicKeySkipper(publicPaths []string) func(echo.Context) bool {
	return func(c echo.Context) bool {
		path := c.Request().URL.Path
		for _, publicPath := range publicPaths {
			if strings.HasPrefix(path, publicPath) {
				return true
			}
		}
		return false
	}
}
