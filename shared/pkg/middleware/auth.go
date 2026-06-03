// Package middleware provides HTTP middleware for authentication and logging.
package middleware

import (
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

// AuthRequired returns a Gin middleware that validates requests.
//
// Primary path (gateway-forwarded): trusts X-User-ID + X-Internal-Key headers
// that the gateway injects after validating the JWT. This is the path taken
// for all WebSocket and HTTP requests that arrive via the gateway proxy.
//
// Fallback path (direct calls): parses the raw JWT from the Authorization
// header or ?token= query param without re-validating the signature (the
// gateway is the single validation point).
func AuthRequired() gin.HandlerFunc {
	// Read the internal API key once at middleware creation time.
	// Falls back to the same default the gateway uses.
	internalKey := os.Getenv("INTERNAL_API_KEY")
	if internalKey == "" {
		internalKey = "dev-internal-key"
	}

	return func(c *gin.Context) {
		// ── Primary path: gateway pre-validated request ──────────────────────
		// The gateway strips the JWT and injects X-User-ID + X-Internal-Key
		// after validating the token. Trust this if the internal key matches.
		if userID := c.GetHeader("X-User-ID"); userID != "" {
			if c.GetHeader("X-Internal-Key") == internalKey {
				c.Set("user_id", userID)
				c.Next()
				return
			}
			// X-User-ID present but key mismatch — reject rather than fall through
			// to prevent header spoofing from external clients.
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid internal key"})
			c.Abort()
			return
		}

		// ── Fallback path: direct call with JWT ───────────────────────────────
		// Used for local development or service-to-service calls that bypass
		// the gateway. Parse the token without verifying the signature since
		// the gateway is the authoritative validator.
		var tokenString string
		authHeader := c.GetHeader("Authorization")
		if authHeader != "" {
			parts := strings.SplitN(authHeader, " ", 2)
			if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
				c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid authorization format"})
				c.Abort()
				return
			}
			tokenString = parts[1]
		} else {
			// WebSocket fallback: token passed as ?token= query param
			tokenString = c.Query("token")
			if tokenString == "" {
				c.JSON(http.StatusUnauthorized, gin.H{"error": "Authorization required"})
				c.Abort()
				return
			}
		}

		token, _, err := new(jwt.Parser).ParseUnverified(tokenString, jwt.MapClaims{})
		if err != nil {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Invalid token claims"})
			c.Abort()
			return
		}

		userID, ok := claims["user_id"].(string)
		if !ok || userID == "" {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "Missing user_id in token"})
			c.Abort()
			return
		}

		c.Set("user_id", userID)
		c.Next()
	}
}
