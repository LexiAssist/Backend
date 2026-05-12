// Package config provides Gateway-specific configuration.
package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds Gateway configuration.
type Config struct {
	Port                    string
	Environment             string
	LogLevel                string
	
	// Redis
	RedisURL                string
	
	// Rate Limiting
	RateLimitRPM            int
	AIRateLimitRPM          int
	AIDailyQuota            int
	
	// Circuit Breaker
	CircuitBreakerThreshold int
	CircuitBreakerTimeout   time.Duration
	
	// CORS
	AllowedOrigins          []string
	
	// Go Upstream Services
	UserServiceURL          string
	ContentServiceURL       string
	AnalyticsServiceURL     string
	NotificationServiceURL  string
	SyncServiceURL          string
	
	// Python AI Services
	AIOrchestratorURL       string
	RetrievalServiceURL     string
	AudioServiceURL         string
	IngestionServiceURL     string
	
	// New AI Service (FastAPI monolith)
	AIServiceURL            string
	AIServiceTimeout        time.Duration
	
	// Security
	InternalAPIKey          string
}

// Load loads configuration from environment variables.
func Load() (*Config, error) {
	cfg := &Config{
		Port:                    getEnv("PORT", "8080"),
		Environment:             getEnv("ENVIRONMENT", "development"),
		LogLevel:                getEnv("LOG_LEVEL", "info"),
		RedisURL:                getEnv("REDIS_URL", "localhost:6379"),
		RateLimitRPM:            getEnvInt("RATE_LIMIT_RPM", 100),
		AIRateLimitRPM:          getEnvInt("AI_RATE_LIMIT_RPM", 100),	// Increased from 20 for development
		AIDailyQuota:            getEnvInt("AI_DAILY_QUOTA", 50),
		CircuitBreakerThreshold: getEnvInt("CIRCUIT_BREAKER_THRESHOLD", 3),
		CircuitBreakerTimeout:   getEnvDuration("CIRCUIT_BREAKER_TIMEOUT", 60*time.Second),
		AllowedOrigins:          getEnvSlice("ALLOWED_ORIGINS", []string{"http://localhost:3000"}),
		UserServiceURL:          getEnv("USER_SERVICE_URL", "http://localhost:8081"),
		ContentServiceURL:       getEnv("CONTENT_SERVICE_URL", "http://localhost:8082"),
		AnalyticsServiceURL:     getEnv("ANALYTICS_SERVICE_URL", "http://localhost:8083"),
		NotificationServiceURL:  getEnv("NOTIFICATION_SERVICE_URL", "http://localhost:8084"),
		SyncServiceURL:          getEnv("SYNC_SERVICE_URL", "http://localhost:8085"),
		AIOrchestratorURL:       getEnv("AI_ORCHESTRATOR_URL", "http://localhost:5005"),
		RetrievalServiceURL:     getEnv("RETRIEVAL_SERVICE_URL", "http://localhost:5003"),
		AudioServiceURL:         getEnv("AUDIO_SERVICE_URL", "http://localhost:5004"),
		IngestionServiceURL:     getEnv("INGESTION_SERVICE_URL", "http://localhost:5002"),
		AIServiceURL:            getEnv("AI_SERVICE_URL", "http://localhost:8000"),
		AIServiceTimeout:        getEnvDuration("AI_SERVICE_TIMEOUT", 120*time.Second),
		InternalAPIKey:          getEnv("INTERNAL_API_KEY", "dev-internal-key-change-in-production"),
	}
	
	return cfg, nil
}

// GetServiceURL returns the URL for a given service name.
func (c *Config) GetServiceURL(serviceName string) string {
	switch serviceName {
	case "user":
		return c.UserServiceURL
	case "content":
		return c.ContentServiceURL
	case "analytics":
		return c.AnalyticsServiceURL
	case "notification":
		return c.NotificationServiceURL
	case "sync":
		return c.SyncServiceURL
	case "ai", "orchestrator":
		return c.AIOrchestratorURL
	case "retrieval":
		return c.RetrievalServiceURL
	case "audio":
		return c.AudioServiceURL
	case "ingestion":
		return c.IngestionServiceURL
	default:
		return ""
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}

func getEnvSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		return strings.Split(value, ",")
	}
	return defaultValue
}
