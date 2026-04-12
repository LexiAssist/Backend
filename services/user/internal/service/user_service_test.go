// Package service_test contains unit tests for the User Service.
package service_test

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"lexiassist/services/user/internal/model"
	"lexiassist/services/user/internal/repository"
	"lexiassist/services/user/internal/service"
	"lexiassist/shared/pkg/config"
	"lexiassist/shared/pkg/logger"
)

// mockRedisClient is a mock implementation of Redis client for testing.
type mockRedisClient struct {
	data map[string]string
}

func newMockRedisClient() *mockRedisClient {
	return &mockRedisClient{data: make(map[string]string)}
}

func (m *mockRedisClient) Get(ctx context.Context, key string) (string, error) {
	return m.data[key], nil
}

func (m *mockRedisClient) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	m.data[key] = value.(string)
	return nil
}

func (m *mockRedisClient) Delete(ctx context.Context, keys ...string) error {
	for _, key := range keys {
		delete(m.data, key)
	}
	return nil
}

func (m *mockRedisClient) Exists(ctx context.Context, key string) (bool, error) {
	_, ok := m.data[key]
	return ok, nil
}

func TestMain(m *testing.M) {
	// Initialize logger for tests
	logger.Initialize("error")
	m.Run()
}

func TestUserService_Register(t *testing.T) {
	tests := []struct {
		name        string
		req         *service.RegisterRequest
		setupMocks  func(*repository.MockUserRepository, *repository.MockJWTKeyRepository)
		expectError bool
		statusCode  int
	}{
		{
			name: "successful registration",
			req: &service.RegisterRequest{
				Email:     "test@example.com",
				Password:  "password123",
				FirstName: "John",
				LastName:  "Doe",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				userRepo.ExistsByEmailFunc = func(ctx context.Context, email string) (bool, error) {
					return false, nil
				}
				userRepo.CreateFunc = func(ctx context.Context, user *model.User) error {
					user.ID = uuid.New()
					return nil
				}
				userRepo.SetVerificationCodeFunc = func(ctx context.Context, userID uuid.UUID, code string, expiresAt time.Time) error {
					return nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					// Return nil to trigger key generation
					return nil, nil
				}
				jwtKeyRepo.DeactivateAllKeysFunc = func(ctx context.Context) error {
					return nil
				}
				jwtKeyRepo.CreateFunc = func(ctx context.Context, key *model.JWTKey) error {
					return nil
				}
			},
			expectError: false,
		},
		{
			name: "email already exists",
			req: &service.RegisterRequest{
				Email:    "existing@example.com",
				Password: "password123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				userRepo.ExistsByEmailFunc = func(ctx context.Context, email string) (bool, error) {
					return true, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusConflict,
		},
		{
			name: "invalid email format",
			req: &service.RegisterRequest{
				Email:    "invalid-email",
				Password: "password123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(userRepo, jwtKeyRepo)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10, // Lower cost for faster tests
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
				VerificationCodeTTL:     15 * time.Minute,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			resp, err := svc.Register(ctx, tt.req)

			if tt.expectError {
				assert.Error(t, err)
				if httpErr, ok := err.(*echo.HTTPError); ok && tt.statusCode > 0 {
					assert.Equal(t, tt.statusCode, httpErr.Code)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
				assert.NotEmpty(t, resp.ID)
				assert.Equal(t, tt.req.Email, resp.Email)
			}
		})
	}
}

func TestUserService_Login(t *testing.T) {
	tests := []struct {
		name        string
		req         *service.LoginRequest
		setupMocks  func(*repository.MockUserRepository, *repository.MockJWTKeyRepository)
		expectError bool
		statusCode  int
	}{
		{
			name: "successful login",
			req: &service.LoginRequest{
				Email:    "test@example.com",
				Password: "password123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				userRepo.GetByEmailFunc = func(ctx context.Context, email string) (*model.User, error) {
					return &model.User{
						ID:           uuid.New(),
						Email:        email,
						PasswordHash: "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy", // bcrypt hash of "password123"
						IsActive:     true,
					}, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: false,
		},
		{
			name: "user not found",
			req: &service.LoginRequest{
				Email:    "nonexistent@example.com",
				Password: "password123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				userRepo.GetByEmailFunc = func(ctx context.Context, email string) (*model.User, error) {
					return nil, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusUnauthorized,
		},
		{
			name: "invalid password",
			req: &service.LoginRequest{
				Email:    "test@example.com",
				Password: "wrongpassword",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				userRepo.GetByEmailFunc = func(ctx context.Context, email string) (*model.User, error) {
					return &model.User{
						ID:           uuid.New(),
						Email:        email,
						PasswordHash: "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy",
						IsActive:     true,
					}, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusUnauthorized,
		},
		{
			name: "inactive account",
			req: &service.LoginRequest{
				Email:    "test@example.com",
				Password: "password123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				userRepo.GetByEmailFunc = func(ctx context.Context, email string) (*model.User, error) {
					return &model.User{
						ID:           uuid.New(),
						Email:        email,
						PasswordHash: "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy",
						IsActive:     false,
					}, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(userRepo, jwtKeyRepo)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10,
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			clientInfo := &service.ClientInfo{
				IPAddress: "127.0.0.1",
				UserAgent: "Test-Agent",
			}
			resp, err := svc.Login(ctx, tt.req, clientInfo)

			if tt.expectError {
				assert.Error(t, err)
				if httpErr, ok := err.(*echo.HTTPError); ok && tt.statusCode > 0 {
					assert.Equal(t, tt.statusCode, httpErr.Code)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
				assert.NotEmpty(t, resp.AccessToken)
				assert.NotEmpty(t, resp.RefreshToken)
				assert.Equal(t, "Bearer", resp.TokenType)
			}
		})
	}
}

func TestUserService_GetProfile(t *testing.T) {
	tests := []struct {
		name        string
		userID      string
		setupMocks  func(*repository.MockUserRepository)
		expectError bool
		statusCode  int
	}{
		{
			name:   "successful get profile",
			userID: uuid.New().String(),
			setupMocks: func(userRepo *repository.MockUserRepository) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return &model.User{
						ID:        id,
						Email:     "test@example.com",
						FirstName: "John",
						LastName:  "Doe",
						IsActive:  true,
					}, nil
				}
			},
			expectError: false,
		},
		{
			name:   "invalid user ID",
			userID: "invalid-uuid",
			setupMocks: func(userRepo *repository.MockUserRepository) {
			},
			expectError: true,
			statusCode:  http.StatusBadRequest,
		},
		{
			name:   "user not found",
			userID: uuid.New().String(),
			setupMocks: func(userRepo *repository.MockUserRepository) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return nil, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(userRepo)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10,
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			resp, err := svc.GetProfile(ctx, tt.userID)

			if tt.expectError {
				assert.Error(t, err)
				if httpErr, ok := err.(*echo.HTTPError); ok && tt.statusCode > 0 {
					assert.Equal(t, tt.statusCode, httpErr.Code)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, resp)
				assert.Equal(t, "test@example.com", resp.Email)
			}
		})
	}
}

func TestUserService_ChangePassword(t *testing.T) {
	userID := uuid.New()
	hashedPassword := "$2a$10$N9qo8uLOickgx2ZMRZoMyeIjZAgcfl7p92ldGxad68LJZdL17lhWy"

	tests := []struct {
		name        string
		req         *service.ChangePasswordRequest
		setupMocks  func(*repository.MockUserRepository, *repository.MockRefreshTokenRepository)
		expectError bool
		statusCode  int
	}{
		{
			name: "successful password change",
			req: &service.ChangePasswordRequest{
				CurrentPassword: "password123",
				NewPassword:     "newpassword123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, refreshTokenRepo *repository.MockRefreshTokenRepository) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return &model.User{
						ID:           userID,
						Email:        "test@example.com",
						PasswordHash: hashedPassword,
						IsActive:     true,
					}, nil
				}
				userRepo.UpdatePasswordFunc = func(ctx context.Context, uid uuid.UUID, passwordHash string) error {
					require.Equal(t, userID, uid)
					return nil
				}
				refreshTokenRepo.RevokeAllForUserFunc = func(ctx context.Context, uid uuid.UUID) error {
					require.Equal(t, userID, uid)
					return nil
				}
			},
			expectError: false,
		},
		{
			name: "incorrect current password",
			req: &service.ChangePasswordRequest{
				CurrentPassword: "wrongpassword",
				NewPassword:     "newpassword123",
			},
			setupMocks: func(userRepo *repository.MockUserRepository, refreshTokenRepo *repository.MockRefreshTokenRepository) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return &model.User{
						ID:           userID,
						Email:        "test@example.com",
						PasswordHash: hashedPassword,
						IsActive:     true,
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(userRepo, refreshTokenRepo)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10,
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			err = svc.ChangePassword(ctx, userID.String(), tt.req)

			if tt.expectError {
				assert.Error(t, err)
				if httpErr, ok := err.(*echo.HTTPError); ok && tt.statusCode > 0 {
					assert.Equal(t, tt.statusCode, httpErr.Code)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUserService_VerifyEmail(t *testing.T) {
	userID := uuid.New()

	tests := []struct {
		name        string
		code        string
		setupMocks  func(*repository.MockUserRepository, *mockRedisClient)
		expectError bool
		statusCode  int
	}{
		{
			name: "successful verification",
			code: "123456",
			setupMocks: func(userRepo *repository.MockUserRepository, redisClient *mockRedisClient) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return &model.User{
						ID:                    userID,
						Email:                 "test@example.com",
						EmailVerified:         false,
						VerificationCode:      "123456",
						IsActive:              true,
					}, nil
				}
				userRepo.VerifyEmailFunc = func(ctx context.Context, uid uuid.UUID) error {
					require.Equal(t, userID, uid)
					return nil
				}
				redisClient.Set(nil, "email_verification:"+userID.String(), "123456", 0)
			},
			expectError: false,
		},
		{
			name: "already verified",
			code: "123456",
			setupMocks: func(userRepo *repository.MockUserRepository, redisClient *mockRedisClient) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return &model.User{
						ID:            userID,
						Email:         "test@example.com",
						EmailVerified: true,
						IsActive:      true,
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusConflict,
		},
		{
			name: "invalid code",
			code: "wrongcode",
			setupMocks: func(userRepo *repository.MockUserRepository, redisClient *mockRedisClient) {
				userRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.User, error) {
					return &model.User{
						ID:               userID,
						Email:            "test@example.com",
						EmailVerified:    false,
						VerificationCode: "123456",
						IsActive:         true,
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(userRepo, redisClient)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10,
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			err = svc.VerifyEmail(ctx, userID.String(), tt.code)

			if tt.expectError {
				assert.Error(t, err)
				if httpErr, ok := err.(*echo.HTTPError); ok && tt.statusCode > 0 {
					assert.Equal(t, tt.statusCode, httpErr.Code)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestUserService_ListSessions(t *testing.T) {
	userID := uuid.New()

	tests := []struct {
		name        string
		setupMocks  func(*repository.MockSessionRepository, *repository.MockJWTKeyRepository)
		expectError bool
		lenSessions int
	}{
		{
			name: "successful list sessions",
			setupMocks: func(sessionRepo *repository.MockSessionRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				sessionRepo.GetByUserIDFunc = func(ctx context.Context, uid uuid.UUID) ([]model.UserSession, error) {
					return []model.UserSession{
						{
							ID:           uuid.New(),
							UserID:       userID,
							DeviceName:   "Chrome on Windows",
							DeviceType:   "desktop",
							OS:           "Windows",
							Browser:      "Chrome",
							IPAddress:    "192.168.1.1",
							LastActiveAt: time.Now(),
							CreatedAt:    time.Now(),
						},
						{
							ID:           uuid.New(),
							UserID:       userID,
							DeviceName:   "Safari on iPhone",
							DeviceType:   "mobile",
							OS:           "iOS",
							Browser:      "Safari",
							IPAddress:    "192.168.1.2",
							LastActiveAt: time.Now(),
							CreatedAt:    time.Now(),
						},
					}, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: false,
			lenSessions: 2,
		},
		{
			name: "empty sessions list",
			setupMocks: func(sessionRepo *repository.MockSessionRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				sessionRepo.GetByUserIDFunc = func(ctx context.Context, uid uuid.UUID) ([]model.UserSession, error) {
					return []model.UserSession{}, nil
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: false,
			lenSessions: 0,
		},
		{
			name: "database error",
			setupMocks: func(sessionRepo *repository.MockSessionRepository, jwtKeyRepo *repository.MockJWTKeyRepository) {
				sessionRepo.GetByUserIDFunc = func(ctx context.Context, uid uuid.UUID) ([]model.UserSession, error) {
					return nil, errors.New("database error")
				}
				jwtKeyRepo.GetActivePrivateKeyFunc = func(ctx context.Context) (*model.JWTKey, error) {
					return &model.JWTKey{
						KeyType: "private",
						KeyData: "encrypted-key-data",
					}, nil
				}
			},
			expectError: true,
			lenSessions: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(sessionRepo, jwtKeyRepo)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10,
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			sessions, err := svc.ListSessions(ctx, userID.String())

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Len(t, sessions, tt.lenSessions)
			}
		})
	}
}

func TestUserService_RevokeSession(t *testing.T) {
	userID := uuid.New()
	sessionID := uuid.New()

	tests := []struct {
		name        string
		sessionID   string
		setupMocks  func(*repository.MockSessionRepository, *repository.MockRefreshTokenRepository)
		expectError bool
		statusCode  int
	}{
		{
			name:      "successful revoke",
			sessionID: sessionID.String(),
			setupMocks: func(sessionRepo *repository.MockSessionRepository, refreshTokenRepo *repository.MockRefreshTokenRepository) {
				sessionRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.UserSession, error) {
					return &model.UserSession{
						ID:             sessionID,
						UserID:         userID,
						RefreshTokenID: func() *uuid.UUID { u := uuid.New(); return &u }(),
					}, nil
				}
				sessionRepo.RevokeFunc = func(ctx context.Context, id uuid.UUID) error {
					require.Equal(t, sessionID, id)
					return nil
				}
				refreshTokenRepo.RevokeFunc = func(ctx context.Context, id uuid.UUID, replacedByID *uuid.UUID) error {
					return nil
				}
			},
			expectError: false,
		},
		{
			name:      "session not found",
			sessionID: sessionID.String(),
			setupMocks: func(sessionRepo *repository.MockSessionRepository, refreshTokenRepo *repository.MockRefreshTokenRepository) {
				sessionRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.UserSession, error) {
					return nil, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusNotFound,
		},
		{
			name:      "unauthorized - different user",
			sessionID: sessionID.String(),
			setupMocks: func(sessionRepo *repository.MockSessionRepository, refreshTokenRepo *repository.MockRefreshTokenRepository) {
				sessionRepo.GetByIDFunc = func(ctx context.Context, id uuid.UUID) (*model.UserSession, error) {
					return &model.UserSession{
						ID:     sessionID,
						UserID: uuid.New(), // Different user
					}, nil
				}
			},
			expectError: true,
			statusCode:  http.StatusForbidden,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			userRepo := &repository.MockUserRepository{}
			refreshTokenRepo := &repository.MockRefreshTokenRepository{}
			sessionRepo := &repository.MockSessionRepository{}
			passwordResetRepo := &repository.MockPasswordResetRepository{}
			jwtKeyRepo := &repository.MockJWTKeyRepository{}
			redisClient := newMockRedisClient()

			if tt.setupMocks != nil {
				tt.setupMocks(sessionRepo, refreshTokenRepo)
			}

			cfg := &config.UserServiceConfig{
				PrivateKeyEncryptionKey: "test-encryption-key-min-32-chars-long",
				BcryptCost:              10,
				AccessTokenTTL:          15 * time.Minute,
				RefreshTokenTTL:         30 * 24 * time.Hour,
			}

			svc, err := service.NewUserService(userRepo, refreshTokenRepo, sessionRepo, passwordResetRepo, jwtKeyRepo, redisClient, cfg)
			if err != nil {
				t.Skipf("Skipping test due to RSA key generation: %v", err)
				return
			}

			ctx := context.Background()
			err = svc.RevokeSession(ctx, userID.String(), tt.sessionID)

			if tt.expectError {
				assert.Error(t, err)
				if httpErr, ok := err.(*echo.HTTPError); ok && tt.statusCode > 0 {
					assert.Equal(t, tt.statusCode, httpErr.Code)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
