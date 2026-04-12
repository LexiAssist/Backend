// Package service contains business logic for the User Service.
package service

import (
	"context"
	"crypto/rsa"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/mail"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v4"

	"lexiassist/services/user/internal/model"
	"lexiassist/services/user/internal/repository"
	"lexiassist/shared/pkg/auth"
	"lexiassist/shared/pkg/config"
	"lexiassist/shared/pkg/logger"
	"lexiassist/shared/pkg/redis"
)

var (
	ErrUserNotFound           = errors.New("user not found")
	ErrEmailAlreadyExists     = errors.New("email already exists")
	ErrInvalidCredentials     = errors.New("invalid email or password")
	ErrInvalidRefreshToken    = errors.New("invalid or expired refresh token")
	ErrTokenRevoked           = errors.New("token has been revoked")
	ErrInvalidVerificationCode = errors.New("invalid or expired verification code")
	ErrEmailAlreadyVerified   = errors.New("email already verified")
	ErrInvalidPassword        = errors.New("invalid password")
	ErrSessionNotFound        = errors.New("session not found")
	ErrUnauthorized           = errors.New("unauthorized")
)

// UserService defines the interface for user business logic.
type UserService interface {
	Register(ctx context.Context, req *RegisterRequest) (*UserResponse, error)
	VerifyEmail(ctx context.Context, userID string, code string) error
	ResendVerification(ctx context.Context, userID string) error
	Login(ctx context.Context, req *LoginRequest, clientInfo *ClientInfo) (*TokenResponse, error)
	RefreshToken(ctx context.Context, refreshToken string, clientInfo *ClientInfo) (*TokenResponse, error)
	Logout(ctx context.Context, userID string, accessTokenJTI string) error
	LogoutAll(ctx context.Context, userID string) error
	GetProfile(ctx context.Context, userID string) (*UserResponse, error)
	UpdateProfile(ctx context.Context, userID string, req *UpdateProfileRequest) (*UserResponse, error)
	ChangePassword(ctx context.Context, userID string, req *ChangePasswordRequest) error
	RequestPasswordReset(ctx context.Context, email string) error
	ResetPassword(ctx context.Context, token string, newPassword string) error
	ListSessions(ctx context.Context, userID string) ([]SessionResponse, error)
	RevokeSession(ctx context.Context, userID string, sessionID string) error
	GetPublicKey(ctx context.Context) (string, error)
}

// ClientInfo holds information about the client making the request.
type ClientInfo struct {
	IPAddress string
	UserAgent string
}

// RegisterRequest represents a registration request.
type RegisterRequest struct {
	Email         string `json:"email" validate:"required,email"`
	Password      string `json:"password" validate:"required,min=8"`
	FirstName     string `json:"first_name"`
	LastName      string `json:"last_name"`
	School        string `json:"school"`
	Department    string `json:"department"`
	AcademicLevel string `json:"academic_level"`
}

// LoginRequest represents a login request.
type LoginRequest struct {
	Email    string `json:"email" validate:"required,email"`
	Password string `json:"password" validate:"required"`
}

// UpdateProfileRequest represents a profile update request.
type UpdateProfileRequest struct {
	FirstName     string `json:"first_name,omitempty"`
	LastName      string `json:"last_name,omitempty"`
	School        string `json:"school,omitempty"`
	Department    string `json:"department,omitempty"`
	AcademicLevel string `json:"academic_level,omitempty"`
	Timezone      string `json:"timezone,omitempty"`
}

// ChangePasswordRequest represents a password change request.
type ChangePasswordRequest struct {
	CurrentPassword string `json:"current_password" validate:"required"`
	NewPassword     string `json:"new_password" validate:"required,min=8"`
}

// UserResponse represents a user in responses.
type UserResponse struct {
	ID            string    `json:"id"`
	Email         string    `json:"email"`
	FirstName     string    `json:"first_name,omitempty"`
	LastName      string    `json:"last_name,omitempty"`
	FullName      string    `json:"full_name,omitempty"`
	School        string    `json:"school,omitempty"`
	Department    string    `json:"department,omitempty"`
	AcademicLevel string    `json:"academic_level,omitempty"`
	Timezone      string    `json:"timezone,omitempty"`
	EmailVerified bool      `json:"email_verified"`
	CreatedAt     time.Time `json:"created_at"`
}

// TokenResponse represents tokens in responses.
type TokenResponse struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	TokenType    string    `json:"token_type"`
	ExpiresAt    time.Time `json:"expires_at"`
	User         *UserResponse `json:"user,omitempty"`
}

// SessionResponse represents a session in responses.
type SessionResponse struct {
	ID           string    `json:"id"`
	DeviceName   string    `json:"device_name"`
	DeviceType   string    `json:"device_type"`
	OS           string    `json:"os"`
	Browser      string    `json:"browser"`
	IPAddress    string    `json:"ip_address"`
	Location     string    `json:"location,omitempty"`
	LastActiveAt time.Time `json:"last_active_at"`
	CreatedAt    time.Time `json:"created_at"`
	IsCurrent    bool      `json:"is_current"`
}

// userService implements UserService.
type userService struct {
	userRepo         repository.UserRepository
	refreshTokenRepo repository.RefreshTokenRepository
	sessionRepo      repository.SessionRepository
	passwordResetRepo repository.PasswordResetRepository
	jwtKeyRepo       repository.JWTKeyRepository
	jwtManager       *auth.JWTManager
	passwordHasher   *auth.PasswordHasher
	keyEncryption    *auth.KeyEncryption
	redisClient      *redis.Client
	config           *config.UserServiceConfig
}

// NewUserService creates a new user service.
func NewUserService(
	userRepo repository.UserRepository,
	refreshTokenRepo repository.RefreshTokenRepository,
	sessionRepo repository.SessionRepository,
	passwordResetRepo repository.PasswordResetRepository,
	jwtKeyRepo repository.JWTKeyRepository,
	redisClient *redis.Client,
	cfg *config.UserServiceConfig,
) (UserService, error) {
	// Initialize key encryption
	keyEncryption, err := auth.NewKeyEncryption(cfg.PrivateKeyEncryptionKey)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize key encryption: %w", err)
	}

	// Load or generate RSA keys
	privateKey, err := loadOrGenerateKeys(jwtKeyRepo, keyEncryption)
	if err != nil {
		return nil, fmt.Errorf("failed to load RSA keys: %w", err)
	}

	jwtManager := auth.NewJWTManager(privateKey, &privateKey.PublicKey)
	passwordHasher := auth.NewPasswordHasher(cfg.BcryptCost)

	return &userService{
		userRepo:         userRepo,
		refreshTokenRepo: refreshTokenRepo,
		sessionRepo:      sessionRepo,
		passwordResetRepo: passwordResetRepo,
		jwtKeyRepo:       jwtKeyRepo,
		jwtManager:       jwtManager,
		passwordHasher:   passwordHasher,
		keyEncryption:    keyEncryption,
		redisClient:      redisClient,
		config:           cfg,
	}, nil
}

// loadOrGenerateKeys loads existing RSA keys or generates new ones.
func loadOrGenerateKeys(jwtKeyRepo repository.JWTKeyRepository, keyEncryption *auth.KeyEncryption) (*rsa.PrivateKey, error) {
	ctx := context.Background()

	// Try to load existing private key
	privateKeyRecord, err := jwtKeyRepo.GetActivePrivateKey(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get private key from database: %w", err)
	}

	if privateKeyRecord != nil {
		// Decrypt and load the private key
		decryptedKey, err := keyEncryption.Decrypt(privateKeyRecord.KeyData)
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt private key: %w", err)
		}

		privateKey, err := auth.LoadPrivateKeyFromPEM(decryptedKey)
		if err != nil {
			return nil, fmt.Errorf("failed to load private key from PEM: %w", err)
		}

		return privateKey, nil
	}

	// Generate new RSA key pair
	logger.Info("Generating new RSA key pair")
	privateKey, publicKey, err := auth.GenerateRSAKeyPair(2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate RSA key pair: %w", err)
	}

	// Convert to PEM
	privateKeyPEM := auth.PrivateKeyToPEM(privateKey)
	publicKeyPEM, err := auth.PublicKeyToPEM(publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to convert public key to PEM: %w", err)
	}

	// Encrypt private key
	encryptedPrivateKey, err := keyEncryption.Encrypt(privateKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt private key: %w", err)
	}

	// Deactivate old keys and store new ones
	if err := jwtKeyRepo.DeactivateAllKeys(ctx); err != nil {
		return nil, fmt.Errorf("failed to deactivate old keys: %w", err)
	}

	privateKeyRecord = &model.JWTKey{
		KeyType:  "private",
		KeyData:  encryptedPrivateKey,
		IsActive: true,
	}
	if err := jwtKeyRepo.Create(ctx, privateKeyRecord); err != nil {
		return nil, fmt.Errorf("failed to store private key: %w", err)
	}

	publicKeyRecord := &model.JWTKey{
		KeyType:  "public",
		KeyData:  string(publicKeyPEM),
		IsActive: true,
	}
	if err := jwtKeyRepo.Create(ctx, publicKeyRecord); err != nil {
		return nil, fmt.Errorf("failed to store public key: %w", err)
	}

	return privateKey, nil
}

// Register creates a new user account.
func (s *userService) Register(ctx context.Context, req *RegisterRequest) (*UserResponse, error) {
	// Validate email format
	if _, err := mail.ParseAddress(req.Email); err != nil {
		return nil, echo.NewHTTPError(http.StatusBadRequest, "invalid email format")
	}

	// Check if email already exists
	exists, err := s.userRepo.ExistsByEmail(ctx, req.Email)
	if err != nil {
		logger.Error("failed to check email existence")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if exists {
		return nil, echo.NewHTTPError(http.StatusConflict, "email already registered")
	}

	// Hash password
	passwordHash, err := s.passwordHasher.HashPassword(req.Password)
	if err != nil {
		logger.Error("failed to hash password")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Create user
	user := &model.User{
		Email:         req.Email,
		PasswordHash:  passwordHash,
		FirstName:     req.FirstName,
		LastName:      req.LastName,
		School:        req.School,
		Department:    req.Department,
		AcademicLevel: req.AcademicLevel,
		EmailVerified: true, // Always verify for local dev/testing without SMTP
		IsActive:      true,
	}

	if err := s.userRepo.Create(ctx, user); err != nil {
		logger.Error("failed to create user")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Generate and store verification code
	verificationCode := auth.GenerateVerificationCode()
	expiresAt := time.Now().Add(s.config.VerificationCodeTTL)
	
	if err := s.userRepo.SetVerificationCode(ctx, user.ID, verificationCode, expiresAt); err != nil {
		logger.Error("failed to set verification code")
		// Don't fail registration, just log the error
	}

	// Store verification code in Redis as backup
	redisKey := fmt.Sprintf("email_verification:%s", user.ID.String())
	if err := s.redisClient.Set(ctx, redisKey, verificationCode, s.config.VerificationCodeTTL); err != nil {
		logger.Error("failed to store verification code in redis")
	}

	// TODO: Send verification email (async via notification service)
	logger.Info("verification code generated")

	return mapUserToResponse(user), nil
}

// VerifyEmail verifies a user's email with the provided code.
func (s *userService) VerifyEmail(ctx context.Context, userID string, code string) error {
	id, err := uuid.Parse(userID)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	user, err := s.userRepo.GetByID(ctx, id)
	if err != nil {
		logger.Error("failed to get user")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil {
		return echo.NewHTTPError(http.StatusNotFound, "user not found")
	}

	if user.EmailVerified {
		return echo.NewHTTPError(http.StatusConflict, "email already verified")
	}

	// Check code from Redis first
	redisKey := fmt.Sprintf("email_verification:%s", userID)
	storedCode, err := s.redisClient.Get(ctx, redisKey)
	if err != nil {
		logger.Error("failed to get verification code from redis")
	}

	// If not in Redis or doesn't match, check database
	if storedCode == "" || storedCode != code {
		if user.VerificationCode != code {
			return echo.NewHTTPError(http.StatusBadRequest, "invalid verification code")
		}
		if user.VerificationCodeExpiresAt != nil && time.Now().After(*user.VerificationCodeExpiresAt) {
			return echo.NewHTTPError(http.StatusBadRequest, "verification code expired")
		}
	}

	// Mark email as verified
	if err := s.userRepo.VerifyEmail(ctx, id); err != nil {
		logger.Error("failed to verify email")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Delete verification code from Redis
	s.redisClient.Delete(ctx, redisKey)

	return nil
}

// ResendVerification resends the verification email.
func (s *userService) ResendVerification(ctx context.Context, userID string) error {
	id, err := uuid.Parse(userID)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	user, err := s.userRepo.GetByID(ctx, id)
	if err != nil {
		logger.Error("failed to get user")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil {
		return echo.NewHTTPError(http.StatusNotFound, "user not found")
	}

	if user.EmailVerified {
		return echo.NewHTTPError(http.StatusConflict, "email already verified")
	}

	// Rate limit: check if recently sent
	rateLimitKey := fmt.Sprintf("verification_rate_limit:%s", userID)
	exists, err := s.redisClient.Exists(ctx, rateLimitKey)
	if err != nil {
		logger.Error("failed to check rate limit")
	}
	if exists {
		return echo.NewHTTPError(http.StatusTooManyRequests, "please wait before requesting another code")
	}

	// Generate new code
	verificationCode := auth.GenerateVerificationCode()
	expiresAt := time.Now().Add(s.config.VerificationCodeTTL)

	if err := s.userRepo.SetVerificationCode(ctx, id, verificationCode, expiresAt); err != nil {
		logger.Error("failed to set verification code")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Store in Redis
	redisKey := fmt.Sprintf("email_verification:%s", userID)
	if err := s.redisClient.Set(ctx, redisKey, verificationCode, s.config.VerificationCodeTTL); err != nil {
		logger.Error("failed to store verification code in redis")
	}

	// Set rate limit
	s.redisClient.Set(ctx, rateLimitKey, "1", time.Minute)

	// TODO: Send verification email
	logger.Info("verification code resent")

	return nil
}

// Login authenticates a user and returns tokens.
func (s *userService) Login(ctx context.Context, req *LoginRequest, clientInfo *ClientInfo) (*TokenResponse, error) {
	user, err := s.userRepo.GetByEmail(ctx, strings.ToLower(req.Email))
	if err != nil {
		logger.Error("failed to get user by email")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil {
		return nil, echo.NewHTTPError(http.StatusUnauthorized, "invalid credentials")
	}

	if !user.IsActive {
		return nil, echo.NewHTTPError(http.StatusUnauthorized, "account is deactivated")
	}

	// Verify password
	if err := s.passwordHasher.VerifyPassword(req.Password, user.PasswordHash); err != nil {
		return nil, echo.NewHTTPError(http.StatusUnauthorized, "invalid credentials")
	}

	return s.createTokenPair(ctx, user, clientInfo)
}

// createTokenPair creates a new access and refresh token pair.
func (s *userService) createTokenPair(ctx context.Context, user *model.User, clientInfo *ClientInfo) (*TokenResponse, error) {
	// Generate tokens
	tokenPair, err := s.jwtManager.GenerateTokenPair(
		user.ID.String(),
		user.Email,
		s.config.AccessTokenTTL,
		s.config.RefreshTokenTTL,
	)
	if err != nil {
		logger.Error("failed to generate token pair")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Parse device info
	deviceInfo := parseUserAgent(clientInfo.UserAgent)

	// Create refresh token record
	refreshTokenHash := auth.HashRefreshToken(tokenPair.RefreshToken)
	refreshToken := &model.RefreshToken{
		UserID:     user.ID,
		TokenHash:  refreshTokenHash,
		DeviceInfo: deviceInfo,
		IPAddress:  clientInfo.IPAddress,
		ExpiresAt:  time.Now().Add(s.config.RefreshTokenTTL),
		Revoked:    false,
	}

	if err := s.refreshTokenRepo.Create(ctx, refreshToken); err != nil {
		logger.Error("failed to create refresh token")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Create session
	session := &model.UserSession{
		UserID:         user.ID,
		RefreshTokenID: &refreshToken.ID,
		DeviceName:     deviceInfo.DeviceName,
		DeviceType:     deviceInfo.DeviceType,
		OS:             deviceInfo.OS,
		Browser:        deviceInfo.Browser,
		IPAddress:      clientInfo.IPAddress,
	}
	if err := s.sessionRepo.Create(ctx, session); err != nil {
		logger.Error("failed to create session")
		// Don't fail login, just log the error
	}

	return &TokenResponse{
		AccessToken:  tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		TokenType:    tokenPair.TokenType,
		ExpiresAt:    tokenPair.ExpiresAt,
		User:         mapUserToResponse(user),
	}, nil
}

// RefreshToken refreshes an access token using a refresh token.
func (s *userService) RefreshToken(ctx context.Context, refreshToken string, clientInfo *ClientInfo) (*TokenResponse, error) {
	// Hash the provided token
	tokenHash := auth.HashRefreshToken(refreshToken)

	// Get the stored refresh token
	storedToken, err := s.refreshTokenRepo.GetByTokenHash(ctx, tokenHash)
	if err != nil {
		logger.Error("failed to get refresh token")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if storedToken == nil {
		return nil, echo.NewHTTPError(http.StatusUnauthorized, "invalid refresh token")
	}

	// Check if token is valid
	if !storedToken.CanBeUsed() {
		return nil, echo.NewHTTPError(http.StatusUnauthorized, "refresh token expired or revoked")
	}

	// Get user
	user, err := s.userRepo.GetByID(ctx, storedToken.UserID)
	if err != nil {
		logger.Error("failed to get user")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil || !user.IsActive {
		return nil, echo.NewHTTPError(http.StatusUnauthorized, "user not found or inactive")
	}

	// Create new token pair (this will revoke the old one)
	newTokens, err := s.createTokenPair(ctx, user, clientInfo)
	if err != nil {
		return nil, err
	}

	// Revoke the old refresh token and link to new one
	newTokenHash := auth.HashRefreshToken(newTokens.RefreshToken)
	newStoredToken, err := s.refreshTokenRepo.GetByTokenHash(ctx, newTokenHash)
	if err != nil {
		logger.Error("failed to get new refresh token")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	if err := s.refreshTokenRepo.Revoke(ctx, storedToken.ID, &newStoredToken.ID); err != nil {
		logger.Error("failed to revoke old refresh token")
		// Don't fail, just log
	}

	return newTokens, nil
}

// Logout revokes a user's session.
func (s *userService) Logout(ctx context.Context, userID string, accessTokenJTI string) error {
	id, err := uuid.Parse(userID)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	// Blacklist the access token
	if accessTokenJTI != "" {
		blacklistKey := fmt.Sprintf("token_blacklist:%s", accessTokenJTI)
		if err := s.redisClient.Set(ctx, blacklistKey, userID, s.config.AccessTokenTTL); err != nil {
			logger.Error("failed to blacklist token")
		}
	}

	// Revoke all refresh tokens for user (or implement session-specific logout)
	if err := s.refreshTokenRepo.RevokeAllForUser(ctx, id); err != nil {
		logger.Error("failed to revoke refresh tokens")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Revoke all sessions
	if err := s.sessionRepo.RevokeAllForUser(ctx, id, nil); err != nil {
		logger.Error("failed to revoke sessions")
	}

	return nil
}

// LogoutAll revokes all sessions for a user.
func (s *userService) LogoutAll(ctx context.Context, userID string) error {
	return s.Logout(ctx, userID, "")
}

// GetProfile retrieves a user's profile.
func (s *userService) GetProfile(ctx context.Context, userID string) (*UserResponse, error) {
	id, err := uuid.Parse(userID)
	if err != nil {
		return nil, echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	user, err := s.userRepo.GetByID(ctx, id)
	if err != nil {
		logger.Error("failed to get user")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil {
		return nil, echo.NewHTTPError(http.StatusNotFound, "user not found")
	}

	return mapUserToResponse(user), nil
}

// UpdateProfile updates a user's profile.
func (s *userService) UpdateProfile(ctx context.Context, userID string, req *UpdateProfileRequest) (*UserResponse, error) {
	id, err := uuid.Parse(userID)
	if err != nil {
		return nil, echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	updates := make(map[string]interface{})
	if req.FirstName != "" {
		updates["first_name"] = req.FirstName
	}
	if req.LastName != "" {
		updates["last_name"] = req.LastName
	}
	if req.School != "" {
		updates["school"] = req.School
	}
	if req.Department != "" {
		updates["department"] = req.Department
	}
	if req.AcademicLevel != "" {
		updates["academic_level"] = req.AcademicLevel
	}
	if req.Timezone != "" {
		updates["timezone"] = req.Timezone
	}

	if len(updates) == 0 {
		return nil, echo.NewHTTPError(http.StatusBadRequest, "no fields to update")
	}

	if err := s.userRepo.UpdateProfile(ctx, id, updates); err != nil {
		logger.Error("failed to update profile")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	return s.GetProfile(ctx, userID)
}

// ChangePassword changes a user's password.
func (s *userService) ChangePassword(ctx context.Context, userID string, req *ChangePasswordRequest) error {
	id, err := uuid.Parse(userID)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	user, err := s.userRepo.GetByID(ctx, id)
	if err != nil {
		logger.Error("failed to get user")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil {
		return echo.NewHTTPError(http.StatusNotFound, "user not found")
	}

	// Verify current password
	if err := s.passwordHasher.VerifyPassword(req.CurrentPassword, user.PasswordHash); err != nil {
		return echo.NewHTTPError(http.StatusUnauthorized, "current password is incorrect")
	}

	// Hash new password
	newPasswordHash, err := s.passwordHasher.HashPassword(req.NewPassword)
	if err != nil {
		logger.Error("failed to hash password")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Update password
	if err := s.userRepo.UpdatePassword(ctx, id, newPasswordHash); err != nil {
		logger.Error("failed to update password")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Revoke all refresh tokens (force re-login on all devices)
	if err := s.refreshTokenRepo.RevokeAllForUser(ctx, id); err != nil {
		logger.Error("failed to revoke refresh tokens")
	}

	return nil
}

// RequestPasswordReset initiates a password reset flow.
func (s *userService) RequestPasswordReset(ctx context.Context, email string) error {
	user, err := s.userRepo.GetByEmail(ctx, strings.ToLower(email))
	if err != nil {
		logger.Error("failed to get user by email")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if user == nil {
		// Don't reveal that the email doesn't exist
		logger.Info("password reset requested for non-existent email")
		return nil
	}

	// Rate limit
	rateLimitKey := fmt.Sprintf("password_reset_rate_limit:%s", user.ID.String())
	exists, err := s.redisClient.Exists(ctx, rateLimitKey)
	if err != nil {
		logger.Error("failed to check rate limit")
	}
	if exists {
		return echo.NewHTTPError(http.StatusTooManyRequests, "please wait before requesting another reset")
	}

	// Generate reset token
	resetTokenBytes := make([]byte, 32)
	// In production, use crypto/rand
	// For simplicity, using timestamp-based generation here
	resetToken := auth.HashRefreshToken(string(resetTokenBytes))
	
	passwordReset := &model.PasswordReset{
		UserID:    user.ID,
		TokenHash: auth.HashRefreshToken(resetToken),
		ExpiresAt: time.Now().Add(1 * time.Hour),
		Used:      false,
	}

	if err := s.passwordResetRepo.Create(ctx, passwordReset); err != nil {
		logger.Error("failed to create password reset")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Set rate limit
	s.redisClient.Set(ctx, rateLimitKey, "1", 5*time.Minute)

	// TODO: Send password reset email
	logger.Info("password reset token generated")

	return nil
}

// ResetPassword resets a user's password using a reset token.
func (s *userService) ResetPassword(ctx context.Context, token string, newPassword string) error {
	tokenHash := auth.HashRefreshToken(token)

	reset, err := s.passwordResetRepo.GetByTokenHash(ctx, tokenHash)
	if err != nil {
		logger.Error("failed to get password reset")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if reset == nil || !reset.CanBeUsed() {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid or expired reset token")
	}

	// Hash new password
	newPasswordHash, err := s.passwordHasher.HashPassword(newPassword)
	if err != nil {
		logger.Error("failed to hash password")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Update password
	if err := s.userRepo.UpdatePassword(ctx, reset.UserID, newPasswordHash); err != nil {
		logger.Error("failed to update password")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Mark reset as used
	if err := s.passwordResetRepo.MarkAsUsed(ctx, reset.ID); err != nil {
		logger.Error("failed to mark reset as used")
	}

	// Revoke all refresh tokens
	if err := s.refreshTokenRepo.RevokeAllForUser(ctx, reset.UserID); err != nil {
		logger.Error("failed to revoke refresh tokens")
	}

	return nil
}

// ListSessions lists all active sessions for a user.
func (s *userService) ListSessions(ctx context.Context, userID string) ([]SessionResponse, error) {
	id, err := uuid.Parse(userID)
	if err != nil {
		return nil, echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	sessions, err := s.sessionRepo.GetByUserID(ctx, id)
	if err != nil {
		logger.Error("failed to get sessions")
		return nil, echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	responses := make([]SessionResponse, len(sessions))
	for i, session := range sessions {
		responses[i] = SessionResponse{
			ID:           session.ID.String(),
			DeviceName:   session.DeviceName,
			DeviceType:   session.DeviceType,
			OS:           session.OS,
			Browser:      session.Browser,
			IPAddress:    session.IPAddress,
			Location:     session.Location,
			LastActiveAt: session.LastActiveAt,
			CreatedAt:    session.CreatedAt,
			IsCurrent:    false, // TODO: track current session
		}
	}

	return responses, nil
}

// RevokeSession revokes a specific session.
func (s *userService) RevokeSession(ctx context.Context, userID string, sessionID string) error {
	uid, err := uuid.Parse(userID)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid user ID")
	}

	sid, err := uuid.Parse(sessionID)
	if err != nil {
		return echo.NewHTTPError(http.StatusBadRequest, "invalid session ID")
	}

	session, err := s.sessionRepo.GetByID(ctx, sid)
	if err != nil {
		logger.Error("failed to get session")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}
	if session == nil {
		return echo.NewHTTPError(http.StatusNotFound, "session not found")
	}

	if session.UserID != uid {
		return echo.NewHTTPError(http.StatusForbidden, "unauthorized")
	}

	if err := s.sessionRepo.Revoke(ctx, sid); err != nil {
		logger.Error("failed to revoke session")
		return echo.NewHTTPError(http.StatusInternalServerError, "internal server error")
	}

	// Also revoke associated refresh token
	if session.RefreshTokenID != nil {
		if err := s.refreshTokenRepo.Revoke(ctx, *session.RefreshTokenID, nil); err != nil {
			logger.Error("failed to revoke refresh token")
		}
	}

	return nil
}

// GetPublicKey returns the active public key for JWT validation.
func (s *userService) GetPublicKey(ctx context.Context) (string, error) {
	publicKey, err := s.jwtKeyRepo.GetActivePublicKey(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to get public key: %w", err)
	}
	if publicKey == nil {
		return "", errors.New("no active public key found")
	}
	return publicKey.KeyData, nil
}

// mapUserToResponse maps a user model to a response DTO.
func mapUserToResponse(user *model.User) *UserResponse {
	return &UserResponse{
		ID:            user.ID.String(),
		Email:         user.Email,
		FirstName:     user.FirstName,
		LastName:      user.LastName,
		FullName:      user.FullName(),
		School:        user.School,
		Department:    user.Department,
		AcademicLevel: user.AcademicLevel,
		Timezone:      user.Timezone,
		EmailVerified: user.EmailVerified,
		CreatedAt:     user.CreatedAt,
	}
}

// parseUserAgent parses a user agent string into device info.
func parseUserAgent(userAgent string) model.DeviceInfo {
	// Simple parsing - in production, use a proper user agent parser
	info := model.DeviceInfo{}

	if strings.Contains(userAgent, "Mobile") {
		info.DeviceType = "mobile"
	} else if strings.Contains(userAgent, "Tablet") {
		info.DeviceType = "tablet"
	} else {
		info.DeviceType = "desktop"
	}

	if strings.Contains(userAgent, "Windows") {
		info.OS = "Windows"
	} else if strings.Contains(userAgent, "Mac") {
		info.OS = "macOS"
	} else if strings.Contains(userAgent, "Linux") {
		info.OS = "Linux"
	} else if strings.Contains(userAgent, "Android") {
		info.OS = "Android"
	} else if strings.Contains(userAgent, "iPhone") || strings.Contains(userAgent, "iPad") {
		info.OS = "iOS"
	}

	if strings.Contains(userAgent, "Chrome") {
		info.Browser = "Chrome"
	} else if strings.Contains(userAgent, "Firefox") {
		info.Browser = "Firefox"
	} else if strings.Contains(userAgent, "Safari") {
		info.Browser = "Safari"
	} else if strings.Contains(userAgent, "Edge") {
		info.Browser = "Edge"
	}

	// Set device name
	if info.OS != "" && info.Browser != "" {
		info.DeviceName = info.OS + " - " + info.Browser
	} else {
		info.DeviceName = "Unknown Device"
	}

	return info
}

// getClientIP extracts the client IP from the request.
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for proxies)
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// Check X-Real-IP header
	xri := r.Header.Get("X-Real-Ip")
	if xri != "" {
		return xri
	}

	// Fall back to RemoteAddr
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
