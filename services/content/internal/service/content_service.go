// Package service provides business logic for content management.
package service

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"

	"lexiassist/services/content/internal/model"
	"lexiassist/services/content/internal/repository"
)

var (
	ErrCourseNotFound      = errors.New("course not found")
	ErrMaterialNotFound    = errors.New("material not found")
	ErrQuizNotFound        = errors.New("quiz not found")
	ErrFlashcardNotFound   = errors.New("flashcard not found")
	ErrUnauthorized        = errors.New("unauthorized access")
	ErrInvalidInput        = errors.New("invalid input")
)

// ContentService defines the content management interface.
type ContentService interface {
	// Course operations
	CreateCourse(ctx context.Context, userID uuid.UUID, req *CreateCourseRequest) (*model.Course, error)
	GetCourse(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) (*model.Course, error)
	GetUserCourses(ctx context.Context, userID uuid.UUID) ([]model.Course, error)
	UpdateCourse(ctx context.Context, userID uuid.UUID, courseID uuid.UUID, req *UpdateCourseRequest) (*model.Course, error)
	DeleteCourse(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) error

	// Material operations
	CreateMaterial(ctx context.Context, userID uuid.UUID, req *CreateMaterialRequest) (*model.Material, error)
	GetMaterial(ctx context.Context, userID uuid.UUID, materialID uuid.UUID) (*model.Material, error)
	GetUserMaterials(ctx context.Context, userID uuid.UUID, limit, offset int) ([]model.Material, error)
	GetCourseMaterials(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) ([]model.Material, error)
	UpdateMaterial(ctx context.Context, userID uuid.UUID, materialID uuid.UUID, req *UpdateMaterialRequest) (*model.Material, error)
	DeleteMaterial(ctx context.Context, userID uuid.UUID, materialID uuid.UUID) error
	GeneratePresignURL(ctx context.Context, userID uuid.UUID, materialID uuid.UUID, action string) (*PresignResponse, error)

	// Quiz operations
	CreateQuiz(ctx context.Context, userID uuid.UUID, req *CreateQuizRequest) (*model.Quiz, error)
	GetQuiz(ctx context.Context, userID uuid.UUID, quizID uuid.UUID) (*model.Quiz, error)
	GetUserQuizzes(ctx context.Context, userID uuid.UUID) ([]model.Quiz, error)
	GetCourseQuizzes(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) ([]model.Quiz, error)
	UpdateQuiz(ctx context.Context, userID uuid.UUID, quizID uuid.UUID, req *UpdateQuizRequest) (*model.Quiz, error)
	DeleteQuiz(ctx context.Context, userID uuid.UUID, quizID uuid.UUID) error
	
	// Quiz question operations
	AddQuizQuestion(ctx context.Context, userID uuid.UUID, quizID uuid.UUID, req *AddQuestionRequest) (*model.QuizQuestion, error)
	UpdateQuizQuestion(ctx context.Context, userID uuid.UUID, questionID uuid.UUID, req *UpdateQuestionRequest) (*model.QuizQuestion, error)
	DeleteQuizQuestion(ctx context.Context, userID uuid.UUID, questionID uuid.UUID) error

	// Flashcard operations
	CreateFlashcardDeck(ctx context.Context, userID uuid.UUID, req *CreateDeckRequest) (*model.FlashcardDeck, error)
	GetFlashcardDeck(ctx context.Context, userID uuid.UUID, deckID uuid.UUID) (*model.FlashcardDeck, error)
	GetUserFlashcardDecks(ctx context.Context, userID uuid.UUID) ([]model.FlashcardDeck, error)
	UpdateFlashcardDeck(ctx context.Context, userID uuid.UUID, deckID uuid.UUID, req *UpdateDeckRequest) (*model.FlashcardDeck, error)
	DeleteFlashcardDeck(ctx context.Context, userID uuid.UUID, deckID uuid.UUID) error
	
	// Flashcard card operations
	AddFlashcard(ctx context.Context, userID uuid.UUID, deckID uuid.UUID, req *AddFlashcardRequest) (*model.Flashcard, error)
	UpdateFlashcard(ctx context.Context, userID uuid.UUID, cardID uuid.UUID, req *UpdateFlashcardRequest) (*model.Flashcard, error)
	DeleteFlashcard(ctx context.Context, userID uuid.UUID, cardID uuid.UUID) error
}

// PresignResponse holds the generated presigned URL.
type PresignResponse struct {
	URL         string `json:"url"`
	MaterialID  string `json:"material_id"`
	ExpiresAt   int64  `json:"expires_at"`
}

// contentService implements ContentService.
type contentService struct {
	courseRepo     repository.CourseRepository
	materialRepo   repository.MaterialRepository
	quizRepo       repository.QuizRepository
	flashcardRepo  repository.FlashcardRepository
	minioClient    *minio.Client
	minioBucket    string
	minioPublicURL string
}

// NewContentService creates a new content service.
func NewContentService(
	courseRepo repository.CourseRepository,
	materialRepo repository.MaterialRepository,
	quizRepo repository.QuizRepository,
	flashcardRepo repository.FlashcardRepository,
	minioClient *minio.Client,
	minioBucket string,
	minioPublicURL string,
) ContentService {
	return &contentService{
		courseRepo:     courseRepo,
		materialRepo:   materialRepo,
		quizRepo:       quizRepo,
		flashcardRepo:  flashcardRepo,
		minioClient:    minioClient,
		minioBucket:    minioBucket,
		minioPublicURL: minioPublicURL,
	}
}

// Request/Response types

type CreateCourseRequest struct {
	Name        string `json:"name" validate:"required,max=255"`
	Description string `json:"description"`
	Color       string `json:"color"`
	Semester    string `json:"semester"`
	Year        int    `json:"year"`
}

type UpdateCourseRequest struct {
	Name        string `json:"name,omitempty" validate:"omitempty,max=255"`
	Description string `json:"description,omitempty"`
	Color       string `json:"color,omitempty"`
	Semester    string `json:"semester,omitempty"`
	Year        int    `json:"year,omitempty"`
}

type CreateMaterialRequest struct {
	CourseID *uuid.UUID `json:"course_id,omitempty"`
	Title    string     `json:"title" validate:"required,max=255"`
	FileURL  string     `json:"file_url,omitempty"`
	FileSize int64      `json:"file_size,omitempty"`
	MimeType string     `json:"mime_type,omitempty"`
}

type UpdateMaterialRequest struct {
	CourseID *uuid.UUID `json:"course_id,omitempty"`
	Title    string     `json:"title,omitempty" validate:"omitempty,max=255"`
}

type CreateQuizRequest struct {
	CourseID         *uuid.UUID           `json:"course_id,omitempty"`
	MaterialID       *uuid.UUID           `json:"material_id,omitempty"`
	Title            string               `json:"title" validate:"required,max=255"`
	Description      string               `json:"description,omitempty"`
	TimeLimitMinutes int                  `json:"time_limit_minutes,omitempty"`
	Difficulty       model.DifficultyLevel `json:"difficulty,omitempty"`
	ShuffleQuestions bool                 `json:"shuffle_questions,omitempty"`
	Questions        []AddQuestionRequest `json:"questions,omitempty"`
}

type UpdateQuizRequest struct {
	Title            string               `json:"title,omitempty" validate:"omitempty,max=255"`
	Description      string               `json:"description,omitempty"`
	TimeLimitMinutes int                  `json:"time_limit_minutes,omitempty"`
	Difficulty       model.DifficultyLevel `json:"difficulty,omitempty"`
	ShuffleQuestions *bool                `json:"shuffle_questions,omitempty"`
}

type AddQuestionRequest struct {
	QuestionText  string                `json:"question_text" validate:"required"`
	QuestionType  model.QuestionType    `json:"question_type" validate:"required"`
	Options       model.QuizOptions     `json:"options,omitempty"`
	CorrectAnswer string                `json:"correct_answer,omitempty"`
	Explanation   string                `json:"explanation,omitempty"`
	Points        int                   `json:"points,omitempty"`
	OrderIndex    int                   `json:"order_index"`
	Difficulty    model.DifficultyLevel `json:"difficulty,omitempty"`
}

type UpdateQuestionRequest struct {
	QuestionText  string                `json:"question_text,omitempty"`
	QuestionType  model.QuestionType    `json:"question_type,omitempty"`
	Options       model.QuizOptions     `json:"options,omitempty"`
	CorrectAnswer string                `json:"correct_answer,omitempty"`
	Explanation   string                `json:"explanation,omitempty"`
	Points        int                   `json:"points,omitempty"`
	OrderIndex    int                   `json:"order_index,omitempty"`
	Difficulty    model.DifficultyLevel `json:"difficulty,omitempty"`
}

type CreateDeckRequest struct {
	CourseID    *uuid.UUID `json:"course_id,omitempty"`
	MaterialID  *uuid.UUID `json:"material_id,omitempty"`
	Title       string     `json:"title" validate:"required,max=255"`
	Description string     `json:"description,omitempty"`
	Cards       []AddFlashcardRequest `json:"cards,omitempty"`
}

type UpdateDeckRequest struct {
	Title       string `json:"title,omitempty" validate:"omitempty,max=255"`
	Description string `json:"description,omitempty"`
}

type AddFlashcardRequest struct {
	FrontText  string                `json:"front_text" validate:"required"`
	BackText   string                `json:"back_text" validate:"required"`
	Difficulty model.DifficultyLevel `json:"difficulty,omitempty"`
	OrderIndex int                   `json:"order_index"`
}

type UpdateFlashcardRequest struct {
	FrontText  string                `json:"front_text,omitempty"`
	BackText   string                `json:"back_text,omitempty"`
	Difficulty model.DifficultyLevel `json:"difficulty,omitempty"`
	OrderIndex int                   `json:"order_index,omitempty"`
}

// ==================== Course Operations ====================

func (s *contentService) CreateCourse(ctx context.Context, userID uuid.UUID, req *CreateCourseRequest) (*model.Course, error) {
	course := &model.Course{
		UserID:      userID,
		Name:        req.Name,
		Description: req.Description,
		Color:       req.Color,
		Semester:    req.Semester,
		Year:        req.Year,
	}
	if course.Color == "" {
		course.Color = "#3B82F6"
	}
	
	if err := s.courseRepo.Create(ctx, course); err != nil {
		return nil, fmt.Errorf("failed to create course: %w", err)
	}
	return course, nil
}

func (s *contentService) GetCourse(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) (*model.Course, error) {
	course, err := s.courseRepo.GetByID(ctx, courseID)
	if err != nil {
		return nil, ErrCourseNotFound
	}
	if course.UserID != userID {
		return nil, ErrUnauthorized
	}
	return course, nil
}

func (s *contentService) GetUserCourses(ctx context.Context, userID uuid.UUID) ([]model.Course, error) {
	return s.courseRepo.GetByUserID(ctx, userID)
}

func (s *contentService) UpdateCourse(ctx context.Context, userID uuid.UUID, courseID uuid.UUID, req *UpdateCourseRequest) (*model.Course, error) {
	course, err := s.GetCourse(ctx, userID, courseID)
	if err != nil {
		return nil, err
	}
	
	if req.Name != "" {
		course.Name = req.Name
	}
	if req.Description != "" {
		course.Description = req.Description
	}
	if req.Color != "" {
		course.Color = req.Color
	}
	if req.Semester != "" {
		course.Semester = req.Semester
	}
	if req.Year != 0 {
		course.Year = req.Year
	}
	
	if err := s.courseRepo.Update(ctx, course); err != nil {
		return nil, fmt.Errorf("failed to update course: %w", err)
	}
	return course, nil
}

func (s *contentService) DeleteCourse(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) error {
	_, err := s.GetCourse(ctx, userID, courseID)
	if err != nil {
		return err
	}
	return s.courseRepo.Delete(ctx, courseID)
}

// ==================== Material Operations ====================

func (s *contentService) CreateMaterial(ctx context.Context, userID uuid.UUID, req *CreateMaterialRequest) (*model.Material, error) {
	material := &model.Material{
		UserID:           userID,
		CourseID:         req.CourseID,
		Title:            req.Title,
		FileURL:          req.FileURL,
		FileSize:         req.FileSize,
		MimeType:         req.MimeType,
		ProcessingStatus: model.ProcessingStatusPending,
	}
	
	if err := s.materialRepo.Create(ctx, material); err != nil {
		return nil, fmt.Errorf("failed to create material: %w", err)
	}
	return material, nil
}

func (s *contentService) GetMaterial(ctx context.Context, userID uuid.UUID, materialID uuid.UUID) (*model.Material, error) {
	material, err := s.materialRepo.GetByID(ctx, materialID)
	if err != nil {
		return nil, ErrMaterialNotFound
	}
	if material.UserID != userID {
		return nil, ErrUnauthorized
	}
	return material, nil
}

func (s *contentService) GetUserMaterials(ctx context.Context, userID uuid.UUID, limit, offset int) ([]model.Material, error) {
	return s.materialRepo.GetByUserID(ctx, userID, limit, offset)
}

func (s *contentService) GetCourseMaterials(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) ([]model.Material, error) {
	// Verify course ownership
	_, err := s.GetCourse(ctx, userID, courseID)
	if err != nil {
		return nil, err
	}
	return s.materialRepo.GetByCourseID(ctx, courseID)
}

func (s *contentService) UpdateMaterial(ctx context.Context, userID uuid.UUID, materialID uuid.UUID, req *UpdateMaterialRequest) (*model.Material, error) {
	material, err := s.GetMaterial(ctx, userID, materialID)
	if err != nil {
		return nil, err
	}
	
	if req.Title != "" {
		material.Title = req.Title
	}
	if req.CourseID != nil {
		material.CourseID = req.CourseID
	}
	
	if err := s.materialRepo.Update(ctx, material); err != nil {
		return nil, fmt.Errorf("failed to update material: %w", err)
	}
	return material, nil
}

func (s *contentService) DeleteMaterial(ctx context.Context, userID uuid.UUID, materialID uuid.UUID) error {
	_, err := s.GetMaterial(ctx, userID, materialID)
	if err != nil {
		return err
	}
	return s.materialRepo.Delete(ctx, materialID)
}

// GeneratePresignURL generates a presigned URL for uploading a file to MinIO.
func (s *contentService) GeneratePresignURL(ctx context.Context, userID uuid.UUID, materialID uuid.UUID, action string) (*PresignResponse, error) {
	// Verify the material exists and belongs to the user
	_, err := s.GetMaterial(ctx, userID, materialID)
	if err != nil {
		return nil, err
	}

	if s.minioClient == nil {
		return nil, fmt.Errorf("minio client not initialized")
	}

	// Build object name: materials/{user_id}/{material_id}
	objectName := fmt.Sprintf("materials/%s/%s", userID.String(), materialID.String())

	// Generate presigned PUT URL (valid for 15 minutes)
	expiry := 15 * time.Minute
	presignedURL, err := s.minioClient.PresignedPutObject(ctx, s.minioBucket, objectName, expiry)
	if err != nil {
		return nil, fmt.Errorf("failed to generate presigned URL: %w", err)
	}

	urlString := presignedURL.String()
	if s.minioPublicURL != "" {
		publicURL, err := url.Parse(s.minioPublicURL)
		if err == nil {
			presignedURL.Scheme = publicURL.Scheme
			presignedURL.Host = publicURL.Host
			urlString = presignedURL.String()
		}
	}

	return &PresignResponse{
		URL:        urlString,
		MaterialID: materialID.String(),
		ExpiresAt:  time.Now().Add(expiry).Unix(),
	}, nil
}

// ==================== Quiz Operations ====================

func (s *contentService) CreateQuiz(ctx context.Context, userID uuid.UUID, req *CreateQuizRequest) (*model.Quiz, error) {
	quiz := &model.Quiz{
		UserID:           userID,
		CourseID:         req.CourseID,
		MaterialID:       req.MaterialID,
		Title:            req.Title,
		Description:      req.Description,
		TimeLimitMinutes: req.TimeLimitMinutes,
		Difficulty:       req.Difficulty,
		ShuffleQuestions: req.ShuffleQuestions,
	}
	
	if err := s.quizRepo.Create(ctx, quiz); err != nil {
		return nil, fmt.Errorf("failed to create quiz: %w", err)
	}
	
	// Add questions if provided
	for i, qReq := range req.Questions {
		// Generate IDs for options if not provided
		options := make(model.QuizOptions, len(qReq.Options))
		for j, opt := range qReq.Options {
			if opt.ID == "" {
				opt.ID = uuid.New().String()
			}
			opt.OrderIndex = j
			options[j] = opt
		}
		
		question := &model.QuizQuestion{
			QuizID:        quiz.ID,
			QuestionText:  qReq.QuestionText,
			QuestionType:  qReq.QuestionType,
			Options:       options,
			CorrectAnswer: qReq.CorrectAnswer,
			Explanation:   qReq.Explanation,
			Points:        qReq.Points,
			OrderIndex:    i,
			Difficulty:    qReq.Difficulty,
		}
		if question.Points == 0 {
			question.Points = 1
		}
		if err := s.quizRepo.CreateQuestion(ctx, question); err != nil {
			return nil, fmt.Errorf("failed to create question: %w", err)
		}
	}
	
	// Reload quiz with questions
	return s.quizRepo.GetByID(ctx, quiz.ID)
}

func (s *contentService) GetQuiz(ctx context.Context, userID uuid.UUID, quizID uuid.UUID) (*model.Quiz, error) {
	quiz, err := s.quizRepo.GetByID(ctx, quizID)
	if err != nil {
		return nil, ErrQuizNotFound
	}
	if quiz.UserID != userID {
		return nil, ErrUnauthorized
	}
	return quiz, nil
}

func (s *contentService) GetUserQuizzes(ctx context.Context, userID uuid.UUID) ([]model.Quiz, error) {
	return s.quizRepo.GetByUserID(ctx, userID)
}

func (s *contentService) GetCourseQuizzes(ctx context.Context, userID uuid.UUID, courseID uuid.UUID) ([]model.Quiz, error) {
	// Verify course ownership
	_, err := s.GetCourse(ctx, userID, courseID)
	if err != nil {
		return nil, err
	}
	return s.quizRepo.GetByCourseID(ctx, courseID)
}

func (s *contentService) UpdateQuiz(ctx context.Context, userID uuid.UUID, quizID uuid.UUID, req *UpdateQuizRequest) (*model.Quiz, error) {
	quiz, err := s.GetQuiz(ctx, userID, quizID)
	if err != nil {
		return nil, err
	}
	
	if req.Title != "" {
		quiz.Title = req.Title
	}
	if req.Description != "" {
		quiz.Description = req.Description
	}
	if req.TimeLimitMinutes != 0 {
		quiz.TimeLimitMinutes = req.TimeLimitMinutes
	}
	if req.Difficulty != "" {
		quiz.Difficulty = req.Difficulty
	}
	if req.ShuffleQuestions != nil {
		quiz.ShuffleQuestions = *req.ShuffleQuestions
	}
	
	if err := s.quizRepo.Update(ctx, quiz); err != nil {
		return nil, fmt.Errorf("failed to update quiz: %w", err)
	}
	return quiz, nil
}

func (s *contentService) DeleteQuiz(ctx context.Context, userID uuid.UUID, quizID uuid.UUID) error {
	_, err := s.GetQuiz(ctx, userID, quizID)
	if err != nil {
		return err
	}
	return s.quizRepo.Delete(ctx, quizID)
}

func (s *contentService) AddQuizQuestion(ctx context.Context, userID uuid.UUID, quizID uuid.UUID, req *AddQuestionRequest) (*model.QuizQuestion, error) {
	// Verify quiz ownership
	_, err := s.GetQuiz(ctx, userID, quizID)
	if err != nil {
		return nil, err
	}
	
	// Generate IDs for options if not provided
	options := make(model.QuizOptions, len(req.Options))
	for j, opt := range req.Options {
		if opt.ID == "" {
			opt.ID = uuid.New().String()
		}
		opt.OrderIndex = j
		options[j] = opt
	}
	
	question := &model.QuizQuestion{
		QuizID:        quizID,
		QuestionText:  req.QuestionText,
		QuestionType:  req.QuestionType,
		Options:       options,
		CorrectAnswer: req.CorrectAnswer,
		Explanation:   req.Explanation,
		Points:        req.Points,
		OrderIndex:    req.OrderIndex,
		Difficulty:    req.Difficulty,
	}
	if question.Points == 0 {
		question.Points = 1
	}
	
	if err := s.quizRepo.CreateQuestion(ctx, question); err != nil {
		return nil, fmt.Errorf("failed to create question: %w", err)
	}
	return question, nil
}

func (s *contentService) UpdateQuizQuestion(ctx context.Context, userID uuid.UUID, questionID uuid.UUID, req *UpdateQuestionRequest) (*model.QuizQuestion, error) {
	// Get question and verify ownership through quiz
	question, err := s.quizRepo.GetQuestionsByQuizID(ctx, uuid.Nil)
	if err != nil {
		return nil, err
	}
	
	// Find the specific question
	var targetQuestion *model.QuizQuestion
	for i := range question {
		if question[i].ID == questionID {
			targetQuestion = &question[i]
			break
		}
	}
	if targetQuestion == nil {
		return nil, errors.New("question not found")
	}
	
	// Verify quiz ownership
	_, err = s.GetQuiz(ctx, userID, targetQuestion.QuizID)
	if err != nil {
		return nil, err
	}
	
	// Update fields
	if req.QuestionText != "" {
		targetQuestion.QuestionText = req.QuestionText
	}
	if req.QuestionType != "" {
		targetQuestion.QuestionType = req.QuestionType
	}
	if req.Options != nil {
		// Generate IDs for new options
		options := make(model.QuizOptions, len(req.Options))
		for j, opt := range req.Options {
			if opt.ID == "" {
				opt.ID = uuid.New().String()
			}
			opt.OrderIndex = j
			options[j] = opt
		}
		targetQuestion.Options = options
	}
	if req.CorrectAnswer != "" {
		targetQuestion.CorrectAnswer = req.CorrectAnswer
	}
	if req.Explanation != "" {
		targetQuestion.Explanation = req.Explanation
	}
	if req.Points != 0 {
		targetQuestion.Points = req.Points
	}
	if req.OrderIndex != 0 {
		targetQuestion.OrderIndex = req.OrderIndex
	}
	if req.Difficulty != "" {
		targetQuestion.Difficulty = req.Difficulty
	}
	
	if err := s.quizRepo.UpdateQuestion(ctx, targetQuestion); err != nil {
		return nil, fmt.Errorf("failed to update question: %w", err)
	}
	return targetQuestion, nil
}

func (s *contentService) DeleteQuizQuestion(ctx context.Context, userID uuid.UUID, questionID uuid.UUID) error {
	// Get question to find quiz ID
	question, err := s.quizRepo.GetQuestionsByQuizID(ctx, uuid.Nil)
	if err != nil {
		return err
	}
	
	var quizID uuid.UUID
	for _, q := range question {
		if q.ID == questionID {
			quizID = q.QuizID
			break
		}
	}
	if quizID == uuid.Nil {
		return errors.New("question not found")
	}
	
	// Verify quiz ownership
	_, err = s.GetQuiz(ctx, userID, quizID)
	if err != nil {
		return err
	}
	
	return s.quizRepo.DeleteQuestion(ctx, questionID)
}

// ==================== Flashcard Operations ====================

func (s *contentService) CreateFlashcardDeck(ctx context.Context, userID uuid.UUID, req *CreateDeckRequest) (*model.FlashcardDeck, error) {
	deck := &model.FlashcardDeck{
		UserID:      userID,
		CourseID:    req.CourseID,
		MaterialID:  req.MaterialID,
		Title:       req.Title,
		Description: req.Description,
	}
	
	if err := s.flashcardRepo.CreateDeck(ctx, deck); err != nil {
		return nil, fmt.Errorf("failed to create deck: %w", err)
	}
	
	// Add cards if provided
	for i, cReq := range req.Cards {
		card := &model.Flashcard{
			DeckID:     deck.ID,
			FrontText:  cReq.FrontText,
			BackText:   cReq.BackText,
			Difficulty: cReq.Difficulty,
			OrderIndex: i,
		}
		if err := s.flashcardRepo.CreateCard(ctx, card); err != nil {
			return nil, fmt.Errorf("failed to create card: %w", err)
		}
	}
	
	// Reload deck with cards
	return s.flashcardRepo.GetDeckByID(ctx, deck.ID)
}

func (s *contentService) GetFlashcardDeck(ctx context.Context, userID uuid.UUID, deckID uuid.UUID) (*model.FlashcardDeck, error) {
	deck, err := s.flashcardRepo.GetDeckByID(ctx, deckID)
	if err != nil {
		return nil, ErrFlashcardNotFound
	}
	if deck.UserID != userID {
		return nil, ErrUnauthorized
	}
	return deck, nil
}

func (s *contentService) GetUserFlashcardDecks(ctx context.Context, userID uuid.UUID) ([]model.FlashcardDeck, error) {
	return s.flashcardRepo.GetDecksByUserID(ctx, userID)
}

func (s *contentService) UpdateFlashcardDeck(ctx context.Context, userID uuid.UUID, deckID uuid.UUID, req *UpdateDeckRequest) (*model.FlashcardDeck, error) {
	deck, err := s.GetFlashcardDeck(ctx, userID, deckID)
	if err != nil {
		return nil, err
	}
	
	if req.Title != "" {
		deck.Title = req.Title
	}
	if req.Description != "" {
		deck.Description = req.Description
	}
	
	if err := s.flashcardRepo.UpdateDeck(ctx, deck); err != nil {
		return nil, fmt.Errorf("failed to update deck: %w", err)
	}
	return deck, nil
}

func (s *contentService) DeleteFlashcardDeck(ctx context.Context, userID uuid.UUID, deckID uuid.UUID) error {
	_, err := s.GetFlashcardDeck(ctx, userID, deckID)
	if err != nil {
		return err
	}
	return s.flashcardRepo.DeleteDeck(ctx, deckID)
}

func (s *contentService) AddFlashcard(ctx context.Context, userID uuid.UUID, deckID uuid.UUID, req *AddFlashcardRequest) (*model.Flashcard, error) {
	// Verify deck ownership
	_, err := s.GetFlashcardDeck(ctx, userID, deckID)
	if err != nil {
		return nil, err
	}
	
	card := &model.Flashcard{
		DeckID:     deckID,
		FrontText:  req.FrontText,
		BackText:   req.BackText,
		Difficulty: req.Difficulty,
		OrderIndex: req.OrderIndex,
	}
	
	if err := s.flashcardRepo.CreateCard(ctx, card); err != nil {
		return nil, fmt.Errorf("failed to create card: %w", err)
	}
	return card, nil
}

func (s *contentService) UpdateFlashcard(ctx context.Context, userID uuid.UUID, cardID uuid.UUID, req *UpdateFlashcardRequest) (*model.Flashcard, error) {
	// Get card
	card, err := s.flashcardRepo.GetCardByID(ctx, cardID)
	if err != nil {
		return nil, ErrFlashcardNotFound
	}
	
	// Verify deck ownership
	_, err = s.GetFlashcardDeck(ctx, userID, card.DeckID)
	if err != nil {
		return nil, err
	}
	
	if req.FrontText != "" {
		card.FrontText = req.FrontText
	}
	if req.BackText != "" {
		card.BackText = req.BackText
	}
	if req.Difficulty != "" {
		card.Difficulty = req.Difficulty
	}
	if req.OrderIndex != 0 {
		card.OrderIndex = req.OrderIndex
	}
	
	if err := s.flashcardRepo.UpdateCard(ctx, card); err != nil {
		return nil, fmt.Errorf("failed to update card: %w", err)
	}
	return card, nil
}

func (s *contentService) DeleteFlashcard(ctx context.Context, userID uuid.UUID, cardID uuid.UUID) error {
	// Get card
	card, err := s.flashcardRepo.GetCardByID(ctx, cardID)
	if err != nil {
		return ErrFlashcardNotFound
	}
	
	// Verify deck ownership
	_, err = s.GetFlashcardDeck(ctx, userID, card.DeckID)
	if err != nil {
		return err
	}
	
	return s.flashcardRepo.DeleteCard(ctx, cardID)
}
