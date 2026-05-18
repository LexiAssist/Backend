package services

import (
	"bytes"
	"fmt"
	"html/template"
	"net/smtp"
	"os"
	"strings"

	"go.uber.org/zap"
	"lexiassist/shared/pkg/logger"
)

// EmailService handles SMTP email sending
type EmailService struct {
	host     string
	port     string
	username string
	password string
	from     string
	enabled  bool
}

// NewEmailService creates a new email service
func NewEmailService() *EmailService {
	host := os.Getenv("SMTP_HOST")
	port := os.Getenv("SMTP_PORT")
	username := os.Getenv("SMTP_USERNAME")
	password := os.Getenv("SMTP_PASSWORD")
	from := os.Getenv("SMTP_FROM")

	if host == "" || port == "" {
		logger.Warn("SMTP configuration incomplete, email service disabled")
		return &EmailService{enabled: false}
	}

	if from == "" {
		from = "notifications@lexiassist.com"
	}

	logger.Info("Email service initialized",
		zap.String("smtp_host", host),
		zap.String("smtp_port", port),
	)
	return &EmailService{
		host:     host,
		port:     port,
		username: username,
		password: password,
		from:     from,
		enabled:  true,
	}
}

// SendEmail sends a plain text email
func (s *EmailService) SendEmail(to, subject, body string) error {
	if !s.enabled {
		logger.Warn("Email service is disabled, email not sent")
		return nil
	}

	msg := []byte(fmt.Sprintf(
		"To: %s\r\n"+
			"From: %s\r\n"+
			"Subject: %s\r\n"+
			"Content-Type: text/plain; charset=UTF-8\r\n"+
			"\r\n"+
			"%s",
		to, s.from, subject, body,
	))

	addr := fmt.Sprintf("%s:%s", s.host, s.port)

	var auth smtp.Auth
	if s.username != "" && s.password != "" {
		auth = smtp.PlainAuth("", s.username, s.password, s.host)
	}

	err := smtp.SendMail(addr, auth, s.from, []string{to}, msg)
	if err != nil {
		return fmt.Errorf("failed to send email: %w", err)
	}

	logger.Debug("Email sent", zap.String("to", to))
	return nil
}

// SendHTMLEmail sends an HTML email
func (s *EmailService) SendHTMLEmail(to, subject, htmlBody string) error {
	if !s.enabled {
		logger.Warn("Email service is disabled, email not sent")
		return nil
	}

	// Create multipart message
	boundary := "boundary-lexiassist-" + generateBoundary()

	var msg bytes.Buffer
	fmt.Fprintf(&msg, "To: %s\r\n", to)
	fmt.Fprintf(&msg, "From: %s\r\n", s.from)
	fmt.Fprintf(&msg, "Subject: %s\r\n", subject)
	fmt.Fprintf(&msg, "MIME-Version: 1.0\r\n")
	fmt.Fprintf(&msg, "Content-Type: multipart/alternative; boundary=%s\r\n", boundary)
	fmt.Fprintf(&msg, "\r\n")
	fmt.Fprintf(&msg, "--%s\r\n", boundary)
	fmt.Fprintf(&msg, "Content-Type: text/plain; charset=UTF-8\r\n")
	fmt.Fprintf(&msg, "\r\n")
	fmt.Fprintf(&msg, "%s\r\n", stripHTML(htmlBody))
	fmt.Fprintf(&msg, "\r\n")
	fmt.Fprintf(&msg, "--%s\r\n", boundary)
	fmt.Fprintf(&msg, "Content-Type: text/html; charset=UTF-8\r\n")
	fmt.Fprintf(&msg, "\r\n")
	fmt.Fprintf(&msg, "%s\r\n", htmlBody)
	fmt.Fprintf(&msg, "\r\n")
	fmt.Fprintf(&msg, "--%s--\r\n", boundary)

	addr := fmt.Sprintf("%s:%s", s.host, s.port)

	var auth smtp.Auth
	if s.username != "" && s.password != "" {
		auth = smtp.PlainAuth("", s.username, s.password, s.host)
	}

	err := smtp.SendMail(addr, auth, s.from, []string{to}, msg.Bytes())
	if err != nil {
		return fmt.Errorf("failed to send HTML email: %w", err)
	}

	logger.Debug("HTML email sent", zap.String("to", to))
	return nil
}

// SendTemplateEmail sends an email using a template
func (s *EmailService) SendTemplateEmail(to, subject, templateName string, data interface{}) error {
	if !s.enabled {
		logger.Warn("Email service is disabled, email not sent")
		return nil
	}

	tmpl, ok := emailTemplates[templateName]
	if !ok {
		return fmt.Errorf("template %s not found", templateName)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	return s.SendHTMLEmail(to, subject, buf.String())
}

// IsEnabled returns whether email service is enabled
func (s *EmailService) IsEnabled() bool {
	return s.enabled
}

// Email templates
var emailTemplates = map[string]*template.Template{}

func init() {
	// Quiz completion template
	emailTemplates["quiz_completed"] = template.Must(template.New("quiz_completed").Parse(`
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #4CAF50; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .score { font-size: 24px; font-weight: bold; color: #4CAF50; }
        .footer { padding: 20px; text-align: center; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Quiz Completed!</h1>
        </div>
        <div class="content">
            <p>Hi {{.Name}},</p>
            <p>Congratulations on completing <strong>{{.QuizTitle}}</strong>!</p>
            <p class="score">Your Score: {{.Score}}%</p>
            <p>{{.Message}}</p>
            <p><a href="{{.QuizURL}}" style="background: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View Results</a></p>
        </div>
        <div class="footer">
            <p>You're receiving this because you completed a quiz on LexiAssist.</p>
        </div>
    </div>
</body>
</html>
`))

	// Streak achievement template
	emailTemplates["streak_achieved"] = template.Must(template.New("streak_achieved").Parse(`
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #FF9800; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .streak { font-size: 48px; text-align: center; }
        .footer { padding: 20px; text-align: center; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Streak Achieved!</h1>
        </div>
        <div class="content">
            <p>Hi {{.Name}},</p>
            <div class="streak">{{.StreakCount}} {{if eq .StreakCount 1}}day{{else}}days{{end}}</div>
            <p style="text-align: center;">You're on fire! Keep up the amazing work!</p>
            <p>{{.Message}}</p>
        </div>
        <div class="footer">
            <p>Keep the momentum going!</p>
        </div>
    </div>
</body>
</html>
`))

	// Email verification template
	emailTemplates["email_verification"] = template.Must(template.New("email_verification").Parse(`
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #673AB7; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .code { font-size: 32px; font-weight: bold; color: #673AB7; text-align: center; letter-spacing: 4px; }
        .footer { padding: 20px; text-align: center; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Verify Your Email</h1>
        </div>
        <div class="content">
            <p>Hi {{.Name}},</p>
            <p>Thanks for signing up for LexiAssist. Use the code below to verify your email address:</p>
            <p class="code">{{.Code}}</p>
            <p>This code expires in 15 minutes.</p>
        </div>
        <div class="footer">
            <p>You're receiving this because you registered on LexiAssist.</p>
        </div>
    </div>
</body>
</html>
`))

	// Password reset template
	emailTemplates["password_reset"] = template.Must(template.New("password_reset").Parse(`
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #F44336; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .code { font-size: 32px; font-weight: bold; color: #F44336; text-align: center; letter-spacing: 4px; }
        .footer { padding: 20px; text-align: center; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <p>Hi {{.Name}},</p>
            <p>We received a request to reset your LexiAssist password. Use the code below:</p>
            <p class="code">{{.Code}}</p>
            <p>This code expires in 15 minutes. If you didn't request this, you can safely ignore this email.</p>
        </div>
        <div class="footer">
            <p>You're receiving this because a password reset was requested on LexiAssist.</p>
        </div>
    </div>
</body>
</html>
`))

	// Study reminder template
	emailTemplates["study_reminder"] = template.Must(template.New("study_reminder").Parse(`
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #2196F3; color: white; padding: 20px; text-align: center; }
        .content { padding: 20px; background: #f9f9f9; }
        .footer { padding: 20px; text-align: center; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Time to Study!</h1>
        </div>
        <div class="content">
            <p>Hi {{.Name}},</p>
            <p>{{.Message}}</p>
            <p><a href="{{.StudyURL}}" style="background: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Start Studying</a></p>
        </div>
        <div class="footer">
            <p>Consistency is key to learning!</p>
        </div>
    </div>
</body>
</html>
`))
}

func generateBoundary() string {
	// Simple boundary generator
	chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	result := make([]byte, 16)
	for i := range result {
		result[i] = chars[i%len(chars)]
	}
	return string(result)
}

func stripHTML(html string) string {
	// Simple HTML stripper
	result := html
	result = strings.ReplaceAll(result, "<br>", "\n")
	result = strings.ReplaceAll(result, "<br/>", "\n")
	result = strings.ReplaceAll(result, "<p>", "\n")
	result = strings.ReplaceAll(result, "</p>", "")

	// Remove all tags
	for {
		start := strings.Index(result, "<")
		if start == -1 {
			break
		}
		end := strings.Index(result[start:], ">")
		if end == -1 {
			break
		}
		result = result[:start] + result[start+end+1:]
	}

	return strings.TrimSpace(result)
}
