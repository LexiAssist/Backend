-- Migration: Add role column to users table for RBAC
-- Created: 2026-04-03

-- Add role column with default 'student'
ALTER TABLE auth.users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'student';

-- Create index for role lookups
CREATE INDEX IF NOT EXISTS idx_users_role ON auth.users(role);

-- Add check constraint for valid roles
ALTER TABLE auth.users DROP CONSTRAINT IF EXISTS chk_user_role;
ALTER TABLE auth.users ADD CONSTRAINT chk_user_role 
    CHECK (role IN ('student', 'instructor', 'admin', 'super_admin'));

-- Create super admin user for development
-- Password: SuperAdmin123! (bcrypt hash)
INSERT INTO auth.users (
    id, 
    email, 
    password_hash, 
    first_name, 
    last_name, 
    email_verified, 
    is_active,
    role,
    school,
    created_at,
    updated_at
) VALUES (
    '11111111-1111-1111-1111-111111111111',
    'admin@lexiassist.dev',
    '$2a$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUW',  -- bcrypt of 'SuperAdmin123!'
    'Super',
    'Admin',
    true,
    true,
    'super_admin',
    'Development',
    NOW(),
    NOW()
) ON CONFLICT (email) DO UPDATE SET 
    role = 'super_admin',
    email_verified = true,
    is_active = true,
    updated_at = NOW();

-- Also create a regular test user
INSERT INTO auth.users (
    id, 
    email, 
    password_hash, 
    first_name, 
    last_name, 
    email_verified, 
    is_active,
    role,
    school,
    created_at,
    updated_at
) VALUES (
    '22222222-2222-2222-2222-222222222222',
    'test@lexiassist.dev',
    '$2a$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUW',  -- bcrypt of 'SuperAdmin123!'
    'Test',
    'User',
    true,
    true,
    'student',
    'Test University',
    NOW(),
    NOW()
) ON CONFLICT (email) DO UPDATE SET 
    email_verified = true,
    is_active = true,
    updated_at = NOW();

COMMENT ON COLUMN auth.users.role IS 'User role: student, instructor, admin, super_admin';
