"""
Job manager for async document analysis.
Uses in-memory storage with background task processing.
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import threading


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisJob:
    job_id: str
    user_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100
    progress_message: str = ""
    session_id: Optional[str] = None  # Set when complete


class JobManager:
    """Manages async analysis jobs."""
    
    def __init__(self, retention_minutes: int = 60):
        self.jobs: Dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()
        self.retention_minutes = retention_minutes
        
    def create_job(self, user_id: str) -> AnalysisJob:
        """Create a new job and return its ID."""
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        job = AnalysisJob(
            job_id=job_id,
            user_id=user_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[AnalysisJob]:
        """Get job by ID."""
        with self._lock:
            return self.jobs.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        progress_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[AnalysisJob]:
        """Update job status."""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if progress_message is not None:
                job.progress_message = progress_message
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if session_id is not None:
                job.session_id = session_id
                
            job.updated_at = datetime.utcnow()
            return job
    
    def cleanup_old_jobs(self):
        """Remove jobs older than retention period."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.retention_minutes)
        with self._lock:
            to_remove = [
                job_id for job_id, job in self.jobs.items()
                if job.created_at < cutoff
            ]
            for job_id in to_remove:
                del self.jobs[job_id]


# Global job manager instance
job_manager = JobManager()
