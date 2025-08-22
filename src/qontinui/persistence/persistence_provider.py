"""Persistence provider interface - ported from Qontinui framework.

Interface for pluggable persistence providers.
"""

from typing import Protocol, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SessionMetadata:
    """Session metadata container.
    
    Port of SessionMetadata from Qontinui framework inner class.
    """
    
    session_id: str = ""
    name: str = ""
    application: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_actions: int = 0
    successful_actions: int = 0
    metadata: str = ""
    
    def __init__(self, session_id: str = "", name: str = "", application: str = ""):
        """Initialize session metadata.
        
        Args:
            session_id: Session identifier
            name: Session name
            application: Application being automated
        """
        self.session_id = session_id
        self.name = name
        self.application = application
        self.start_time = datetime.now() if session_id else None
        self.end_time = None
        self.total_actions = 0
        self.successful_actions = 0
        self.metadata = ""
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage.
        
        Returns:
            Success rate as percentage (0-100)
        """
        if self.total_actions > 0:
            return (self.successful_actions / self.total_actions) * 100
        return 0.0


class PersistenceProvider(Protocol):
    """Interface for pluggable persistence providers.
    
    Port of PersistenceProvider from Qontinui framework interface.
    
    Allows different persistence implementations without forcing database dependencies.
    
    Implementations can provide:
    - File-based persistence (JSON/CSV)
    - Database persistence (SQLAlchemy/etc)
    - In-memory persistence (for testing)
    - Cloud storage persistence
    - Custom persistence solutions
    """
    
    def start_session(self, session_name: str, application: str, metadata: str) -> str:
        """Start a new recording session.
        
        Args:
            session_name: Name of the session
            application: Application being automated
            metadata: Optional metadata for the session
            
        Returns:
            Session identifier
        """
        ...
    
    def stop_session(self) -> str:
        """Stop the current recording session.
        
        Returns:
            Session identifier of the stopped session
        """
        ...
    
    def pause_recording(self) -> None:
        """Pause recording without ending the session."""
        ...
    
    def resume_recording(self) -> None:
        """Resume a paused recording session."""
        ...
    
    def is_recording(self) -> bool:
        """Check if currently recording.
        
        Returns:
            True if recording is active
        """
        ...
    
    def record_action(self, record: 'ActionRecord', state_object: Optional['StateObject']) -> None:
        """Record an action execution.
        
        Args:
            record: The ActionRecord to persist
            state_object: The StateObject context (optional)
        """
        ...
    
    def record_batch(self, records: List['ActionRecord']) -> None:
        """Record multiple actions in batch.
        
        Args:
            records: List of ActionRecords to persist
        """
        ...
    
    def export_session(self, session_id: str) -> 'ActionHistory':
        """Export a session as ActionHistory.
        
        Args:
            session_id: The session identifier
            
        Returns:
            ActionHistory containing all records from the session
        """
        ...
    
    def import_session(self, history: 'ActionHistory', session_name: str) -> str:
        """Import ActionHistory as a new session.
        
        Args:
            history: The ActionHistory to import
            session_name: Name for the imported session
            
        Returns:
            Session identifier of the imported session
        """
        ...
    
    def get_all_sessions(self) -> List[str]:
        """Get all available sessions.
        
        Returns:
            List of session identifiers
        """
        ...
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its records.
        
        Args:
            session_id: The session identifier
        """
        ...
    
    def get_session_metadata(self, session_id: str) -> SessionMetadata:
        """Get session metadata.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Session metadata
        """
        ...
    
    def get_current_session_id(self) -> Optional[str]:
        """Get the current active session identifier.
        
        Returns:
            Current session ID or None if not recording
        """
        ...


# Forward references
class ActionRecord:
    """Placeholder for ActionRecord class."""
    pass


class StateObject:
    """Placeholder for StateObject class."""
    pass


class ActionHistory:
    """Placeholder for ActionHistory class."""
    pass