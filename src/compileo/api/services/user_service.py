from typing import Optional, List
from ...storage.src.database import get_system_db_connection
from ..models.user import UserInDB, UserCreate
from ..core.auth import get_password_hash, generate_uuid

class UserService:
    @staticmethod
    def get_user_by_email(email: str) -> Optional[UserInDB]:
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            if row:
                return UserInDB(**dict(row))
            return None

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[UserInDB]:
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return UserInDB(**dict(row))
            return None

    @staticmethod
    def create_user(user: UserCreate) -> UserInDB:
        user_id = generate_uuid()
        hashed_password = get_password_hash(user.password)
        with get_system_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (id, email, username, hashed_password) VALUES (?, ?, ?, ?)",
                (user_id, user.email, user.username, hashed_password)
            )
            
            # Initialize default quota
            cursor.execute(
                "INSERT INTO user_quotas (user_id) VALUES (?)",
                (user_id,)
            )
            
            conn.commit()
        return UserService.get_user_by_id(user_id)
