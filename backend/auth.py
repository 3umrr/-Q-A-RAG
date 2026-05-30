

import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext


SECRET_KEY  = os.getenv("SECRET_KEY", "change-this-secret-in-production-please")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 24   


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")



def hash_password(plain_password: str) -> str:
    """Turn a plain-text password into a bcrypt hash."""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check if a plain-text password matches a stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


# ── JWT helpers ────────────────────────────────────────────────────────────────

def create_access_token(user_id: int, username: str) -> str:
  
    payload = {
        "sub": str(user_id),   
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict | None:
   
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
