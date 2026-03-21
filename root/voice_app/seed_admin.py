"""Create or promote a user to admin in the SQLite database.

Usage
-----
    # Promote an existing user:
    python seed_admin.py someuser

    # Create a new admin account (you will be prompted for a password):
    python seed_admin.py --create adminuser
"""

import getpass
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, User, init_db  # noqa: E402
from passlib.context import CryptContext  # noqa: E402

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def promote_existing(username: str) -> None:
    """Find an existing user by username and set is_admin = True."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            print(f"No user found with username '{username}'.")
            print("Use --create to create a new admin account.")
            sys.exit(1)

        user.is_admin = True
        db.commit()
        print(f"Success: {username} (id={user.id}) is now an admin.")
    finally:
        db.close()


def create_admin(username: str) -> None:
    """Create a brand-new admin account, prompting for a password."""
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            print(f"An account for '{username}' already exists. Run without --create to promote.")
            sys.exit(1)

        password = getpass.getpass("Password (min 8 chars): ")
        if len(password) < 8:
            print("Password too short.")
            sys.exit(1)

        user = User(
            username=username,
            full_name="Admin",
            hashed_password=hash_password(password),
            is_admin=True,
        )
        db.add(user)
        db.commit()
        print(f"Admin account '{username}' created (id={user.id}).")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--create":
        if len(sys.argv) < 3:
            print("Usage: python seed_admin.py --create <username>")
            sys.exit(1)
        create_admin(sys.argv[2])
    else:
        promote_existing(sys.argv[1])
