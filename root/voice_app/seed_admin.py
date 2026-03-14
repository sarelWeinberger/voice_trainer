"""Create or promote a user to admin role in the SQLite database.

Usage
-----
    # Promote an existing user:
    python seed_admin.py user@example.com

    # Create a new admin account (you will be prompted for a password):
    python seed_admin.py --create admin@example.com

The script is intentionally simple and meant to be run once during initial
setup. In production, use the Admin Dashboard to manage roles.
"""

import getpass
import sys
import os

# Make sure imports resolve when run from the project root.
sys.path.insert(0, os.path.dirname(__file__))

from database import SessionLocal, User, init_db  # noqa: E402
from auth_middleware import hash_password  # noqa: E402


def promote_existing(email: str) -> None:
    """Find an existing user by email and set their role to 'admin'."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email.lower()).first()
        if user is None:
            print(f"No user found with email '{email}'.")
            print("Use --create to create a new admin account.")
            sys.exit(1)

        user.role = "admin"
        db.commit()
        print(f"Success: {email} (id={user.id}) is now an admin.")
    finally:
        db.close()


def create_admin(email: str) -> None:
    """Create a brand-new admin account, prompting for a password."""
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email.lower()).first():
            print(f"An account for '{email}' already exists. Use --promote instead.")
            sys.exit(1)

        password = getpass.getpass(f"Password for {email}: ")
        if len(password) < 8:
            print("Password must be at least 8 characters.")
            sys.exit(1)

        user = User(
            email=email.lower(),
            password_hash=hash_password(password),
            role="admin",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created admin account: {email} (id={user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()  # ensure tables exist

    args = sys.argv[1:]
    if not args:
        print("Usage:")
        print("  python seed_admin.py <email>           # promote existing user")
        print("  python seed_admin.py --create <email>  # create new admin")
        sys.exit(1)

    if args[0] == "--create":
        if len(args) < 2:
            print("Usage: python seed_admin.py --create <email>")
            sys.exit(1)
        create_admin(args[1])
    else:
        promote_existing(args[0])
