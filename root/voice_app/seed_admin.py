"""One-time script to set a user as admin in Firestore.

Usage: python seed_admin.py user@example.com

The user must have logged in at least once so their profile
exists in Firestore.
"""

import sys
import os
import firebase_admin
from firebase_admin import credentials, firestore

cred_path = os.path.join(os.path.dirname(__file__), "service-account-key.json")
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()


def make_admin(email):
    users = db.collection("users").where("email", "==", email).stream()
    found = False
    for doc in users:
        doc.reference.update({"role": "admin"})
        print(f"Set {email} (uid={doc.id}) as admin.")
        found = True

    if not found:
        print(f"No user found with email '{email}'.")
        print("Make sure they have logged in at least once.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python seed_admin.py user@example.com")
        sys.exit(1)
    make_admin(sys.argv[1])
