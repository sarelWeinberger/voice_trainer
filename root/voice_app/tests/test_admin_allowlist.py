"""Admin-managed allowlist: invite, scoping, revocation, per-user history.

The DB URL is pinned to a throwaway file *before* app/database are imported,
because database.py resolves it at import time.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP_DIR = tempfile.mkdtemp(prefix="voice-allowlist-test-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP_DIR, "test.db")

from fastapi.testclient import TestClient  # noqa: E402

import app as appmod  # noqa: E402
from database import Result, SessionLocal, User  # noqa: E402

SUPER = ("sarelw1@gmail.com", "admin1234")
SHARED_PW = "qwertyui"


def auth(client, username, password):
    resp = client.post("/api/login", data={"username": username, "password": password})
    assert resp.status_code == 200, resp.json()
    return {"Authorization": "Bearer " + resp.json()["access_token"]}


def make_admin(client, super_headers, email):
    """Invite an email, register it, and promote it to class admin (role 2)."""
    client.post("/api/admin/allowed-emails", json={"email": email}, headers=super_headers)
    client.post(
        "/api/register",
        json={"username": email, "password": SHARED_PW, "accepted_terms": True},
    )
    db = SessionLocal()
    uid = db.query(User).filter(User.username == email).first().id
    db.close()
    resp = client.put(
        f"/api/admin/users/{uid}/role",
        json={"role": 2, "is_admin": True},
        headers=super_headers,
    )
    assert resp.status_code == 200, resp.json()
    return auth(client, email, SHARED_PW)


class AllowlistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(appmod.app)
        cls.client.__enter__()          # fire startup events (seeding, super admin)
        cls.super_headers = auth(cls.client, *SUPER)

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def _invite(self, headers, email, **extra):
        return self.client.post(
            "/api/admin/allowed-emails", json={"email": email, **extra}, headers=headers
        )

    def _revoke(self, headers, email):
        return self.client.request(
            "DELETE", "/api/admin/allowed-emails", json={"email": email}, headers=headers
        )

    def test_invite_then_register_then_history(self):
        email = "invited@example.com"
        resp = self._invite(self.super_headers, email, full_name="בדיקה")
        self.assertEqual(resp.status_code, 200, resp.json())
        self.assertFalse(resp.json()["registered"])

        # Logging in before signing up explains what to do instead of "wrong password".
        early = self.client.post("/api/login", data={"username": email, "password": SHARED_PW})
        self.assertEqual(early.status_code, 400)
        self.assertIn("הרשמה", early.json()["detail"])

        reg = self.client.post(
            "/api/register",
            json={"username": email, "password": SHARED_PW, "accepted_terms": True},
        )
        self.assertEqual(reg.status_code, 200, reg.json())
        # Auto-attached to the inviting admin's class — no class code typed.
        self.assertIsNotNone(reg.json()["joined_class_code"])

        headers = auth(self.client, email, SHARED_PW)
        self.assertEqual(self.client.get("/api/history", headers=headers).json()["history"], [])

        db = SessionLocal()
        user = db.query(User).filter(User.username == email).first()
        db.add(Result(user_id=user.id, title="t", raw_scores=json.dumps({"scores": {"G": 1}})))
        db.commit()
        uid = user.id
        db.close()

        self.assertEqual(len(self.client.get("/api/history", headers=headers).json()["history"]), 1)

        admin_view = self.client.get(f"/api/admin/users/{uid}/results", headers=self.super_headers)
        self.assertEqual(len(admin_view.json()["results"]), 1)

        row = next(
            i for i in self.client.get("/api/admin/allowed-emails", headers=self.super_headers).json()["invites"]
            if i["email"] == email
        )
        self.assertTrue(row["registered"])
        self.assertEqual(row["results_count"], 1)

    def test_revoking_blocks_login_but_keeps_history(self):
        email = "revoked@example.com"
        self._invite(self.super_headers, email)
        self.client.post(
            "/api/register",
            json={"username": email, "password": SHARED_PW, "accepted_terms": True},
        )
        db = SessionLocal()
        user = db.query(User).filter(User.username == email).first()
        db.add(Result(user_id=user.id, title="t", raw_scores=json.dumps({"scores": {"G": 0}})))
        db.commit()
        uid = user.id
        db.close()

        self.assertEqual(self._revoke(self.super_headers, email).status_code, 200)

        blocked = self.client.post("/api/login", data={"username": email, "password": SHARED_PW})
        self.assertEqual(blocked.status_code, 403)

        db = SessionLocal()
        self.assertEqual(db.query(Result).filter(Result.user_id == uid).count(), 1)
        db.close()

    def test_super_admin_cannot_be_removed(self):
        self.assertEqual(self._revoke(self.super_headers, SUPER[0]).status_code, 400)

    def test_bad_and_duplicate_emails_rejected(self):
        self.assertEqual(self._invite(self.super_headers, "not-an-email").status_code, 400)
        self._invite(self.super_headers, "dup@example.com")
        self.assertEqual(self._invite(self.super_headers, "DUP@example.com").status_code, 400)

    def test_class_admins_only_see_and_touch_their_own_invites(self):
        a = make_admin(self.client, self.super_headers, "teacher.a@example.com")
        b = make_admin(self.client, self.super_headers, "teacher.b@example.com")

        self.assertEqual(self._invite(a, "student.a@example.com").status_code, 200)
        self.assertEqual(self._invite(b, "student.b@example.com").status_code, 200)

        a_emails = [i["email"] for i in self.client.get("/api/admin/allowed-emails", headers=a).json()["invites"]]
        b_emails = [i["email"] for i in self.client.get("/api/admin/allowed-emails", headers=b).json()["invites"]]
        self.assertEqual(a_emails, ["student.a@example.com"])
        self.assertEqual(b_emails, ["student.b@example.com"])

        self.assertEqual(self._revoke(a, "student.b@example.com").status_code, 403)

        # A student registers into A's class; only A can read their history.
        self.client.post(
            "/api/register",
            json={"username": "student.a@example.com", "password": SHARED_PW, "accepted_terms": True},
        )
        db = SessionLocal()
        student = db.query(User).filter(User.username == "student.a@example.com").first()
        db.add(Result(user_id=student.id, title="t", raw_scores=json.dumps({"scores": {"G": 2}})))
        db.commit()
        sid = student.id
        db.close()

        a_users = self.client.get("/api/admin/users", headers=a).json()["users"]
        self.assertEqual([(u["username"], u["results_count"]) for u in a_users],
                         [("student.a@example.com", 1)])
        self.assertEqual(self.client.get("/api/admin/users", headers=b).json()["users"], [])
        self.assertEqual(self.client.get(f"/api/admin/users/{sid}/results", headers=a).status_code, 200)
        self.assertEqual(self.client.get(f"/api/admin/users/{sid}/results", headers=b).status_code, 403)


if __name__ == "__main__":
    unittest.main()
