/**
 * auth.js – Custom JWT authentication layer (replaces Firebase Auth).
 *
 * The JWT returned by the server is stored in localStorage under "jwt_token".
 * Every authenticated API call adds an  Authorization: Bearer <token>  header.
 *
 * Security notes:
 *  - Passwords are transmitted over HTTPS (enforce at the server/proxy level).
 *  - The token is stored in localStorage; for highest security, consider
 *    httpOnly cookies, but localStorage is acceptable for this use-case.
 *  - Tokens expire after 24 h; the user must log in again to get a fresh one.
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _token = localStorage.getItem("jwt_token") || null;
let _currentUser = null;   // { email, role }

// ---------------------------------------------------------------------------
// DOM references (declared as `let` so pages without these elements don't crash)
// ---------------------------------------------------------------------------

const authForms   = document.getElementById("auth-forms");
const authEmail   = document.getElementById("auth-email");
const authPassword = document.getElementById("auth-password");
const loginBtn    = document.getElementById("login-btn");
const registerBtn = document.getElementById("register-btn");
const authError   = document.getElementById("auth-error");
const userBar     = document.getElementById("user-bar");
const userEmailEl = document.getElementById("user-email");
const logoutBtn   = document.getElementById("logout-btn");
const adminLink   = document.getElementById("admin-link");

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/**
 * Return Authorization headers for fetch calls.
 * If the user is not authenticated, returns an empty object.
 */
function getAuthHeaders() {
    if (!_token) return {};
    return { "Authorization": "Bearer " + _token };
}

// ---------------------------------------------------------------------------
// Core auth actions
// ---------------------------------------------------------------------------

/** POST /api/auth/register and log in on success. */
async function register() {
    const email    = (authEmail?.value || "").trim();
    const password = authPassword?.value || "";

    if (!email || !password) return showAuthError("Enter email and password.");
    if (password.length < 8)  return showAuthError("Password must be at least 8 characters.");

    try {
        const resp = await fetch("/api/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        const data = await resp.json();

        if (!resp.ok) return showAuthError(data.error || "Registration failed.");

        _saveSession(data);
        showAuthError("");
    } catch (err) {
        showAuthError("Network error: " + err.message);
    }
}

/** POST /api/auth/login and store the returned JWT. */
async function login() {
    const email    = (authEmail?.value || "").trim();
    const password = authPassword?.value || "";

    if (!email || !password) return showAuthError("Enter email and password.");

    try {
        const resp = await fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        const data = await resp.json();

        if (!resp.ok) return showAuthError(data.error || "Login failed.");

        _saveSession(data);
        showAuthError("");
    } catch (err) {
        showAuthError("Network error: " + err.message);
    }
}

/** Clear the stored token and update the UI. */
function logout() {
    _token = null;
    _currentUser = null;
    localStorage.removeItem("jwt_token");
    localStorage.removeItem("jwt_user");
    showLoggedOutUI();
    if (typeof onAuthSignedOut === "function") onAuthSignedOut();
}

// ---------------------------------------------------------------------------
// Session persistence
// ---------------------------------------------------------------------------

/** Persist token + user info and trigger the authenticated UI. */
function _saveSession(data) {
    _token = data.token;
    _currentUser = { email: data.email, role: data.role };
    localStorage.setItem("jwt_token", _token);
    localStorage.setItem("jwt_user", JSON.stringify(_currentUser));
    _applyAuthState();
}

/** Restore session from localStorage on page load (no round-trip needed). */
function _restoreSession() {
    const stored = localStorage.getItem("jwt_user");
    if (_token && stored) {
        try {
            _currentUser = JSON.parse(stored);
            _applyAuthState();
        } catch (_) {
            logout();
        }
    } else {
        showLoggedOutUI();
    }
}

/** Update UI and call page-specific hook. */
function _applyAuthState() {
    if (!_currentUser) { showLoggedOutUI(); return; }
    showLoggedInUI(_currentUser.email);
    if (_currentUser.role === "admin" && adminLink) {
        adminLink.hidden = false;
    }
    if (typeof onAuthReady === "function") onAuthReady(_currentUser);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function showLoggedInUI(email) {
    if (authForms) authForms.hidden = true;
    if (userBar) {
        userBar.hidden = false;
        if (userEmailEl) userEmailEl.textContent = email;
    }
}

function showLoggedOutUI() {
    if (authForms) authForms.hidden = false;
    if (userBar) userBar.hidden = true;
    if (adminLink) adminLink.hidden = true;
}

function showAuthError(msg) {
    if (authError) authError.textContent = msg;
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

if (loginBtn)    loginBtn.addEventListener("click", login);
if (registerBtn) registerBtn.addEventListener("click", register);
if (logoutBtn)   logoutBtn.addEventListener("click", logout);

// Allow Enter key to submit.
if (authPassword) {
    authPassword.addEventListener("keydown", (e) => {
        if (e.key === "Enter") login();
    });
}

// ---------------------------------------------------------------------------
// Bootstrap – restore session on page load
// ---------------------------------------------------------------------------

_restoreSession();
