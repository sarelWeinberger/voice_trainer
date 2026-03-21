/**
 * auth.js – JWT authentication layer for the new FastAPI user system.
 *
 * Login uses OAuth2 password flow (form-encoded).
 * Register sends JSON to /api/register.
 * After login, /api/me is fetched to get user details.
 * The JWT is stored in sessionStorage so each new tab requires fresh login.
 */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let _token = sessionStorage.getItem("jwt_token") || null;
let _currentUser = null;   // { username, full_name, is_admin }

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const authForms    = document.getElementById("auth-forms");
const authUsername  = document.getElementById("auth-username");
const authPassword = document.getElementById("auth-password");
const loginBtn     = document.getElementById("login-btn");
const registerBtn  = document.getElementById("register-btn");
const authError    = document.getElementById("auth-error");
const userBar      = document.getElementById("user-bar");
const userNameEl   = document.getElementById("user-name");
const logoutBtn    = document.getElementById("logout-btn");
const adminLink    = document.getElementById("admin-link");

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/**
 * Return Authorization headers for fetch calls.
 */
function getAuthHeaders() {
    if (!_token) return {};
    return { "Authorization": "Bearer " + _token };
}

// ---------------------------------------------------------------------------
// Core auth actions
// ---------------------------------------------------------------------------

/** POST /api/register (JSON) then auto-login. */
async function register() {
    const username = (authUsername?.value || "").trim();
    const password = authPassword?.value || "";

    if (!username || !password) return showAuthError("Enter username and password.");
    if (password.length < 8)   return showAuthError("Password must be at least 8 characters.");

    try {
        const resp = await fetch("/api/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });
        const data = await resp.json();

        if (!resp.ok) return showAuthError(data.detail || "Registration failed.");

        // Auto-login after successful registration
        await loginWithCredentials(username, password);
    } catch (err) {
        showAuthError("Network error: " + err.message);
    }
}

/** POST /api/login (form-encoded, OAuth2 password flow). */
async function login() {
    const username = (authUsername?.value || "").trim();
    const password = authPassword?.value || "";

    if (!username || !password) return showAuthError("Enter username and password.");

    await loginWithCredentials(username, password);
}

async function loginWithCredentials(username, password) {
    try {
        const formData = new URLSearchParams();
        formData.append("username", username);
        formData.append("password", password);

        const resp = await fetch("/api/login", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: formData,
        });
        const data = await resp.json();

        if (!resp.ok) return showAuthError(data.detail || "Login failed.");

        _token = data.access_token;
        sessionStorage.setItem("jwt_token", _token);

        // Fetch user profile
        await _fetchMe();
        showAuthError("");
    } catch (err) {
        showAuthError("Network error: " + err.message);
    }
}

/** Clear the stored token and update the UI. */
function logout() {
    _token = null;
    _currentUser = null;
    sessionStorage.removeItem("jwt_token");
    sessionStorage.removeItem("jwt_user");
    showLoggedOutUI();
    if (typeof onAuthSignedOut === "function") onAuthSignedOut();
}

// ---------------------------------------------------------------------------
// Session persistence
// ---------------------------------------------------------------------------

/** Fetch /api/me to get user details and save to localStorage. */
async function _fetchMe() {
    try {
        const resp = await fetch("/api/me", { headers: getAuthHeaders() });
        if (!resp.ok) { logout(); return; }
        const data = await resp.json();
        _currentUser = {
            username: data.username,
            full_name: data.full_name,
            is_admin: data.is_admin,
        };
        sessionStorage.setItem("jwt_user", JSON.stringify(_currentUser));
        _applyAuthState();
    } catch (_) {
        logout();
    }
}

/** Restore session from localStorage on page load. */
function _restoreSession() {
    const stored = sessionStorage.getItem("jwt_user");
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
    showLoggedInUI(_currentUser.username);
    if (_currentUser.is_admin && adminLink) {
        adminLink.hidden = false;
    }
    if (typeof onAuthReady === "function") onAuthReady(_currentUser);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function showLoggedInUI(name) {
    if (authForms) authForms.hidden = true;
    if (userBar) {
        userBar.hidden = false;
        if (userNameEl) userNameEl.textContent = name;
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

// Toggle password visibility
const togglePasswordBtn = document.getElementById("toggle-password-btn");
if (togglePasswordBtn && authPassword) {
    togglePasswordBtn.addEventListener("click", () => {
        const isHidden = authPassword.type === "password";
        authPassword.type = isHidden ? "text" : "password";
        togglePasswordBtn.textContent = isHidden ? "🙈" : "👁";
        togglePasswordBtn.title = isHidden ? "Hide password" : "Show password";
    });
}

// Allow Enter key to submit.
if (authPassword) {
    authPassword.addEventListener("keydown", (e) => {
        if (e.key === "Enter") login();
    });
}

// ---------------------------------------------------------------------------
// Bootstrap – restore session on page load
// ---------------------------------------------------------------------------

// Clean up any old localStorage tokens from previous versions
localStorage.removeItem("jwt_token");
localStorage.removeItem("jwt_user");

_restoreSession();
