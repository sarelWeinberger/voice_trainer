// Firebase Auth - handles login, register, logout, and token management

firebase.initializeApp(firebaseConfig);
const fbAuth = firebase.auth();

let currentUser = null;
let idToken = null;
let _tokenRefreshInterval = null;

// DOM refs (created by index.html / admin.html)
const authForms = document.getElementById("auth-forms");
const authEmail = document.getElementById("auth-email");
const authPassword = document.getElementById("auth-password");
const loginBtn = document.getElementById("login-btn");
const registerBtn = document.getElementById("register-btn");
const authError = document.getElementById("auth-error");
const userBar = document.getElementById("user-bar");
const userEmailEl = document.getElementById("user-email");
const logoutBtn = document.getElementById("logout-btn");
const adminLink = document.getElementById("admin-link");

// Auth state listener
fbAuth.onAuthStateChanged(async (user) => {
    currentUser = user;
    if (user) {
        idToken = await user.getIdToken();

        // Refresh token every 50 minutes (expires at 60)
        if (_tokenRefreshInterval) clearInterval(_tokenRefreshInterval);
        _tokenRefreshInterval = setInterval(async () => {
            idToken = await user.getIdToken(true);
        }, 50 * 60 * 1000);

        showLoggedInUI(user.email);
        createOrUpdateProfile();
        checkAdminStatus();

        // Hook for page-specific logic
        if (typeof onAuthReady === "function") onAuthReady(user);
    } else {
        idToken = null;
        if (_tokenRefreshInterval) clearInterval(_tokenRefreshInterval);
        showLoggedOutUI();

        if (typeof onAuthSignedOut === "function") onAuthSignedOut();
    }
});

// Register
async function register() {
    const email = authEmail.value.trim();
    const password = authPassword.value;
    if (!email || !password) return showAuthError("Enter email and password.");
    if (password.length < 6) return showAuthError("Password must be at least 6 characters.");

    try {
        await fbAuth.createUserWithEmailAndPassword(email, password);
        showAuthError("");
    } catch (err) {
        showAuthError(err.message);
    }
}

// Login
async function login() {
    const email = authEmail.value.trim();
    const password = authPassword.value;
    if (!email || !password) return showAuthError("Enter email and password.");

    try {
        await fbAuth.signInWithEmailAndPassword(email, password);
        showAuthError("");
    } catch (err) {
        showAuthError(err.message);
    }
}

// Logout
async function logout() {
    await fbAuth.signOut();
}

// Create or update user profile in Firestore via backend
async function createOrUpdateProfile() {
    try {
        await fetch("/api/user/profile", {
            method: "POST",
            headers: getAuthHeaders(),
        });
    } catch (err) {
        console.warn("Profile update failed:", err);
    }
}

// Check if current user is admin
async function checkAdminStatus() {
    try {
        const resp = await fetch("/api/user/role", {
            headers: getAuthHeaders(),
        });
        const data = await resp.json();
        if (data.role === "admin" && adminLink) {
            adminLink.hidden = false;
        }
    } catch (err) {
        console.warn("Admin check failed:", err);
    }
}

// Returns auth headers for fetch calls
function getAuthHeaders() {
    if (!idToken) return {};
    return { "Authorization": "Bearer " + idToken };
}

// UI helpers
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

// Event listeners
if (loginBtn) loginBtn.addEventListener("click", login);
if (registerBtn) registerBtn.addEventListener("click", register);
if (logoutBtn) logoutBtn.addEventListener("click", logout);

// Allow Enter key to submit
if (authPassword) {
    authPassword.addEventListener("keydown", (e) => {
        if (e.key === "Enter") login();
    });
}
