/**
 * admin.js – Admin Dashboard logic with 3-level role system.
 *
 * Role 1: Super Admin – sees all users grouped by class, can promote/demote
 * Role 2: Admin – sees users in their class only
 * Role 3: User – no admin access
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRADE_COLORS_ADMIN = ["#3ddc84", "#a8e063", "#ffb74d", "#ff8a50", "#ff4f5e"];
const ROLE_LABELS = { 1: "סופר אדמין", 2: "מנהל", 3: "משתמש" };

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const adminContent     = document.getElementById("admin-content");
const adminDenied      = document.getElementById("admin-denied");
const usersTbody       = document.getElementById("users-tbody");
const usersTitle       = document.getElementById("users-title");
const detailPanel      = document.getElementById("user-detail-panel");
const detailTitle      = document.getElementById("detail-title");
const detailContent    = document.getElementById("user-detail-content");
const detailCloseBtn   = document.getElementById("detail-close-btn");
const classTabsEl      = document.getElementById("class-tabs");
const classCodeBox     = document.getElementById("admin-class-code-box");
const classCodeValueEl = document.getElementById("admin-class-code-value");
const copyClassCodeBtn = document.getElementById("copy-class-code");

let _allUsers = [];
let _currentFilter = "all";
let _myRole = 3;

// ---------------------------------------------------------------------------
// Auth hooks (called by auth.js)
// ---------------------------------------------------------------------------

async function onAuthReady(user) {
    _myRole = user.role || 3;

    if (user.is_admin || _myRole <= 2) {
        adminContent.hidden = false;
        adminDenied.hidden  = true;

        // Show class code
        if (user.class_code && classCodeBox) {
            classCodeBox.hidden = false;
            classCodeValueEl.textContent = user.class_code;
        }

        // Show class filter tabs only for super admin
        if (_myRole === 1 && classTabsEl) {
            classTabsEl.hidden = false;
        }

        loadUsers();
    } else {
        adminContent.hidden = true;
        adminDenied.hidden  = false;
    }
}

function onAuthSignedOut() {
    adminContent.hidden  = true;
    adminDenied.hidden   = true;
    detailPanel.hidden   = true;
}

// ---------------------------------------------------------------------------
// Copy class code
// ---------------------------------------------------------------------------

if (copyClassCodeBtn) {
    copyClassCodeBtn.addEventListener("click", () => {
        const code = classCodeValueEl?.textContent || "";
        if (code) {
            navigator.clipboard.writeText(code).then(() => {
                copyClassCodeBtn.textContent = "✓";
                setTimeout(() => { copyClassCodeBtn.textContent = "📋"; }, 1500);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Class filter tabs (super admin)
// ---------------------------------------------------------------------------

function buildClassTabs(users) {
    if (!classTabsEl || _myRole !== 1) return;

    // Collect unique class codes from admins
    const adminClasses = new Map(); // code -> admin username
    for (const u of users) {
        if (u.class_code && u.role <= 2) {
            adminClasses.set(u.class_code, u.username);
        }
    }

    // Remove old dynamic tabs
    classTabsEl.querySelectorAll(".class-tab-dynamic").forEach(el => el.remove());

    for (const [code, name] of adminClasses) {
        const btn = document.createElement("button");
        btn.className = "class-tab class-tab-dynamic";
        btn.dataset.filter = "class:" + code;
        btn.textContent = `${name} (${code})`;
        classTabsEl.appendChild(btn);
    }

    // Bind events
    classTabsEl.querySelectorAll(".class-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            classTabsEl.querySelectorAll(".class-tab").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            _currentFilter = btn.dataset.filter;
            renderUsers();
        });
    });
}

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------

async function loadUsers() {
    usersTbody.innerHTML = '<tr><td colspan="11">טוען…</td></tr>';

    try {
        const resp = await fetch("/api/admin/users", { headers: getAuthHeaders() });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        _allUsers = data.users || [];

        if (_myRole === 1) {
            buildClassTabs(_allUsers);
        }

        renderUsers();
    } catch (err) {
        usersTbody.innerHTML = `<tr><td colspan="11">שגיאה: ${escapeHtml(err.message)}</td></tr>`;
    }
}

function renderUsers() {
    let users = _allUsers;

    // Apply filter for super admin
    if (_myRole === 1) {
        if (_currentFilter === "no-class") {
            users = _allUsers.filter(u => !u.joined_class_code && u.role === 3);
        } else if (_currentFilter.startsWith("class:")) {
            const code = _currentFilter.split(":")[1];
            users = _allUsers.filter(u => u.joined_class_code === code || u.class_code === code);
        }
    }

    usersTbody.innerHTML = "";

    if (users.length === 0) {
        usersTbody.innerHTML = '<tr><td colspan="11">לא נמצאו משתמשים.</td></tr>';
        return;
    }

    for (const user of users) {
        const tr = document.createElement("tr");
        const scores     = user.latest_analysis ? user.latest_analysis.scores : {};
        const registered = user.created_at
            ? new Date(user.created_at).toLocaleDateString()
            : "N/A";

        const roleLabel = ROLE_LABELS[user.role] || "משתמש";
        const roleCls   = user.role === 1 ? "role-super" : (user.role === 2 ? "role-admin" : "role-user");
        const classInfo = user.joined_class_code || user.class_code || "-";

        tr.innerHTML = `
            <td>${escapeHtml(user.username)}</td>
            <td>${escapeHtml(user.full_name || "")}</td>
            <td><span class="role-badge ${roleCls}">${roleLabel}</span></td>
            <td>${escapeHtml(classInfo)}</td>
            <td>${registered}</td>
            ${scoreCell(scores.G)}
            ${scoreCell(scores.R)}
            ${scoreCell(scores.B)}
            ${scoreCell(scores.A)}
            ${scoreCell(scores.S)}
            <td class="actions-cell">
                <button class="small-btn view-btn"
                    data-uid="${user.id}"
                    data-username="${escapeHtml(user.username)}">צפה</button>
                ${_myRole === 1 ? buildRoleActions(user) : ""}
            </td>
        `;
        usersTbody.appendChild(tr);
    }

    // Bind view buttons
    document.querySelectorAll(".view-btn").forEach((btn) => {
        btn.addEventListener("click", () =>
            viewUserResults(btn.dataset.uid, btn.dataset.username));
    });

    // Bind role buttons (super admin only)
    if (_myRole === 1) {
        document.querySelectorAll(".role-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const newRole = parseInt(btn.dataset.newRole, 10);
                changeRole(btn.dataset.uid, newRole);
            });
        });
    }
}

function buildRoleActions(user) {
    // Super admin can promote/demote
    const actions = [];
    if (user.role === 3) {
        actions.push(`<button class="small-btn role-btn" data-uid="${user.id}" data-new-role="2">קדם למנהל</button>`);
    }
    if (user.role === 2) {
        actions.push(`<button class="small-btn role-btn" data-uid="${user.id}" data-new-role="1">קדם לסופר</button>`);
        actions.push(`<button class="small-btn role-btn" data-uid="${user.id}" data-new-role="3">הורד למשתמש</button>`);
    }
    if (user.role === 1) {
        actions.push(`<button class="small-btn role-btn" data-uid="${user.id}" data-new-role="2">הורד למנהל</button>`);
    }
    return actions.join(" ");
}

function scoreCell(value) {
    if (value === undefined || value === null) return '<td class="score-cell">-</td>';
    const color = GRADE_COLORS_ADMIN[value - 1] || "#888";
    return `<td class="score-cell" style="color:${color};font-weight:bold">${value}</td>`;
}

async function viewUserResults(uid, username) {
    detailPanel.hidden = false;
    detailTitle.textContent = `היסטוריית ניתוחים: ${username}`;
    detailContent.innerHTML = "<p>טוען…</p>";

    try {
        const resp = await fetch(`/api/admin/users/${uid}/results`, {
            headers: getAuthHeaders(),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (!data.results || data.results.length === 0) {
            detailContent.innerHTML = "<p>אין ניתוחים עדיין.</p>";
            return;
        }

        let html = '<table class="admin-table"><thead><tr>';
        html += "<th>תאריך</th><th>G</th><th>R</th><th>B</th><th>A</th><th>S</th>";
        html += "<th>F0</th><th>רעידה%</th><th>HNR</th>";
        html += "</tr></thead><tbody>";

        for (const a of data.results) {
            const ts = a.timestamp ? new Date(a.timestamp).toLocaleString() : "N/A";
            const s  = a.scores   || {};
            const f  = a.features || {};
            html += `<tr>
                <td>${ts}</td>
                ${scoreCell(s.G)}${scoreCell(s.R)}${scoreCell(s.B)}${scoreCell(s.A)}${scoreCell(s.S)}
                <td>${f.f0_mean        ?? "-"}</td>
                <td>${f.jitter_percent ?? "-"}</td>
                <td>${f.hnr            ?? "-"}</td>
            </tr>`;
        }
        html += "</tbody></table>";
        detailContent.innerHTML = html;
    } catch (err) {
        detailContent.innerHTML = `<p>שגיאה: ${escapeHtml(err.message)}</p>`;
    }
}

async function changeRole(uid, newRole) {
    const label = ROLE_LABELS[newRole] || "משתמש";
    if (!confirm(`לשנות את התפקיד ל${label}?`)) return;

    try {
        const resp = await fetch(`/api/admin/users/${uid}/role`, {
            method: "PUT",
            headers: { ...getAuthHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ is_admin: newRole <= 2, role: newRole }),
        });
        if (!resp.ok) {
            const data = await resp.json();
            throw new Error(data.detail || `HTTP ${resp.status}`);
        }
        loadUsers();
    } catch (err) {
        alert("שגיאה: " + err.message);
    }
}

if (detailCloseBtn) {
    detailCloseBtn.addEventListener("click", () => { detailPanel.hidden = true; });
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
}
