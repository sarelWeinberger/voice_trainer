/**
 * admin.js – Admin Dashboard logic.
 *
 * Works with the JWT-based auth layer in auth.js (no Firebase).
 * Two tabs: Users (user list + role management) and Activity Log.
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRADE_COLORS_ADMIN = ["#3ddc84", "#a8e063", "#ffb74d", "#ff8a50", "#ff4f5e"];

// Activity log pagination
const ACTIVITY_PAGE_SIZE = 50;
let _activityOffset = 0;
let _activityUserFilter = null;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const adminContent  = document.getElementById("admin-content");
const adminDenied   = document.getElementById("admin-denied");
const usersTbody    = document.getElementById("users-tbody");
const detailPanel   = document.getElementById("user-detail-panel");
const detailTitle   = document.getElementById("detail-title");
const detailContent = document.getElementById("user-detail-content");
const detailCloseBtn = document.getElementById("detail-close-btn");
const activityTbody = document.getElementById("activity-tbody");

// Tab elements
const tabBtns   = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

// Activity log controls
const activityUserFilter = document.getElementById("activity-user-filter");
const activityFilterBtn  = document.getElementById("activity-filter-btn");
const activityClearBtn   = document.getElementById("activity-clear-btn");
const activityPrevBtn    = document.getElementById("activity-prev-btn");
const activityNextBtn    = document.getElementById("activity-next-btn");
const activityPageInfo   = document.getElementById("activity-page-info");

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
        const target = btn.dataset.tab;

        tabBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        tabPanels.forEach((panel) => {
            panel.hidden = panel.id !== `tab-${target}`;
        });

        if (target === "activity") loadActivityLog();
    });
});

// ---------------------------------------------------------------------------
// Auth hooks (called by auth.js)
// ---------------------------------------------------------------------------

/**
 * onAuthReady is called by auth.js after a successful login / session restore.
 * We check the role and either reveal the admin UI or show the denied panel.
 */
async function onAuthReady(user) {
    if (user.role === "admin") {
        adminContent.hidden = false;
        adminDenied.hidden  = true;
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
// Users tab
// ---------------------------------------------------------------------------

async function loadUsers() {
    usersTbody.innerHTML = '<tr><td colspan="10">Loading…</td></tr>';

    try {
        const resp = await fetch("/api/admin/users", { headers: getAuthHeaders() });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        usersTbody.innerHTML = "";

        if (!data.users || data.users.length === 0) {
            usersTbody.innerHTML = '<tr><td colspan="10">No users found.</td></tr>';
            return;
        }

        for (const user of data.users) {
            const tr = document.createElement("tr");
            const scores     = user.latest_analysis ? user.latest_analysis.scores : {};
            const registered = user.created_at
                ? new Date(user.created_at).toLocaleDateString()
                : "N/A";
            const lastLogin  = user.last_login_at
                ? new Date(user.last_login_at).toLocaleString()
                : "Never";

            tr.innerHTML = `
                <td>${escapeHtml(user.email)}</td>
                <td><span class="role-badge role-${user.role}">${user.role}</span></td>
                <td>${registered}</td>
                <td>${lastLogin}</td>
                ${scoreCell(scores.G)}
                ${scoreCell(scores.R)}
                ${scoreCell(scores.B)}
                ${scoreCell(scores.A)}
                ${scoreCell(scores.S)}
                <td class="actions-cell">
                    <button class="small-btn view-btn"
                        data-uid="${user.id}"
                        data-email="${escapeHtml(user.email)}">View</button>
                    <button class="small-btn role-btn"
                        data-uid="${user.id}"
                        data-role="${user.role}">
                        ${user.role === "admin" ? "Demote" : "Promote"}
                    </button>
                </td>
            `;
            usersTbody.appendChild(tr);
        }

        document.querySelectorAll(".view-btn").forEach((btn) => {
            btn.addEventListener("click", () =>
                viewUserAnalyses(btn.dataset.uid, btn.dataset.email));
        });

        document.querySelectorAll(".role-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const newRole = btn.dataset.role === "admin" ? "user" : "admin";
                toggleRole(btn.dataset.uid, newRole);
            });
        });
    } catch (err) {
        usersTbody.innerHTML = `<tr><td colspan="10">Error: ${escapeHtml(err.message)}</td></tr>`;
    }
}

function scoreCell(value) {
    if (value === undefined || value === null) return '<td class="score-cell">-</td>';
    const color = GRADE_COLORS_ADMIN[value - 1] || "#888";
    return `<td class="score-cell" style="color:${color};font-weight:bold">${value}</td>`;
}

async function viewUserAnalyses(uid, email) {
    detailPanel.hidden = false;
    detailTitle.textContent = `Analysis History: ${email}`;
    detailContent.innerHTML = "<p>Loading…</p>";

    try {
        const resp = await fetch(`/api/admin/users/${uid}/analyses`, {
            headers: getAuthHeaders(),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (!data.analyses || data.analyses.length === 0) {
            detailContent.innerHTML = "<p>No analyses yet.</p>";
            return;
        }

        let html = '<table class="admin-table"><thead><tr>';
        html += "<th>Date</th><th>G</th><th>R</th><th>B</th><th>A</th><th>S</th>";
        html += "<th>F0</th><th>Jitter%</th><th>HNR</th>";
        html += "</tr></thead><tbody>";

        for (const a of data.analyses) {
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
        detailContent.innerHTML = `<p>Error: ${escapeHtml(err.message)}</p>`;
    }
}

async function toggleRole(uid, newRole) {
    if (!confirm(`Set this user's role to "${newRole}"?`)) return;
    try {
        const resp = await fetch(`/api/admin/users/${uid}/role`, {
            method: "PUT",
            headers: { ...getAuthHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ role: newRole }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        loadUsers();
    } catch (err) {
        alert("Error: " + err.message);
    }
}

if (detailCloseBtn) {
    detailCloseBtn.addEventListener("click", () => { detailPanel.hidden = true; });
}

// ---------------------------------------------------------------------------
// Activity Log tab
// ---------------------------------------------------------------------------

async function loadActivityLog() {
    if (!activityTbody) return;
    activityTbody.innerHTML = '<tr><td colspan="6">Loading…</td></tr>';

    const params = new URLSearchParams({
        limit:  ACTIVITY_PAGE_SIZE,
        offset: _activityOffset,
    });
    if (_activityUserFilter) params.set("user_id", _activityUserFilter);

    try {
        const resp = await fetch(`/api/admin/activity-log?${params}`, {
            headers: getAuthHeaders(),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        activityTbody.innerHTML = "";

        if (!data.logs || data.logs.length === 0) {
            activityTbody.innerHTML = '<tr><td colspan="6">No activity records found.</td></tr>';
            _updateActivityPagination(0);
            return;
        }

        for (const log of data.logs) {
            const ts      = log.timestamp ? new Date(log.timestamp).toLocaleString() : "N/A";
            const userStr = log.user_email
                ? `${escapeHtml(log.user_email)} <span class="text-muted">(${log.user_id})</span>`
                : log.user_id ? `ID ${log.user_id}` : "<em>anonymous</em>";
            const details = log.details && Object.keys(log.details).length
                ? `<code>${escapeHtml(JSON.stringify(log.details))}</code>`
                : "—";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${ts}</td>
                <td>${userStr}</td>
                <td><span class="action-badge action-${log.action}">${escapeHtml(log.action)}</span></td>
                <td>${log.resource ? escapeHtml(log.resource) : "—"}</td>
                <td>${log.ip_address ? escapeHtml(log.ip_address) : "—"}</td>
                <td class="details-cell">${details}</td>
            `;
            activityTbody.appendChild(tr);
        }

        _updateActivityPagination(data.logs.length);
    } catch (err) {
        activityTbody.innerHTML = `<tr><td colspan="6">Error: ${escapeHtml(err.message)}</td></tr>`;
    }
}

function _updateActivityPagination(count) {
    const page = Math.floor(_activityOffset / ACTIVITY_PAGE_SIZE) + 1;
    if (activityPageInfo) activityPageInfo.textContent = `Page ${page}`;
    if (activityPrevBtn)  activityPrevBtn.disabled = _activityOffset === 0;
    if (activityNextBtn)  activityNextBtn.disabled = count < ACTIVITY_PAGE_SIZE;
}

if (activityFilterBtn) {
    activityFilterBtn.addEventListener("click", () => {
        const val = activityUserFilter?.value.trim();
        _activityUserFilter = val ? parseInt(val, 10) : null;
        _activityOffset = 0;
        loadActivityLog();
    });
}

if (activityClearBtn) {
    activityClearBtn.addEventListener("click", () => {
        _activityUserFilter = null;
        _activityOffset = 0;
        if (activityUserFilter) activityUserFilter.value = "";
        loadActivityLog();
    });
}

if (activityPrevBtn) {
    activityPrevBtn.addEventListener("click", () => {
        _activityOffset = Math.max(0, _activityOffset - ACTIVITY_PAGE_SIZE);
        loadActivityLog();
    });
}

if (activityNextBtn) {
    activityNextBtn.addEventListener("click", () => {
        _activityOffset += ACTIVITY_PAGE_SIZE;
        loadActivityLog();
    });
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
}
