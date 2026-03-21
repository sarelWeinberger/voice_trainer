/**
 * admin.js – Admin Dashboard logic.
 *
 * Works with the JWT-based auth layer in auth.js.
 * Shows a user list with latest GRBAS scores and role management.
 */

"use strict";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRADE_COLORS_ADMIN = ["#3ddc84", "#a8e063", "#ffb74d", "#ff8a50", "#ff4f5e"];

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const adminContent   = document.getElementById("admin-content");
const adminDenied    = document.getElementById("admin-denied");
const usersTbody     = document.getElementById("users-tbody");
const detailPanel    = document.getElementById("user-detail-panel");
const detailTitle    = document.getElementById("detail-title");
const detailContent  = document.getElementById("user-detail-content");
const detailCloseBtn = document.getElementById("detail-close-btn");

// ---------------------------------------------------------------------------
// Auth hooks (called by auth.js)
// ---------------------------------------------------------------------------

async function onAuthReady(user) {
    if (user.is_admin) {
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
// Users
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

            tr.innerHTML = `
                <td>${escapeHtml(user.username)}</td>
                <td>${escapeHtml(user.full_name || "")}</td>
                <td><span class="role-badge role-${user.is_admin ? "admin" : "user"}">${user.is_admin ? "admin" : "user"}</span></td>
                <td>${registered}</td>
                ${scoreCell(scores.G)}
                ${scoreCell(scores.R)}
                ${scoreCell(scores.B)}
                ${scoreCell(scores.A)}
                ${scoreCell(scores.S)}
                <td class="actions-cell">
                    <button class="small-btn view-btn"
                        data-uid="${user.id}"
                        data-username="${escapeHtml(user.username)}">View</button>
                    <button class="small-btn role-btn"
                        data-uid="${user.id}"
                        data-is-admin="${user.is_admin}">
                        ${user.is_admin ? "Demote" : "Promote"}
                    </button>
                </td>
            `;
            usersTbody.appendChild(tr);
        }

        document.querySelectorAll(".view-btn").forEach((btn) => {
            btn.addEventListener("click", () =>
                viewUserResults(btn.dataset.uid, btn.dataset.username));
        });

        document.querySelectorAll(".role-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const newAdmin = btn.dataset.isAdmin !== "true";
                toggleRole(btn.dataset.uid, newAdmin);
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

async function viewUserResults(uid, username) {
    detailPanel.hidden = false;
    detailTitle.textContent = `Analysis History: ${username}`;
    detailContent.innerHTML = "<p>Loading…</p>";

    try {
        const resp = await fetch(`/api/admin/users/${uid}/results`, {
            headers: getAuthHeaders(),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        if (!data.results || data.results.length === 0) {
            detailContent.innerHTML = "<p>No analyses yet.</p>";
            return;
        }

        let html = '<table class="admin-table"><thead><tr>';
        html += "<th>Date</th><th>G</th><th>R</th><th>B</th><th>A</th><th>S</th>";
        html += "<th>F0</th><th>Jitter%</th><th>HNR</th>";
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
        detailContent.innerHTML = `<p>Error: ${escapeHtml(err.message)}</p>`;
    }
}

async function toggleRole(uid, newAdmin) {
    const action = newAdmin ? "promote to admin" : "demote to user";
    if (!confirm(`Are you sure you want to ${action}?`)) return;
    try {
        const resp = await fetch(`/api/admin/users/${uid}/role`, {
            method: "PUT",
            headers: { ...getAuthHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ is_admin: newAdmin }),
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
// Utility
// ---------------------------------------------------------------------------

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
}
