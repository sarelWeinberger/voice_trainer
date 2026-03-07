// Admin Dashboard Logic

const GRADE_COLORS_ADMIN = ["#3ddc84", "#a8e063", "#ffb74d", "#ff8a50", "#ff4f5e"];

const adminContent = document.getElementById("admin-content");
const adminDenied = document.getElementById("admin-denied");
const usersTbody = document.getElementById("users-tbody");
const detailPanel = document.getElementById("user-detail-panel");
const detailTitle = document.getElementById("detail-title");
const detailContent = document.getElementById("user-detail-content");

// Called by auth.js when user is authenticated
async function onAuthReady(user) {
    try {
        const resp = await fetch("/api/user/role", { headers: getAuthHeaders() });
        const data = await resp.json();

        if (data.role === "admin") {
            adminContent.hidden = false;
            adminDenied.hidden = true;
            loadUsers();
        } else {
            adminContent.hidden = true;
            adminDenied.hidden = false;
        }
    } catch (err) {
        adminContent.hidden = true;
        adminDenied.hidden = false;
    }
}

function onAuthSignedOut() {
    adminContent.hidden = true;
    adminDenied.hidden = true;
    detailPanel.hidden = true;
}

async function loadUsers() {
    usersTbody.innerHTML = '<tr><td colspan="9">Loading...</td></tr>';

    try {
        const resp = await fetch("/api/admin/users", { headers: getAuthHeaders() });
        if (!resp.ok) throw new Error("Failed to load users");
        const data = await resp.json();

        usersTbody.innerHTML = "";

        if (data.users.length === 0) {
            usersTbody.innerHTML = '<tr><td colspan="9">No users found.</td></tr>';
            return;
        }

        for (const user of data.users) {
            const tr = document.createElement("tr");
            const scores = user.latestAnalysis ? user.latestAnalysis.scores : {};
            const registered = user.createdAt
                ? new Date(user.createdAt).toLocaleDateString()
                : "N/A";

            tr.innerHTML = `
                <td>${escapeHtml(user.email)}</td>
                <td><span class="role-badge role-${user.role}">${user.role}</span></td>
                <td>${registered}</td>
                ${scoreCell(scores.G)}
                ${scoreCell(scores.R)}
                ${scoreCell(scores.B)}
                ${scoreCell(scores.A)}
                ${scoreCell(scores.S)}
                <td class="actions-cell">
                    <button class="small-btn view-btn" data-uid="${user.uid}" data-email="${escapeHtml(user.email)}">View</button>
                    <button class="small-btn role-btn" data-uid="${user.uid}" data-role="${user.role}">
                        ${user.role === "admin" ? "Demote" : "Promote"}
                    </button>
                </td>
            `;
            usersTbody.appendChild(tr);
        }

        // Attach event listeners
        document.querySelectorAll(".view-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                viewUserAnalyses(btn.dataset.uid, btn.dataset.email);
            });
        });

        document.querySelectorAll(".role-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const newRole = btn.dataset.role === "admin" ? "user" : "admin";
                toggleRole(btn.dataset.uid, newRole);
            });
        });
    } catch (err) {
        usersTbody.innerHTML = `<tr><td colspan="9">Error: ${err.message}</td></tr>`;
    }
}

function scoreCell(value) {
    if (value === undefined || value === null) {
        return '<td class="score-cell">-</td>';
    }
    const color = GRADE_COLORS_ADMIN[value - 1] || "#888";
    return `<td class="score-cell" style="color:${color};font-weight:bold">${value}</td>`;
}

async function viewUserAnalyses(uid, email) {
    detailPanel.hidden = false;
    detailTitle.textContent = `Analysis History: ${email}`;
    detailContent.innerHTML = "<p>Loading...</p>";

    try {
        const resp = await fetch(`/api/admin/users/${uid}/analyses`, {
            headers: getAuthHeaders(),
        });
        if (!resp.ok) throw new Error("Failed to load analyses");
        const data = await resp.json();

        if (data.analyses.length === 0) {
            detailContent.innerHTML = "<p>No analyses yet.</p>";
            return;
        }

        let html = '<table class="admin-table"><thead><tr>';
        html += "<th>Date</th><th>G</th><th>R</th><th>B</th><th>A</th><th>S</th>";
        html += "<th>F0</th><th>Jitter%</th><th>HNR</th>";
        html += "</tr></thead><tbody>";

        for (const a of data.analyses) {
            const ts = a.timestamp
                ? new Date(a.timestamp).toLocaleString()
                : "N/A";
            const s = a.scores || {};
            const f = a.features || {};

            html += "<tr>";
            html += `<td>${ts}</td>`;
            html += scoreCell(s.G);
            html += scoreCell(s.R);
            html += scoreCell(s.B);
            html += scoreCell(s.A);
            html += scoreCell(s.S);
            html += `<td>${f.f0_mean || "-"}</td>`;
            html += `<td>${f.jitter_percent || "-"}</td>`;
            html += `<td>${f.hnr || "-"}</td>`;
            html += "</tr>";
        }

        html += "</tbody></table>";
        detailContent.innerHTML = html;
    } catch (err) {
        detailContent.innerHTML = `<p>Error: ${err.message}</p>`;
    }
}

async function toggleRole(uid, newRole) {
    if (!confirm(`Set this user's role to "${newRole}"?`)) return;

    try {
        const resp = await fetch(`/api/admin/users/${uid}/role`, {
            method: "PUT",
            headers: {
                ...getAuthHeaders(),
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ role: newRole }),
        });

        if (!resp.ok) throw new Error("Failed to update role");
        loadUsers();
    } catch (err) {
        alert("Error: " + err.message);
    }
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
