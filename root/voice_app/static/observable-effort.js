(() => {
    const form = document.getElementById('observable-upload-form');
    if (!form) return;
    const status = document.getElementById('observable-upload-status');
    const output = document.getElementById('observable-upload-report');
    const download = document.getElementById('observable-download');
    let report = null;
    form.addEventListener('submit', async event => {
        event.preventDefault();
        const button = form.querySelector('button');
        button.disabled = true;
        download.hidden = true;
        output.textContent = '';
        status.textContent = 'מנתח מדדים נצפים…';
        try {
            const headers = typeof getAuthHeaders === 'function' ? getAuthHeaders() : {};
            const response = await fetch('/api/observable-effort/recording', {method: 'POST', headers, body: new FormData(form)});
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'הניתוח נכשל');
            report = data;
            output.textContent = '';
            renderNeckAnalysis(output, data);
            status.textContent = 'הניתוח הושלם. הציונים מתארים התנהגות חיצונית נצפית; הסנכרון מבוסס על ההיסט שהוזן.';
            download.hidden = false;
        } catch (error) {
            status.textContent = error.message;
        } finally {
            button.disabled = false;
        }
    });
    download.addEventListener('click', () => {
        if (!report) return;
        const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = 'observable-effort.json';
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
})();
