/**
 * chat.js — Main chat UI controller.
 * Manages session state, message rendering, sending, and orchestrates
 * file uploads and confirmations.
 */

(function () {
  'use strict';

  // ---- State ----
  let sessionId = null;
  let lastResultId = null;
  let fsmState = 'idle';

  const messagesEl = document.getElementById('messages');
  const inputEl = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const typingEl = document.getElementById('typing-indicator');
  const confirmBar = document.getElementById('confirm-bar');
  const confirmYes = document.getElementById('confirm-yes-btn');
  const confirmNo = document.getElementById('confirm-no-btn');
  const uploadPanel = document.getElementById('upload-panel');
  const newSessionBtn = document.getElementById('new-session-btn');

  // ---- Init ----
  appendAssistantMessage(
    "Hello! I'm your Clinical Analytics Assistant. I can help you with:\n\n" +
    "1. **Site List Merger** — Merge CRO and sponsor clinical site lists\n" +
    "2. **Trial Benchmarking** — Benchmark trials by indication, age group, and phase\n" +
    "3. **Drug Reimbursement** — Assess reimbursement outlook by country\n" +
    "4. **Enrollment Forecasting** — Forecast enrollment and site activation curves\n\n" +
    "What would you like to do today?"
  );

  // ---- Event listeners ----
  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  confirmYes.addEventListener('click', function () {
    hideConfirmBar();
    sendConfirmation(true);
  });

  confirmNo.addEventListener('click', function () {
    hideConfirmBar();
    sendConfirmation(false);
  });

  newSessionBtn.addEventListener('click', function () {
    sessionId = null;
    lastResultId = null;
    fsmState = 'idle';
    messagesEl.innerHTML = '';
    hideConfirmBar();
    uploadPanel.classList.add('hidden');
    appendAssistantMessage("New session started. How can I help you?");
  });

  // ---- File upload drag & drop ----
  setupDropZone('cro-drop-zone', 'cro-file-input', 'cro_file', 'cro-file-status');
  setupDropZone('sponsor-drop-zone', 'sponsor-file-input', 'sponsor_file', 'sponsor-file-status');

  // ---- Core functions ----

  function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';
    sendChatText(text);
  }

  function sendChatText(text) {
    appendUserMessage(text);
    setLoading(true);
    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    })
      .then(r => r.json())
      .then(handleResponse)
      .catch(handleError)
      .finally(() => setLoading(false));
  }

  function sendConfirmation(confirmed, editParams) {
    setLoading(true);
    fetch('/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        confirmed: confirmed,
        edit_params: editParams || null,
      }),
    })
      .then(r => r.json())
      .then(handleResponse)
      .catch(handleError)
      .finally(() => setLoading(false));
  }

  function handleResponse(data) {
    if (data.session_id) sessionId = data.session_id;
    if (data.result_id) lastResultId = data.result_id;
    fsmState = data.fsm_state || 'idle';

    // Show or hide upload panel based on active skill
    if (data.active_skill === 'site_list_merger') {
      uploadPanel.classList.remove('hidden');
    }

    // Render assistant message
    if (data.message) {
      appendAssistantMessage(data.message, {
        tableData: data.table_data,
        tableColumns: data.table_columns,
        chartJson: data.chart_json,
        resultId: data.result_id,
      });
    }

    // Show confirmation bar if pending
    if (fsmState === 'confirmation_pending') {
      showConfirmBar();
    } else {
      hideConfirmBar();
    }
  }

  function handleError(err) {
    appendAssistantMessage('Sorry, a network error occurred. Please try again.');
    console.error(err);
  }

  // ---- Message rendering ----

  function appendUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function appendAssistantMessage(text, extras) {
    extras = extras || {};
    const div = document.createElement('div');
    div.className = 'message assistant';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = renderMarkdown(text);
    div.appendChild(bubble);

    // Render table if present
    if (extras.tableData && extras.tableData.length > 0) {
      const tableWrapper = buildTable(extras.tableData, extras.tableColumns);
      div.appendChild(tableWrapper);

      // Export button
      if (extras.resultId) {
        const exportBtn = document.createElement('button');
        exportBtn.className = 'export-btn';
        exportBtn.textContent = '⬇ Export to Dataiku dataset';
        exportBtn.addEventListener('click', function () {
          window.openExportModal(extras.resultId);
        });
        div.appendChild(exportBtn);
      }
    }

    // Render Bokeh chart if present
    if (extras.chartJson) {
      const chartDiv = document.createElement('div');
      chartDiv.className = 'chart-container';
      div.appendChild(chartDiv);
      // Defer to next tick so the element is in DOM
      setTimeout(function () {
        window.renderBokehChart(extras.chartJson, chartDiv);
      }, 0);
    }

    messagesEl.appendChild(div);
    scrollToBottom();
  }

  function buildTable(rows, columns) {
    const wrapper = document.createElement('div');
    wrapper.className = 'result-table-wrapper';

    const table = document.createElement('table');
    table.className = 'result-table';

    // Header
    if (columns && columns.length) {
      const thead = document.createElement('thead');
      const tr = document.createElement('tr');
      columns.forEach(function (col) {
        const th = document.createElement('th');
        th.textContent = col;
        tr.appendChild(th);
      });
      thead.appendChild(tr);
      table.appendChild(thead);
    }

    // Body
    const tbody = document.createElement('tbody');
    rows.forEach(function (row) {
      const tr = document.createElement('tr');
      const values = columns
        ? columns.map(function (c) { return row[c] !== undefined ? row[c] : ''; })
        : Object.values(row);

      values.forEach(function (val, idx) {
        const td = document.createElement('td');
        td.textContent = val !== null && val !== undefined ? val : '';

        // Color-code reimbursement likelihood
        const lowerVal = String(val).toLowerCase();
        if (lowerVal === 'favorable') td.classList.add('favorable');
        else if (lowerVal === 'uncertain') td.classList.add('uncertain');
        else if (lowerVal === 'challenging') td.classList.add('challenging');

        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrapper.appendChild(table);
    return wrapper;
  }

  // ---- Simple Markdown renderer ----
  function renderMarkdown(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>')
      .replace(/^(\s*[-*]\s+)(.*)$/gm, '<span class="md-bullet">• $2</span>')
      .replace(/\n/g, '<br/>');
  }

  // ---- UI helpers ----

  function showConfirmBar() { confirmBar.classList.remove('hidden'); }
  function hideConfirmBar() { confirmBar.classList.add('hidden'); }

  function setLoading(loading) {
    sendBtn.disabled = loading;
    typingEl.classList.toggle('hidden', !loading);
    if (!loading) scrollToBottom();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ---- File upload helpers ----

  function setupDropZone(zoneId, inputId, fileKey, statusId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const statusEl = document.getElementById(statusId);

    if (!zone || !input) return;

    zone.addEventListener('dragover', function (e) {
      e.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', function () { zone.classList.remove('dragover'); });
    zone.addEventListener('drop', function (e) {
      e.preventDefault();
      zone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) uploadFile(file, fileKey, statusEl);
    });
    input.addEventListener('change', function () {
      if (input.files[0]) uploadFile(input.files[0], fileKey, statusEl);
    });
  }

  function uploadFile(file, fileKey, statusEl) {
    statusEl.textContent = 'Uploading…';
    statusEl.className = 'file-status';

    const formData = new FormData();
    formData.append('session_id', sessionId || '');
    formData.append('file_key', fileKey);
    formData.append(fileKey, file);

    fetch('/upload', { method: 'POST', body: formData })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.session_id) sessionId = data.session_id;
        if (data.error) {
          statusEl.textContent = '✗ ' + data.error;
          statusEl.className = 'file-status error';
        } else {
          statusEl.textContent = '✓ ' + file.name;
          handleResponse(data);
        }
      })
      .catch(function (err) {
        statusEl.textContent = '✗ Upload failed';
        statusEl.className = 'file-status error';
        console.error(err);
      });
  }

  // ---- Export modal (called from confirm_dialog.js / exportBtn) ----
  window.openExportModal = function (resultId) {
    const modal = document.getElementById('export-modal');
    modal.classList.remove('hidden');
    modal.dataset.resultId = resultId;
    document.getElementById('export-dataset-name').value = '';
    document.getElementById('export-dataset-name').focus();
  };

})();
