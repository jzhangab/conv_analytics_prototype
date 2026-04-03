/**
 * confirm_dialog.js — Export modal and confirmation dialog helpers.
 */

(function () {
  'use strict';

  const modal = document.getElementById('export-modal');
  const confirmBtn = document.getElementById('export-confirm-btn');
  const cancelBtn = document.getElementById('export-cancel-btn');
  const datasetInput = document.getElementById('export-dataset-name');

  cancelBtn.addEventListener('click', function () {
    modal.classList.add('hidden');
  });

  confirmBtn.addEventListener('click', function () {
    const datasetName = datasetInput.value.trim();
    if (!datasetName) {
      datasetInput.style.borderColor = '#c62828';
      return;
    }
    datasetInput.style.borderColor = '';
    const resultId = modal.dataset.resultId;
    modal.classList.add('hidden');

    const sessionId = window._getSessionId ? window._getSessionId() : null;

    fetch('/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        result_id: resultId,
        dataset_name: datasetName,
      }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        // Append export confirmation as assistant message
        const messagesEl = document.getElementById('messages');
        const div = document.createElement('div');
        div.className = 'message assistant';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = data.message || 'Export complete.';
        div.appendChild(bubble);
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      })
      .catch(function (err) {
        console.error('Export error:', err);
      });
  });

  // Close on backdrop click
  modal.addEventListener('click', function (e) {
    if (e.target === modal) modal.classList.add('hidden');
  });

  // Close on Escape
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      modal.classList.add('hidden');
    }
  });

})();
