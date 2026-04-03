/**
 * chart_viewer.js — Embeds Bokeh charts into the chat panel.
 * Depends on Bokeh JS loaded via CDN in index.html.
 */

window.renderBokehChart = function (chartJson, containerEl) {
  if (!chartJson || !containerEl) return;

  try {
    // Bokeh.embed.embed_item places the chart into the given DOM element
    Bokeh.embed.embed_item(chartJson, containerEl.id || _ensureId(containerEl));
  } catch (e) {
    console.error('Bokeh render error:', e);
    containerEl.innerHTML =
      '<p style="color:#c62828;padding:10px;">Chart could not be rendered. ' +
      'Ensure Bokeh JS is loaded and the chart JSON is valid.</p>';
  }
};

// Ensure the container has an ID (Bokeh needs one)
function _ensureId(el) {
  if (!el.id) {
    el.id = 'bokeh-chart-' + Math.random().toString(36).slice(2, 9);
  }
  return el.id;
}
