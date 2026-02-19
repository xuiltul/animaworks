// ── Shared Utilities ──────────────────────
// Common helpers used across workspace modules.

/**
 * Escape HTML special characters to prevent XSS.
 */
export function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/**
 * Strip LLM emotion tags from text.
 * Emotion tags (<!-- emotion: {"emotion": "smile"} -->) are metadata
 * for expression switching and should never be shown to the user.
 */
export function stripEmotionTag(text) {
  return text.replace(/<!--\s*emotion:\s*\{.*?\}\s*-->/gs, "").trimEnd();
}

/**
 * Lightweight Markdown → HTML renderer.
 * Escapes HTML first, then applies safe transforms.
 */
export function renderSimpleMarkdown(text) {
  if (!text) return "";

  let html = escapeHtml(stripEmotionTag(text));

  // Fenced code blocks: ```lang\n...\n```
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) => {
    return `<pre class="md-code-block"><code>${code}</code></pre>`;
  });

  // Tables: consecutive lines starting with |
  html = html.replace(/((?:^\|.+\|[ \t]*\n?)+)/gm, (block) => {
    const rows = block.trim().split("\n").filter((r) => r.trim());
    if (rows.length < 2) return block;

    // Check if row 2 is a separator (|---|---|)
    const isSep = /^\|[\s:]*-+[\s:]*(\|[\s:]*-+[\s:]*)*\|?$/.test(rows[1].trim());
    const headerRow = isSep ? rows[0] : null;
    const dataRows = isSep ? rows.slice(2) : rows;

    const parseCells = (row) =>
      row.split("|").slice(1, -1).map((c) => c.trim());

    let tableHtml = '<table class="md-table">';
    if (headerRow) {
      const cells = parseCells(headerRow);
      tableHtml += "<thead><tr>" + cells.map((c) => `<th>${c}</th>`).join("") + "</tr></thead>";
    }
    tableHtml += "<tbody>";
    for (const row of dataRows) {
      if (!row.trim()) continue;
      const cells = parseCells(row);
      tableHtml += "<tr>" + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>";
    }
    tableHtml += "</tbody></table>";
    return tableHtml;
  });

  // Inline code: `...`
  html = html.replace(/`([^`]+)`/g, '<code class="md-code-inline">$1</code>');

  // Headings: # ... #### (must be at line start)
  html = html.replace(/^#### (.+)$/gm, '<h4 class="md-heading">$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3 class="md-heading">$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2 class="md-heading">$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1 class="md-heading">$1</h1>');

  // Bold: **...**
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic: *...*
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

  // Links: [text](url) — only allow http/https to prevent javascript: XSS
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener">$1</a>'
  );

  // Unordered list items
  html = html.replace(/^(?:[-*]) (.+)$/gm, "<li>$1</li>");
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

  // Horizontal rule: --- or ***
  html = html.replace(/^(?:---|\*\*\*)$/gm, '<hr class="md-hr">');

  // Line breaks (outside of pre, heading, and table blocks)
  html = html.replace(
    /(<pre[\s\S]*?<\/pre>)|(<h[1-4][^>]*>.*?<\/h[1-4]>)|(<table[\s\S]*?<\/table>)|(\n)/g,
    (_m, pre, heading, table, nl) => (pre || heading || table ? (pre || heading || table) : nl ? "<br>" : "")
  );

  return html;
}

/**
 * Format an ISO timestamp to HH:MM (ja-JP).
 */
export function timeStr(isoOrTs) {
  if (!isoOrTs) return "--:--";
  const d = new Date(isoOrTs);
  if (isNaN(d.getTime())) return "--:--";
  return d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
}

/**
 * Format a timestamp with smart granularity:
 *  - Same day:      "HH:MM"
 *  - Same year:     "MM/DD HH:MM"
 *  - Previous year: "YYYY/MM/DD HH:MM"
 */
export function smartTimestamp(isoOrTs) {
  if (!isoOrTs) return "";
  const d = new Date(isoOrTs);
  if (isNaN(d.getTime())) return "";
  const now = new Date();
  const time = d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  if (sameDay) return time;
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  if (d.getFullYear() === now.getFullYear()) return `${mm}/${dd} ${time}`;
  return `${d.getFullYear()}/${mm}/${dd} ${time}`;
}

/**
 * Strip .md extension from filename for display.
 */
export function stripMdExtension(filename) {
  return filename.replace(/\.md$/, "");
}
