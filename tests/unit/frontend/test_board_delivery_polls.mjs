import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const deliveryPath = resolve(__dirname, "../../../server/static/pages/board/delivery.js");
const translations = {
  "board.delivery_notified": "届いた相手",
  "board.delivery_not_notified": "届かなかった相手",
  "board.delivery_reason_polls": "読みに来たら届く",
  "board.delivery_delivered": "届けた",
  "board.delivery_unread": "未読",
  "board.delivery_reply_waiting": "返事待ち",
};

const source = readFileSync(deliveryPath, "utf8").replace(/^import .*;\r?\n/gm, "");
const stubs = `
  const translations = ${JSON.stringify(translations)};
  const t = (key, params = {}) => {
    let value = translations[key] || key;
    for (const [name, replacement] of Object.entries(params)) {
      value = value.replaceAll("{" + name + "}", replacement);
    }
    return value;
  };
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
  const avatarMarkup = () => '<span class="avatar"></span>';
`;
const modUrl = "data:text/javascript;base64," + Buffer.from(stubs + source).toString("base64");
const { deliveryMarkup } = await import(modUrl);

describe("board delivery footer", () => {
  it("renders poll members with the translated polling hint", () => {
    const html = deliveryMarkup({
      id: "p1",
      delivery: { notified: [], not_notified: { muse: "polls" } },
    });

    assert.match(html, /muse · 読みに来たら届く/);
    assert.doesNotMatch(html, /muse · polls/);
  });

  it("keeps raw failure reasons that have no translation", () => {
    const html = deliveryMarkup({
      id: "p2",
      delivery: { notified: [], not_notified: { muse: "http_404" } },
    });

    assert.match(html, /muse · http_404/);
  });
});
