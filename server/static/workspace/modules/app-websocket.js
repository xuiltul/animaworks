// ── WebSocket Event Handlers ──────────────────────
// All WS event subscriptions for the workspace dashboard.

import { getState, setState, subscribe } from "./state.js";
import { t } from "/shared/i18n.js";
import { mapAnimaStatusToAnim } from "./anima-status.js";
import { getActiveRenderer } from "./office-renderer.js";
export { mapAnimaStatusToAnim } from "./anima-status.js";
import { connect, onEvent } from "./websocket.js";
import { renderAnimaSelector } from "./anima.js";
import { updateCharacterState, removeCharacter, createCharacter } from "./character.js";
import { getDesks, registerClickTarget } from "./office3d.js";
import { cancelBehavior } from "./idle_behavior.js";
import { showMessageEffect } from "./interactions.js";
import { addTimelineEvent, localISOString } from "./timeline.js";
import { updateAnimaStatus, updateCardActivity, showMessageLine, showExternalLine, updateAvatarExpression, isReplayMode, bufferReplayEvent, VISIBLE_TOOL_NAMES } from "./org-dashboard.js";
import { playReveal } from "./reveal.js";
import { createLogger } from "../../shared/logger.js";
import { bustupCandidates, resolveCachedAvatar, invalidateAvatarCache } from "../../modules/avatar-resolver.js";

const logger = createLogger("ws-app");

function applyOrBufferReplay(handler, data) {
  if (isReplayMode()) {
    bufferReplayEvent(handler, data);
  } else {
    handler(data);
  }
}

function is3dOfficeReady() {
  return getActiveRenderer() === "3d" && getState().officeInitialized;
}

// ── WebSocket Setup ──────────────────────

const wsUnsubscribers = [];
const lastAnimaStatus = {};

/** @param {{ dom: object, updateStatusDisplay: Function, getCurrentView: () => string }} deps */
export function setupWebSocket(deps) {
  const { dom, updateStatusDisplay, getCurrentView } = deps;

  wsUnsubscribers.forEach((fn) => fn());
  wsUnsubscribers.length = 0;

  connect();

  wsUnsubscribers.push(onEvent("anima.status", (data) => {
    applyOrBufferReplay((data) => {
      const { animas, selectedAnima } = getState();
      const idx = animas.findIndex((p) => p.name === data.name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], ...data };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (is3dOfficeReady()) {
        const animState = mapAnimaStatusToAnim(data.status);
        updateCharacterState(data.name, animState);
        setState({ characterStates: { ...getState().characterStates, [data.name]: animState } });
      }
      lastAnimaStatus[data.name] = data.status;
      if (getCurrentView() === "org") {
        updateAnimaStatus(data.name, data.status || data);
        updateAvatarExpression(data.name, (typeof data.status === "object" ? (data.status.state || data.status.status || "idle") : String(data.status || "idle")).toLowerCase());
      }
    }, data);
  }));

  // ── anima.interaction — inter-anima messaging visualization ──
  wsUnsubscribers.push(onEvent("anima.interaction", (data) => {
    applyOrBufferReplay((data) => {
      if (!is3dOfficeReady()) return;
      cancelBehavior(data.from_person);
      cancelBehavior(data.to_person);
      if (data.type === "message") {
        showMessageEffect(data.from_person, data.to_person, data.summary || "");
      }
    }, data);

    applyOrBufferReplay((data) => {
      if (data.type === "message" && getCurrentView() === "org") {
        showMessageLine(data.from_person, data.to_person, data.summary || "");
      }
    }, data);

    applyOrBufferReplay((data) => addTimelineEvent({
      id: Date.now().toString(),
      type: "message",
      anima: `${data.from_person} → ${data.to_person}`,
      ts: data.ts || localISOString(),
      summary: `${data.from_person} → ${data.to_person}: ${data.summary || ""}`,
      ctx: data.ctx || "",
      meta: {
        text: data.summary || "",
        message_id: data.message_id || "",
        from_person: data.from_person,
        to_person: data.to_person,
      },
    }), data);
  }));

  wsUnsubscribers.push(onEvent("anima.heartbeat", (data) => {
    applyOrBufferReplay((d) => addTimelineEvent({
      id: Date.now().toString(),
      type: "heartbeat",
      anima: d.name,
      ts: d.ts || localISOString(),
      summary: d.summary || "heartbeat completed",
      ctx: d.ctx || "",
    }), data);
    applyOrBufferReplay((d) => {
      updateCardActivity(d.name, {
        eventType: "heartbeat",
        summary: d.summary || "heartbeat",
      });
    }, data);
  }));

  wsUnsubscribers.push(onEvent("anima.cron", (data) => {
    applyOrBufferReplay((data) => addTimelineEvent({
      id: Date.now().toString(),
      type: "cron",
      anima: data.name,
      ts: data.ts || localISOString(),
      summary: data.summary || `cron: ${data.task || ""}`,
      ctx: data.ctx || "",
    }), data);
    applyOrBufferReplay((data) => {
      updateCardActivity(data.name, {
        eventType: "cron",
        summary: data.summary || `cron: ${data.task || ""}`,
      });
    }, data);
  }));

  // ── anima.tool_activity — live tool usage ──
  wsUnsubscribers.push(onEvent("anima.tool_activity", (data) => {
    const evtType = data.event || data.type || "";
    const toolName = data.tool_name || data.tool || "tool";
    const isStreamingTool = evtType === "tool_start" || evtType === "tool_end" || evtType === "tool_detail";
    if (!isStreamingTool || VISIBLE_TOOL_NAMES.has(toolName)) {
      const _cardUpdatePayload = {
        eventType: evtType,
        toolName,
        toolId: data.tool_id,
        isError: data.is_error,
        detail: data.detail,
        summary: data.summary || "",
        content: data.content || "",
        from_person: data.from_person || "",
        to_person: data.to_person || "",
        channel: data.channel || "",
      };
      applyOrBufferReplay((d) => updateCardActivity(d.name, d.payload), { name: data.name, payload: _cardUpdatePayload });
    }
    if (evtType === "message_sent" && data.to_person) {
      applyOrBufferReplay((d) => {
        if (getCurrentView() === "org") {
          const intent = d.meta?.intent || "";
          const fromType = d.meta?.from_type || "";
          let msgLineType = "internal";
          if (intent === "delegation") msgLineType = "delegation";
          else if (fromType === "external") msgLineType = "external";
          showMessageLine(d.name, d.to_person, d.summary || "", { lineType: msgLineType });
        }
      }, data);
      applyOrBufferReplay((d) => addTimelineEvent({
        id: Date.now().toString(),
        type: "message",
        anima: `${d.name} → ${d.to_person}`,
        ts: d.ts || localISOString(),
        summary: `${d.name} → ${d.to_person}: ${d.summary || ""}`,
        ctx: d.ctx || "",
        meta: { from_person: d.name, to_person: d.to_person },
      }), data);
    }
    if (evtType === "tool_use" || evtType === "tool_end") {
      applyOrBufferReplay((d) => {
        if (getCurrentView() === "org") showExternalLine(d.name, d.toolName, "out");
      }, { ...data, toolName });
    }
    if (evtType === "message_received" && data.meta?.from_type === "external") {
      const channel = data.meta?.channel || data.channel || "";
      const extTool = channel.split(":")[0];
      if (extTool) {
        applyOrBufferReplay((d) => {
          if (getCurrentView() === "org") showExternalLine(d.name, d.extTool, "in");
        }, { ...data, extTool });
      }
    }
    applyOrBufferReplay((d) => {
      document.dispatchEvent(
        new CustomEvent("anima-tool-activity", { detail: { ...d, event: d.evtType, tool_name: d.toolName } })
      );
    }, { ...data, evtType, toolName });
  }));

  // ── board.post — shared channel message ──
  wsUnsubscribers.push(onEvent("board.post", (data) => {
    const from = data.from || "?";
    const channel = data.channel || "?";
    const text = data.text || "";

    applyOrBufferReplay((d) => {
      updateCardActivity(d.from, {
        eventType: "board_post",
        channel: d.channel,
        summary: d.text.slice(0, 60),
      });
    }, { ...data, from, channel, text });

    applyOrBufferReplay((d) => addTimelineEvent({
      id: Date.now().toString(),
      type: "board",
      anima: d.from,
      ts: d.ts || localISOString(),
      summary: `#${d.channel}: ${d.from} — ${d.text}`,
      meta: {
        channel: d.channel,
        from: d.from,
        text: d.text,
        source: d.source || "",
      },
    }), { ...data, from, channel, text });
  }));

  // ── anima.proactive_message — autonomous outbound messages ──
  wsUnsubscribers.push(onEvent("anima.proactive_message", (data) => {
    const animaName = data.name || data.anima || "";
    const summary = data.summary || data.message || "";
    if (animaName) {
      applyOrBufferReplay((d) => addTimelineEvent({
        id: Date.now().toString(),
        type: "dm_sent",
        anima: d.animaName,
        ts: d.ts || localISOString(),
        summary: d.summary.slice(0, 100),
      }), { ...data, animaName, summary });
    }
  }));

  // ── anima.notification — human notifications ──
  wsUnsubscribers.push(onEvent("anima.notification", (_data) => {
    // Timeline entry handled by anima.proactive_message to avoid duplicates
  }));

  wsUnsubscribers.push(onEvent("anima.bootstrap", (data) => {
    applyOrBufferReplay((d) => {
      const { name, status: bsStatus } = d;
      if (bsStatus === "started") {
        const { animas } = getState();
        const idx = animas.findIndex((p) => p.name === name);
        if (idx >= 0) {
          animas[idx] = { ...animas[idx], status: "bootstrapping", bootstrapping: true };
          setState({ animas: [...animas] });
          renderAnimaSelector(dom.animaSelector);
        }
        if (is3dOfficeReady()) {
          updateCharacterState(name, "thinking");
        }
        // The embedded Chat pane loads newly hired animas on this signal.
        window.dispatchEvent(new CustomEvent("aw:animas-changed"));
      } else if (bsStatus === "completed") {
        const { animas } = getState();
        const idx = animas.findIndex((p) => p.name === name);
        if (idx >= 0) {
          animas[idx] = { ...animas[idx], status: "idle", bootstrapping: false };
          setState({ animas: [...animas] });
          renderAnimaSelector(dom.animaSelector);
        }
        if (is3dOfficeReady()) {
          updateCharacterState(name, "idle");
        }
      } else if (bsStatus === "failed") {
        const { animas } = getState();
        const idx = animas.findIndex((p) => p.name === name);
        if (idx >= 0) {
          animas[idx] = { ...animas[idx], status: "error", bootstrapping: false };
          setState({ animas: [...animas] });
          renderAnimaSelector(dom.animaSelector);
        }
        if (is3dOfficeReady()) {
          updateCharacterState(name, "error");
        }
      }
    }, data);
  }));

  wsUnsubscribers.push(onEvent("anima.assets_updated", async (data) => {
    const animaName = data.name;
    invalidateAvatarCache(animaName).catch(() => {});
    applyOrBufferReplay(async (d) => {
      const animaName = d.name;
      const assets = d.assets || [];
      const hasAvatar = assets.some(
        (a) => a.startsWith("avatar_") || a === "icon.png" || a === "icon_realistic.png",
      );
      if (hasAvatar) {
        const avatarUrl = await resolveCachedAvatar(animaName, bustupCandidates(), "M");
        if (avatarUrl) await playReveal({ name: animaName, avatarUrl });
      }

      if (is3dOfficeReady()) {
        const desks = getDesks();
        const deskPos = desks[animaName];
        if (deskPos) {
          removeCharacter(animaName);
          const group = await createCharacter(
            animaName,
            { x: deskPos.x, y: deskPos.y + 0.4, z: deskPos.z - 0.3 },
          );
          if (group) {
            group.traverse((child) => {
              if (child.isMesh) registerClickTarget(animaName, child);
            });
          }
        }
      }

    }, data);
  }));

  // Track connection state for status indicator
  wsUnsubscribers.push(subscribe((state) => {
    if (state.wsConnected) {
      updateStatusDisplay(true, t("status.connected", { count: state.animas.length }));
    } else {
      updateStatusDisplay(false, t("ws.reconnect"));
    }
  }));
}
