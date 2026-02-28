// ── WebSocket Event Handlers ──────────────────────
// All WS event subscriptions for the workspace dashboard.

import { getState, setState, subscribe } from "./state.js";
import { connect, onEvent } from "./websocket.js";
import { renderAnimaSelector, renderStatus } from "./anima.js";
import { updateCharacterState, removeCharacter, createCharacter } from "./character.js";
import { setCharacter } from "./live2d.js";
import { getDesks, registerClickTarget } from "./office3d.js";
import { cancelBehavior } from "./idle_behavior.js";
import { showMessageEffect } from "./interactions.js";
import { addTimelineEvent, localISOString } from "./timeline.js";
import { addActivity } from "./activity.js";
import { getSelectedBoard, appendBoardMessage } from "./board.js";
import { updateAnimaStatus, addActivityItem } from "./org-dashboard.js";
import { playReveal } from "./reveal.js";
import { createLogger } from "../../shared/logger.js";

const logger = createLogger("ws-app");

// ── Status Mapping ──────────────────────

export function mapAnimaStatusToAnim(status) {
  if (!status) return "idle";
  const s = typeof status === "object" ? status.state || status.status || "idle" : String(status);
  const lower = s.toLowerCase();
  if (lower === "not_found" || lower === "stopped") return "sleeping";
  if (lower.includes("bootstrap")) return "thinking";
  if (lower.includes("think") || lower.includes("process")) return "thinking";
  if (lower.includes("work") || lower.includes("busy") || lower.includes("running")) return "working";
  if (lower.includes("error") || lower.includes("fail")) return "error";
  if (lower.includes("sleep") || lower.includes("stop") || lower.includes("inactive")) return "sleeping";
  if (lower.includes("talk") || lower.includes("chat")) return "talking";
  if (lower.includes("report")) return "reporting";
  return "idle";
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
    const { animas, selectedAnima } = getState();
    const idx = animas.findIndex((p) => p.name === data.name);
    if (idx >= 0) {
      animas[idx] = { ...animas[idx], ...data };
      setState({ animas: [...animas] });
      renderAnimaSelector(dom.animaSelector);
    }
    if (data.name === selectedAnima) {
      renderStatus(dom.paneState);
    }
    if (getState().officeInitialized) {
      const animState = mapAnimaStatusToAnim(data.status);
      updateCharacterState(data.name, animState);
      setState({ characterStates: { ...getState().characterStates, [data.name]: animState } });
    }
    if (lastAnimaStatus[data.name] !== data.status) {
      lastAnimaStatus[data.name] = data.status;
      addActivity("system", data.name, `Status: ${data.status}`);
    }
    if (getCurrentView() === "org") {
      updateAnimaStatus(data.name, data.status || data);
    }
  }));

  // ── anima.interaction — inter-anima messaging visualization ──
  wsUnsubscribers.push(onEvent("anima.interaction", (data) => {
    cancelBehavior(data.from_person);
    cancelBehavior(data.to_person);

    if (data.type === "message") {
      showMessageEffect(data.from_person, data.to_person, data.summary || "");
    }

    addTimelineEvent({
      id: Date.now().toString(),
      type: "message",
      anima: `${data.from_person} → ${data.to_person}`,
      ts: data.ts || localISOString(),
      summary: `${data.from_person} → ${data.to_person}: ${data.summary || ""}`,
      meta: {
        text: data.summary || "",
        message_id: data.message_id || "",
        from_person: data.from_person,
        to_person: data.to_person,
      },
    });
    if (getCurrentView() === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "anima.interaction",
        from: data.from_person || data.from || "",
        summary: data.summary || "",
      });
    }
  }));

  wsUnsubscribers.push(onEvent("anima.heartbeat", (data) => {
    addActivity("heartbeat", data.name, data.summary || "heartbeat completed");
    const { selectedAnima } = getState();
    if (data.name === selectedAnima) {
      renderStatus(dom.paneState);
    }
    addTimelineEvent({
      id: Date.now().toString(),
      type: "heartbeat",
      anima: data.name,
      ts: data.ts || localISOString(),
      summary: data.summary || "heartbeat completed",
    });
    if (getCurrentView() === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "anima.heartbeat",
        from: data.name || "",
        summary: data.summary || "heartbeat completed",
      });
    }
  }));

  wsUnsubscribers.push(onEvent("anima.cron", (data) => {
    addActivity("cron", data.name, data.summary || `cron: ${data.task || ""}`);
    addTimelineEvent({
      id: Date.now().toString(),
      type: "cron",
      anima: data.name,
      ts: data.ts || localISOString(),
      summary: data.summary || `cron: ${data.task || ""}`,
    });
    if (getCurrentView() === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "anima.cron",
        from: data.name || "",
        summary: data.summary || `cron: ${data.task || ""}`,
      });
    }
  }));

  // ── board.post — shared channel message ──
  wsUnsubscribers.push(onEvent("board.post", (data) => {
    const from = data.from || "?";
    const channel = data.channel || "?";
    const text = data.text || "";

    const boardSel = getSelectedBoard();
    if (boardSel.type === "channel" && boardSel.channel === channel) {
      appendBoardMessage({
        ts: data.ts || new Date().toISOString(),
        from,
        text,
        source: data.source || "",
      });
    }

    addActivity("chat", from, `[#${channel}] ${text}`);

    addTimelineEvent({
      id: Date.now().toString(),
      type: "board",
      anima: from,
      ts: data.ts || localISOString(),
      summary: `#${channel}: ${from} — ${text}`,
      meta: {
        channel,
        from,
        text,
        source: data.source || "",
      },
    });
    if (getCurrentView() === "org") {
      addActivityItem({
        ts: data.ts || new Date().toISOString(),
        type: "board.post",
        from,
        summary: `[#${channel}] ${text}`,
      });
    }
  }));

  // ── anima.proactive_message — autonomous outbound messages ──
  wsUnsubscribers.push(onEvent("anima.proactive_message", (data) => {
    const animaName = data.name || data.anima || "";
    const summary = data.summary || data.message || "";
    if (animaName) {
      addTimelineEvent({
        id: Date.now().toString(),
        type: "dm_sent",
        anima: animaName,
        ts: data.ts || localISOString(),
        summary: summary.slice(0, 100),
      });
    }
  }));

  // ── anima.notification — human notifications ──
  wsUnsubscribers.push(onEvent("anima.notification", (_data) => {
    // Timeline entry handled by anima.proactive_message to avoid duplicates
  }));

  wsUnsubscribers.push(onEvent("anima.bootstrap", (data) => {
    const { name, status: bsStatus } = data;
    if (bsStatus === "started") {
      const { animas } = getState();
      const idx = animas.findIndex((p) => p.name === name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], status: "bootstrapping", bootstrapping: true };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "thinking");
      }
      addActivity("system", name, "ブートストラップ開始");
    } else if (bsStatus === "completed") {
      const { animas } = getState();
      const idx = animas.findIndex((p) => p.name === name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], status: "idle", bootstrapping: false };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "idle");
      }
      addActivity("system", name, "ブートストラップ完了");
    } else if (bsStatus === "failed") {
      const { animas } = getState();
      const idx = animas.findIndex((p) => p.name === name);
      if (idx >= 0) {
        animas[idx] = { ...animas[idx], status: "error", bootstrapping: false };
        setState({ animas: [...animas] });
        renderAnimaSelector(dom.animaSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "error");
      }
      addActivity("system", name, "ブートストラップ失敗");
    }
  }));

  wsUnsubscribers.push(onEvent("anima.assets_updated", async (data) => {
    const animaName = data.name;
    addActivity("system", animaName, `アセット更新: ${(data.assets || []).join(", ")}`);

    // ── Reveal animation (Anima birth) ──
    const assets = data.assets || [];
    const hasAvatar = assets.some((a) => a.startsWith("avatar_"));
    if (hasAvatar) {
      const avatarUrl = `/api/animas/${encodeURIComponent(animaName)}/assets/avatar_bustup.png`;
      await playReveal({ name: animaName, avatarUrl });
    }

    // Refresh 3D character if office is initialised
    if (getState().officeInitialized) {
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

    // Refresh bust-up if conversation is open for this anima
    if (getState().conversationAnima === animaName) {
      await setCharacter(animaName);
    }
  }));

  // Track connection state for status indicator
  wsUnsubscribers.push(subscribe((state) => {
    if (state.wsConnected) {
      updateStatusDisplay(true, `接続済 (${state.animas.length}名)`);
    } else {
      updateStatusDisplay(false, "再接続中...");
    }
  }));
}
