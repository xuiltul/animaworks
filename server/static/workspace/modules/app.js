// ── App Entry Point ──────────────────────
// Initialization, screen switching, and event delegation.

import { getState, setState, subscribe } from "./state.js";
import { fetchSystemStatus, fetchConversationFull, greetPerson } from "./api.js";
import { connect, onEvent } from "./websocket.js";
import { initLogin, getCurrentUser, logout } from "./login.js";
import { initPerson, loadPersons, selectPerson, renderPersonSelector, renderStatus } from "./person.js";
import { initMemory, loadMemoryTab } from "./memory.js";
import { initSession, loadSessions } from "./session.js";
import { escapeHtml, renderSimpleMarkdown } from "./utils.js";
import { initOffice, getDesks, highlightDesk, setCharacterClickHandler, getScene, registerClickTarget, setCharacterUpdateHook, getObstacles, getFloorDimensions } from "./office3d.js";
import { initCharacters, createCharacter, removeCharacter, updateCharacterState, updateAllCharacters, getCharacterGroup, getCharacterHome, setAppearance } from "./character.js";
import { initBustup, setCharacter, setExpression, setTalking, onClick as onBustupClick, setLive2dAppearance } from "./live2d.js";
import { createNavGrid } from "./navigation.js";
import { initMovement, registerCharacter, updateMovements, moveTo, moveToHome, stopMovement, isMoving } from "./movement.js";
import { computePOIs, initIdleBehaviors, updateIdleBehaviors, cancelBehavior } from "./idle_behavior.js";
import { initInteractions, showMessageEffect, showConversation, updateInteractions } from "./interactions.js";
import { initTimeline, addTimelineEvent, loadHistory } from "./timeline.js";
import { playReveal } from "./reveal.js";
import { parseConvSSE, getErrorMessage } from "../../shared/sse-parser.js";

// ── DOM References ──────────────────────

const dom = {};

function cacheDom() {
  dom.loginContainer = document.getElementById("wsLogin");
  dom.dashboard = document.getElementById("wsDashboard");
  dom.personSelector = document.getElementById("wsPersonSelector");
  dom.systemStatus = document.getElementById("wsSystemStatus");
  dom.userInfo = document.getElementById("wsUserInfo");
  dom.rightTabs = document.getElementById("wsRightTabs");
  dom.tabState = document.getElementById("wsTabState");
  dom.tabActivity = document.getElementById("wsTabActivity");
  dom.tabHistory = document.getElementById("wsTabHistory");
  dom.paneState = document.getElementById("wsPaneState");
  dom.paneActivity = document.getElementById("wsPaneActivity");
  dom.paneHistory = document.getElementById("wsPaneHistory");
  dom.memoryPanel = document.getElementById("wsMemoryPanel");
  dom.logoutBtn = document.getElementById("wsLogoutBtn");

  // 3D Office
  dom.officePanel = document.getElementById("wsOfficePanel");
  dom.officeCanvas = document.getElementById("wsOfficeCanvas");

  // Conversation overlay (3-column)
  dom.convOverlay = document.getElementById("wsConvOverlay");
  dom.convLayout = document.getElementById("wsConvLayout");
  dom.convBack = document.getElementById("wsConvBack");
  dom.convPersonName = document.getElementById("wsConvPersonName");
  dom.convCanvas = document.getElementById("wsConvCanvas");
  dom.convMessages = document.getElementById("wsConvMessages");
  dom.convInput = document.getElementById("wsConvInput");
  dom.convSend = document.getElementById("wsConvSend");
}

// ── Activity Feed ──────────────────────

const TYPE_ICONS = {
  heartbeat: "\uD83D\uDC93",
  cron: "\u23F0",
  chat: "\uD83D\uDCAC",
  system: "\u2699\uFE0F",
};

function addActivity(type, personName, summary) {
  if (!dom.paneActivity) return;

  const now = new Date();
  const ts = now.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  const icon = TYPE_ICONS[type] || "\u2022";

  const entry = document.createElement("div");
  entry.className = "activity-entry";
  entry.innerHTML = `
    <span class="activity-time">${ts}</span>
    <span class="activity-icon">${icon}</span>
    <span class="activity-person">${escapeHtml(personName)}</span>
    <span class="activity-summary">${escapeHtml(summary)}</span>`;

  dom.paneActivity.prepend(entry);

  // Cap at 200 entries
  while (dom.paneActivity.children.length > 200) {
    dom.paneActivity.removeChild(dom.paneActivity.lastChild);
  }
}

// ── Right Panel Tabs ──────────────────────

function activateRightTab(tab) {
  setState({ activeRightTab: tab });

  [dom.tabState, dom.tabActivity, dom.tabHistory].forEach((btn) => {
    btn?.classList.toggle("active", btn.dataset.tab === tab);
  });

  [dom.paneState, dom.paneActivity, dom.paneHistory].forEach((pane) => {
    if (pane) pane.style.display = pane.dataset.pane === tab ? "" : "none";
  });

  if (tab === "history") {
    loadSessions();
  }
}

// ── 3D Office Initialization ──────────────────────

async function initOfficeIfNeeded() {
  if (getState().officeInitialized) return;
  setState({ officeInitialized: true });

  try {
    // Initialize 3D scene with person data for dynamic desk layout
    const { persons } = getState();
    initOffice(dom.officeCanvas, persons);

    // Initialize characters in the scene
    const scene = getScene();
    initCharacters(scene);

    // Register character animation update in the render loop
    setCharacterUpdateHook((dt, elapsed) => {
      updateAllCharacters(dt, elapsed);
      updateMovements(dt);
      updateIdleBehaviors(dt);
      updateInteractions(dt);
    });

    // Create characters at their dynamically assigned desk positions
    const desks = getDesks();
    for (const p of persons) {
      // Register appearance from API before creating character
      if (p.appearance) {
        setAppearance(p.name, p.appearance);
        setLive2dAppearance(p.name, p.appearance);
      }
      const deskPos = desks[p.name];
      if (deskPos) {
        const group = await createCharacter(p.name, { x: deskPos.x, y: 0, z: deskPos.z - 0.55 });
        if (group) {
          group.traverse((child) => {
            if (child.isMesh) {
              registerClickTarget(p.name, child);
            }
          });
        }
        const animState = mapPersonStatusToAnim(p.status);
        updateCharacterState(p.name, animState);
      }
    }

    // ── Navigation + Movement system init ──────────
    const { width: floorW, depth: floorD } = getFloorDimensions();
    const obstacles = getObstacles();
    const navGrid = createNavGrid(floorW, floorD, obstacles);
    initMovement(navGrid, floorW, floorD);

    // Register all characters with the movement system
    for (const p of persons) {
      const group = getCharacterGroup(p.name);
      const home = getCharacterHome(p.name);
      if (group && home) {
        registerCharacter(p.name, group, home);
      }
    }

    // ── Interactions + Idle Behaviors + Timeline init ──────────
    const movementSystem = { moveTo, moveToHome, stopMovement, isMoving };
    const characterMap = Object.fromEntries(persons.map((p) => [p.name, true]));
    initInteractions(scene, characterMap, movementSystem);

    const pois = computePOIs(floorW, floorD);
    initIdleBehaviors(characterMap, movementSystem, pois);

    initTimeline(dom.officePanel, { showMessageEffect, showConversation });
    loadHistory(24);

    // Handle character clicks → open conversation in right panel
    setCharacterClickHandler((personName) => {
      selectPerson(personName);
      openConversation(personName);
    });

    // Highlight selected person's desk
    const { selectedPerson } = getState();
    if (selectedPerson) {
      highlightDesk(selectedPerson);
    }
  } catch (err) {
    console.error("[office] Failed to initialize 3D office:", err);
    setState({ officeInitialized: false });
  }
}

function mapPersonStatusToAnim(status) {
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

// ── Conversation Panel (Right Sidebar) ──────────────────────

let bustupInitialized = false;
let convStreamController = null;

async function openConversation(personName) {
  if (!dom.convOverlay) return;

  setState({ conversationOpen: true, conversationPerson: personName });

  // Show conversation overlay on top of office
  dom.convOverlay.classList.remove("hidden");

  // Update person name
  if (dom.convPersonName) dom.convPersonName.textContent = personName;

  // Initialize bust-up canvas (once)
  if (!bustupInitialized && dom.convCanvas) {
    initBustup(dom.convCanvas);
    bustupInitialized = true;

    onBustupClick(() => {
      setExpression("surprised");
      setTimeout(() => setExpression("smile"), 1200);
      setTimeout(() => setExpression("neutral"), 2500);
    });
  }

  // Set character (may load AI-generated bust-up image) and expression
  await setCharacter(personName);
  setExpression("neutral");

  // Load and render chat history, then trigger greeting
  await loadAndRenderConvMessages(personName);
  triggerGreeting(personName);

  // Focus input
  dom.convInput?.focus();
}

function closeConversation() {
  if (!dom.convOverlay) return;

  // Hide conversation overlay
  dom.convOverlay.classList.add("hidden");

  setState({ conversationOpen: false, conversationPerson: null });
  setTalking(false);

  // Abort any active stream
  if (convStreamController) {
    convStreamController.abort();
    convStreamController = null;
  }
}

// ── Greeting on Character Click ──────────────────────

async function triggerGreeting(personName) {
  try {
    const data = await greetPerson(personName);
    if (!data.response) return;

    // Add assistant-only greeting message to chat
    const { chatMessages } = getState();
    setState({
      chatMessages: [...chatMessages, { role: "assistant", text: data.response }],
    });
    renderConvMessages();

    // Update bust-up expression
    if (data.emotion) {
      setExpression(data.emotion);
      setTimeout(() => setExpression("neutral"), 3000);
    }
  } catch (err) {
    console.error("[greeting] Failed to greet:", err);
  }
}

// ── Chat Rendering in Conversation Panel ──────────────────────

function renderConvBubble(msg) {
  if (msg.role === "user") {
    return `<div class="chat-bubble user">${escapeHtml(msg.text)}</div>`;
  }
  const streamClass = msg.streaming ? " streaming" : "";
  let content = "";
  if (msg.text) {
    content = renderSimpleMarkdown(msg.text);
  } else if (msg.streaming) {
    content = '<span class="cursor-blink"></span>';
  }
  const toolHtml = msg.activeTool
    ? `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`
    : "";
  return `<div class="chat-bubble assistant${streamClass}">${content}${toolHtml}</div>`;
}

function renderConvMessages() {
  if (!dom.convMessages) return;
  const { chatMessages } = getState();
  if (chatMessages.length === 0) {
    dom.convMessages.innerHTML = '<div class="chat-empty">メッセージはまだありません</div>';
    return;
  }
  dom.convMessages.innerHTML = chatMessages.map(renderConvBubble).join("");
  requestAnimationFrame(() => {
    const last = dom.convMessages.lastElementChild;
    if (last) last.scrollIntoView({ block: "end", behavior: "instant" });
  });
}

async function loadAndRenderConvMessages(personName) {
  if (!personName) return;
  try {
    const data = await fetchConversationFull(personName);
    if (data.turns && data.turns.length > 0) {
      const messages = data.turns.map((t) => ({
        role: t.role === "human" ? "user" : "assistant",
        text: t.content || "",
      }));
      setState({ chatMessages: messages });
    } else {
      setState({ chatMessages: [] });
    }
  } catch (err) {
    console.error("Failed to load conversation:", err);
    setState({ chatMessages: [] });
  }
  renderConvMessages();
}

// ── SSE Streaming for Conversation ──────────────────────

async function sendConversationMessage() {
  const text = dom.convInput?.value?.trim();
  if (!text) return;

  const personName = getState().conversationPerson;
  if (!personName) return;

  // Clear input
  dom.convInput.value = "";
  dom.convInput.disabled = true;
  dom.convSend.disabled = true;

  // Add user message + streaming assistant placeholder
  const { chatMessages } = getState();
  const userMsg = { role: "user", text };
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null };
  setState({ chatMessages: [...chatMessages, userMsg, streamingMsg] });
  renderConvMessages();

  // Create AbortController for cancellable streaming
  convStreamController = new AbortController();

  try {
    const userName = getCurrentUser() || "guest";
    const resp = await fetch(`/api/persons/${encodeURIComponent(personName)}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, from_person: userName }),
      signal: convStreamController.signal,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    setTalking(true);
    setExpression("neutral");

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const { parsed, remaining } = parseConvSSE(buffer);
      buffer = remaining;

      for (const { event: evt, data } of parsed) {
        if (evt === "text_delta" && data.text) {
          streamingMsg.text += data.text;
          scheduleStreamingUpdate(streamingMsg);
        } else if (evt === "tool_start") {
          streamingMsg.activeTool = data.tool_name;
          setExpression("thinking");
          updateStreamingBubble(streamingMsg);
        } else if (evt === "tool_end") {
          streamingMsg.activeTool = null;
          setExpression("neutral");
          updateStreamingBubble(streamingMsg);
        } else if (evt === "done") {
          // Use clean summary (emotion tag already stripped server-side)
          if (data.summary) {
            streamingMsg.text = data.summary;
            updateStreamingBubble(streamingMsg);
          }
          const emotion = data.emotion || "neutral";
          setExpression(emotion);
          setTimeout(() => setExpression("neutral"), 3000);
        } else if (evt === "error") {
          setExpression("troubled");
          const errorMsg = getErrorMessage(data);
          streamingMsg.text += `\n[エラー: ${errorMsg}]`;
          updateStreamingBubble(streamingMsg);
        }
      }
    }

    setTalking(false);

    // Finalize streaming message
    streamingMsg.streaming = false;
    if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
    setState({ chatMessages: [...getState().chatMessages] });
    renderConvMessages();
  } catch (err) {
    if (err.name === "AbortError") return;
    console.error("[conversation] Stream error:", err);
    streamingMsg.text = `[エラー] ${err.message}`;
    streamingMsg.streaming = false;
    streamingMsg.activeTool = null;
    setState({ chatMessages: [...getState().chatMessages] });
    renderConvMessages();
    setExpression("troubled");
    setTalking(false);
  } finally {
    convStreamController = null;
    if (dom.convInput) dom.convInput.disabled = false;
    if (dom.convSend) dom.convSend.disabled = false;
    dom.convInput?.focus();
  }
}

// ── Streaming update with rAF throttle ──────────────────────

let _convRafPending = false;
let _convLatestStreamingMsg = null;

function scheduleStreamingUpdate(msg) {
  _convLatestStreamingMsg = msg;
  if (_convRafPending) return;
  _convRafPending = true;
  requestAnimationFrame(() => {
    _convRafPending = false;
    if (_convLatestStreamingMsg) {
      updateStreamingBubble(_convLatestStreamingMsg);
    }
  });
}

function updateStreamingBubble(msg) {
  if (!dom.convMessages) return;
  const bubble = dom.convMessages.querySelector(".chat-bubble.assistant.streaming");
  if (!bubble) return;

  let html = "";
  if (msg.text) {
    html = renderSimpleMarkdown(msg.text);
  } else {
    html = '<span class="cursor-blink"></span>';
  }
  if (msg.activeTool) {
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`;
  }
  bubble.innerHTML = html;
  requestAnimationFrame(() => {
    bubble.scrollIntoView({ block: "end", behavior: "instant" });
  });
}

// ── System Status ──────────────────────

async function loadSystemStatus() {
  if (!dom.systemStatus) return;
  try {
    const data = await fetchSystemStatus();
    updateStatusDisplay(
      data.scheduler_running,
      `${data.scheduler_running ? "稼働中" : "停止"} (${data.persons}名)`
    );
  } catch {
    updateStatusDisplay(false, "接続失敗");
  }
}

function updateStatusDisplay(ok, text) {
  if (!dom.systemStatus) return;
  const dot = dom.systemStatus.querySelector(".status-dot");
  const label = dom.systemStatus.querySelector(".status-text");
  if (dot) dot.className = `status-dot ${ok ? "status-idle" : "status-error"}`;
  if (label) label.textContent = text;
}

// ── WebSocket Handlers ──────────────────────

const wsUnsubscribers = [];
const lastPersonStatus = {};

function setupWebSocket() {
  wsUnsubscribers.forEach((fn) => fn());
  wsUnsubscribers.length = 0;

  connect();

  wsUnsubscribers.push(onEvent("person.status", (data) => {
    const { persons, selectedPerson } = getState();
    const idx = persons.findIndex((p) => p.name === data.name);
    if (idx >= 0) {
      persons[idx] = { ...persons[idx], ...data };
      setState({ persons: [...persons] });
      renderPersonSelector(dom.personSelector);
    }
    if (data.name === selectedPerson) {
      renderStatus(dom.paneState);
    }
    // Update 3D character animation
    if (getState().officeInitialized) {
      const animState = mapPersonStatusToAnim(data.status);
      updateCharacterState(data.name, animState);
      setState({ characterStates: { ...getState().characterStates, [data.name]: animState } });
    }
    // Only log to activity feed when status actually changes
    if (lastPersonStatus[data.name] !== data.status) {
      lastPersonStatus[data.name] = data.status;
      addActivity("system", data.name, `Status: ${data.status}`);
    }
  }));

  // ── person.interaction — inter-person messaging visualization ──
  wsUnsubscribers.push(onEvent("person.interaction", (data) => {
    cancelBehavior(data.from_person);
    cancelBehavior(data.to_person);

    if (data.type === "message") {
      showMessageEffect(data.from_person, data.to_person, data.summary || "");
    }

    addTimelineEvent({
      id: Date.now().toString(),
      type: "message",
      persons: [data.from_person, data.to_person],
      timestamp: new Date().toISOString(),
      summary: `${data.from_person} → ${data.to_person}: ${data.summary || ""}`,
      metadata: { text: data.summary || "" },
    });
  }));

  wsUnsubscribers.push(onEvent("person.heartbeat", (data) => {
    addActivity("heartbeat", data.name, data.summary || "heartbeat completed");
    const { selectedPerson } = getState();
    if (data.name === selectedPerson) {
      renderStatus(dom.paneState);
    }
    addTimelineEvent({
      id: Date.now().toString(),
      type: "heartbeat",
      persons: [data.name],
      timestamp: new Date().toISOString(),
      summary: data.summary || "heartbeat completed",
    });
  }));

  wsUnsubscribers.push(onEvent("person.cron", (data) => {
    addActivity("cron", data.name, data.summary || `cron: ${data.job || ""}`);
    addTimelineEvent({
      id: Date.now().toString(),
      type: "cron",
      persons: [data.name],
      timestamp: new Date().toISOString(),
      summary: data.summary || `cron: ${data.job || ""}`,
    });
  }));

  wsUnsubscribers.push(onEvent("person.bootstrap", (data) => {
    const { name, status: bsStatus } = data;
    if (bsStatus === "started") {
      const { persons } = getState();
      const idx = persons.findIndex((p) => p.name === name);
      if (idx >= 0) {
        persons[idx] = { ...persons[idx], status: "bootstrapping", bootstrapping: true };
        setState({ persons: [...persons] });
        renderPersonSelector(dom.personSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "thinking");
      }
      addActivity("system", name, "ブートストラップ開始");
    } else if (bsStatus === "completed") {
      const { persons } = getState();
      const idx = persons.findIndex((p) => p.name === name);
      if (idx >= 0) {
        persons[idx] = { ...persons[idx], status: "idle", bootstrapping: false };
        setState({ persons: [...persons] });
        renderPersonSelector(dom.personSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "idle");
      }
      addActivity("system", name, "ブートストラップ完了");
    } else if (bsStatus === "failed") {
      const { persons } = getState();
      const idx = persons.findIndex((p) => p.name === name);
      if (idx >= 0) {
        persons[idx] = { ...persons[idx], status: "error", bootstrapping: false };
        setState({ persons: [...persons] });
        renderPersonSelector(dom.personSelector);
      }
      if (getState().officeInitialized) {
        updateCharacterState(name, "error");
      }
      addActivity("system", name, "ブートストラップ失敗");
    }
  }));

  wsUnsubscribers.push(onEvent("person.assets_updated", async (data) => {
    const personName = data.name;
    addActivity("system", personName, `アセット更新: ${(data.assets || []).join(", ")}`);

    // ── Reveal animation (Person birth) ──
    const assets = data.assets || [];
    const hasAvatar = assets.some((a) => a.startsWith("avatar_"));
    if (hasAvatar) {
      const avatarUrl = `/api/persons/${encodeURIComponent(personName)}/assets/avatar_bustup.png`;
      await playReveal({ name: personName, avatarUrl });
    }

    // Refresh 3D character if office is initialised
    if (getState().officeInitialized) {
      const desks = getDesks();
      const deskPos = desks[personName];
      if (deskPos) {
        removeCharacter(personName);
        const group = await createCharacter(
          personName,
          { x: deskPos.x, y: deskPos.y + 0.4, z: deskPos.z - 0.3 },
        );
        if (group) {
          group.traverse((child) => {
            if (child.isMesh) registerClickTarget(personName, child);
          });
        }
      }
    }

    // Refresh bust-up if conversation is open for this person
    if (getState().conversationPerson === personName) {
      await setCharacter(personName);
    }
  }));

  // Track connection state for status indicator
  wsUnsubscribers.push(subscribe((state) => {
    if (state.wsConnected) {
      updateStatusDisplay(true, `接続済 (${state.persons.length}名)`);
    } else {
      updateStatusDisplay(false, "再接続中...");
    }
  }));
}

// ── Dashboard Bootstrap ──────────────────────

let dashboardInitialized = false;

async function startDashboard() {
  if (!dom.dashboard) return;

  dom.dashboard.classList.remove("hidden");
  if (dom.userInfo) {
    dom.userInfo.textContent = getCurrentUser() || "";
  }

  if (dashboardInitialized) {
    await loadPersons();
    await loadSystemStatus();
    // Re-init office if needed
    initOfficeIfNeeded();
    return;
  }
  dashboardInitialized = true;

  // Initialize sub-modules
  initPerson(dom.personSelector, dom.paneState, onPersonSelected);
  initMemory(dom.memoryPanel);
  initSession(dom.paneHistory);

  // Bind right-panel tabs
  [dom.tabState, dom.tabActivity, dom.tabHistory].forEach((btn) => {
    btn?.addEventListener("click", () => activateRightTab(btn.dataset.tab));
  });

  // Bind conversation panel events
  dom.convBack?.addEventListener("click", closeConversation);
  dom.convOverlay?.addEventListener("click", (e) => {
    // Close when clicking the backdrop (outside the card)
    if (e.target === dom.convOverlay) closeConversation();
  });
  dom.convSend?.addEventListener("click", sendConversationMessage);
  dom.convInput?.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      sendConversationMessage();
    }
  });

  // Auto-resize conversation input
  dom.convInput?.addEventListener("input", () => {
    dom.convInput.style.height = "auto";
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, 100) + "px";
  });

  // Close conversation with Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && getState().conversationOpen) {
      closeConversation();
    }
  });

  // Bind logout
  dom.logoutBtn?.addEventListener("click", () => {
    dom.dashboard.classList.add("hidden");
    logout();
  });

  // Load data
  await loadPersons();
  await loadSystemStatus();

  // Connect WebSocket
  setupWebSocket();

  // Activate default right tab
  activateRightTab("state");

  // Auto-init 3D office (always visible now)
  initOfficeIfNeeded();
}

// ── Person Selection Callback ──────────────────────

async function onPersonSelected(name) {
  // Highlight desk in 3D
  if (getState().officeInitialized) {
    highlightDesk(name);
  }

  // Open conversation panel + load memory/sessions in parallel
  await Promise.all([
    openConversation(name),
    loadMemoryTab(getState().activeMemoryTab),
    loadSessions(),
  ]);
}

// ── Main Init ──────────────────────

export function init() {
  cacheDom();

  const savedUser = getCurrentUser();
  if (savedUser) {
    initLogin(dom.loginContainer, onLoginSuccess);
    startDashboard();
  } else {
    dom.dashboard?.classList.add("hidden");
    initLogin(dom.loginContainer, onLoginSuccess);
  }
}

function onLoginSuccess(_username) {
  startDashboard();
}

// Auto-init on DOM ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
