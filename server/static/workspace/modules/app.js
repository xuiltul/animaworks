// ── App Entry Point ──────────────────────
// Initialization, screen switching, and event delegation.

import { getState, setState, subscribe } from "./state.js";
import { fetchSystemStatus, fetchConversationFull, greetAnima } from "./api.js";
import { connect, onEvent } from "./websocket.js";
import { initLogin, getCurrentUser, logout } from "./login.js";
import { initAnima, loadAnimas, selectAnima, renderAnimaSelector, renderStatus } from "./anima.js";
import { initMemory, loadMemoryTab } from "./memory.js";
import { initSession, loadSessions } from "./session.js";
import { escapeHtml, renderSimpleMarkdown, smartTimestamp } from "./utils.js";
import { initOffice, getDesks, highlightDesk, setCharacterClickHandler, getScene, registerClickTarget, setCharacterUpdateHook, getObstacles, getFloorDimensions } from "./office3d.js";
import { initCharacters, createCharacter, removeCharacter, updateCharacterState, updateAllCharacters, getCharacterGroup, getCharacterHome, setAppearance } from "./character.js";
import { initBustup, setCharacter, setExpression, setTalking, onClick as onBustupClick, setLive2dAppearance } from "./live2d.js";
import { createNavGrid } from "./navigation.js";
import { initMovement, registerCharacter, updateMovements, moveTo, moveToHome, stopMovement, isMoving } from "./movement.js";
import { computePOIs, initIdleBehaviors, updateIdleBehaviors, cancelBehavior } from "./idle_behavior.js";
import { initInteractions, showMessageEffect, showConversation, updateInteractions } from "./interactions.js";
import { initTimeline, addTimelineEvent, loadHistory, localISOString } from "./timeline.js";
import { initMessagePopup, isVisible as isMessagePopupVisible, hide as hideMessagePopup } from "./message-popup.js";
import { playReveal } from "./reveal.js";
import { streamChat, fetchActiveStream, fetchStreamProgress } from "../../shared/chat-stream.js";
import { SwipeHandler } from "../../modules/touch.js";
import { createLogger } from "../../shared/logger.js";
import { createImageInput, initLightbox, renderChatImages } from "../../shared/image-input.js";
import { getIcon } from "../../shared/activity-types.js";

const logger = createLogger("ws-app");

// ── Mobile Resource Tracking ────────────
let _swiperInstance = null;
let _mobileMediaQuery = null;

function _cleanupMobileResources() {
  if (_swiperInstance) {
    _swiperInstance.destroy();
    _swiperInstance = null;
  }
  if (_mobileMediaQuery) {
    _mobileMediaQuery.removeEventListener("change", updateConvInputPlaceholder);
    _mobileMediaQuery = null;
  }
}

// ── DOM References ──────────────────────

const dom = {};

function cacheDom() {
  dom.loginContainer = document.getElementById("wsLogin");
  dom.dashboard = document.getElementById("wsDashboard");
  dom.animaSelector = document.getElementById("wsAnimaSelector");
  dom.systemStatus = document.getElementById("wsSystemStatus");
  dom.userInfo = document.getElementById("wsUserInfo");
  dom.rightTabs = document.getElementById("wsRightTabs");
  dom.tabState = document.getElementById("wsTabState");
  dom.tabActivity = document.getElementById("wsTabActivity");
  dom.tabBoard = document.getElementById("wsTabBoard");
  dom.tabHistory = document.getElementById("wsTabHistory");
  dom.paneState = document.getElementById("wsPaneState");
  dom.paneActivity = document.getElementById("wsPaneActivity");
  dom.paneBoard = document.getElementById("wsPaneBoard");
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
  dom.convAnimaName = document.getElementById("wsConvAnimaName");
  dom.convCanvas = document.getElementById("wsConvCanvas");
  dom.convMessages = document.getElementById("wsConvMessages");
  dom.convInput = document.getElementById("wsConvInput");
  dom.convSend = document.getElementById("wsConvSend");
  dom.convPreviewBar = document.getElementById("wsConvPreviewBar");
  dom.convAttachBtn = document.getElementById("wsConvAttachBtn");
  dom.convFileInput = document.getElementById("wsConvFileInput");

  // Mobile controls
  dom.mobileSidebarToggle = document.getElementById("wsMobileSidebarToggle");
  dom.mobileCharacterToggle = document.getElementById("wsMobileCharacterToggle");
  dom.sidebarBackdrop = document.getElementById("wsSidebarBackdrop");
  dom.mobileMemoryClose = document.getElementById("wsMobileMemoryClose");
  dom.convSidebar = document.querySelector(".ws-conv-sidebar");
  dom.convCharacter = document.querySelector(".ws-conv-character");
}

// ── Activity Feed ──────────────────────

function addActivity(type, animaName, summary, isoTs) {
  if (!dom.paneActivity) return;

  const d = isoTs ? new Date(isoTs) : new Date();
  const ts = d.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
  const icon = getIcon(type);

  const entry = document.createElement("div");
  entry.className = "activity-entry";
  entry.innerHTML = `
    <span class="activity-time">${ts}</span>
    <span class="activity-icon">${icon}</span>
    <span class="activity-anima">${escapeHtml(animaName)}</span>
    <span class="activity-summary">${escapeHtml(summary)}</span>`;

  dom.paneActivity.prepend(entry);

  // Cap at 200 entries
  while (dom.paneActivity.children.length > 200) {
    dom.paneActivity.removeChild(dom.paneActivity.lastChild);
  }
}

// ── Activity History Loading ──────────────────────

let _activityHistoryLoaded = false;

async function loadActivityHistory() {
  if (_activityHistoryLoaded || !dom.paneActivity) return;
  _activityHistoryLoaded = true;

  try {
    const res = await fetch("/api/activity/recent?hours=24&limit=50&offset=0");
    if (!res.ok) return;
    const data = await res.json();
    const events = data.events || [];

    for (const evt of events) {
      const type = evt.type || "system";
      const anima = evt.anima || evt.animas || "";
      const summary = evt.summary || evt.content || "";
      const ts = evt.ts || evt.timestamp || "";

      addActivity(type, typeof anima === "string" ? anima : String(anima), summary, ts);
    }
  } catch (err) {
    logger.error("Failed to load activity history", { error: err.message });
  }
}

// ── Right Panel Tabs ──────────────────────

function activateRightTab(tab) {
  setState({ activeRightTab: tab });

  [dom.tabState, dom.tabActivity, dom.tabBoard, dom.tabHistory].forEach((btn) => {
    btn?.classList.toggle("active", btn.dataset.tab === tab);
  });

  [dom.paneState, dom.paneActivity, dom.paneBoard, dom.paneHistory].forEach((pane) => {
    if (pane) pane.style.display = pane.dataset.pane === tab ? "" : "none";
  });

  if (tab === "history") {
    loadSessions();
  }
  if (tab === "board") {
    initBoardTab();
  }
  if (tab === "activity") {
    loadActivityHistory();
  }
}

// ── 3D Office Initialization ──────────────────────

async function initOfficeIfNeeded() {
  if (getState().officeInitialized) return;
  setState({ officeInitialized: true });

  try {
    // Initialize 3D scene with anima data for dynamic desk layout
    const { animas } = getState();
    initOffice(dom.officeCanvas, animas);

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
    for (const p of animas) {
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
        const animState = mapAnimaStatusToAnim(p.status);
        updateCharacterState(p.name, animState);
      }
    }

    // ── Navigation + Movement system init ──────────
    const { width: floorW, depth: floorD } = getFloorDimensions();
    const obstacles = getObstacles();
    const navGrid = createNavGrid(floorW, floorD, obstacles);
    initMovement(navGrid, floorW, floorD);

    // Register all characters with the movement system
    for (const p of animas) {
      const group = getCharacterGroup(p.name);
      const home = getCharacterHome(p.name);
      if (group && home) {
        registerCharacter(p.name, group, home);
      }
    }

    // ── Interactions + Idle Behaviors + Timeline init ──────────
    const movementSystem = { moveTo, moveToHome, stopMovement, isMoving };
    const characterMap = Object.fromEntries(animas.map((p) => [p.name, true]));
    initInteractions(scene, characterMap, movementSystem);

    const pois = computePOIs(floorW, floorD);
    initIdleBehaviors(characterMap, movementSystem, pois);

    initTimeline(dom.officePanel, { showMessageEffect, showConversation });
    initMessagePopup(dom.officePanel);
    loadHistory(24);

    // Handle character clicks → open conversation in right panel
    // selectAnima → onAnimaSelected → openConversation の一本化フロー
    setCharacterClickHandler((animaName) => {
      selectAnima(animaName);
    });

    // Highlight selected anima's desk
    const { selectedAnima } = getState();
    if (selectedAnima) {
      highlightDesk(selectedAnima);
    }
  } catch (err) {
    logger.error("Failed to initialize 3D office", { error: err.message });
    setState({ officeInitialized: false });
  }
}

function mapAnimaStatusToAnim(status) {
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
let convImageInputManager = null;

async function openConversation(animaName) {
  if (!dom.convOverlay) return;

  setState({ conversationOpen: true, conversationAnima: animaName });

  // Show conversation overlay on top of office
  dom.convOverlay.classList.remove("hidden");

  // Update anima name + mobile placeholder
  if (dom.convAnimaName) dom.convAnimaName.textContent = animaName;
  updateConvInputPlaceholder();

  // Reset mobile panels on conversation open
  closeMobileSidebar();
  closeMobileCharacter();

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
  await setCharacter(animaName);
  setExpression("neutral");

  // Load and render chat history, then trigger greeting
  await loadAndRenderConvMessages(animaName);
  triggerGreeting(animaName);

  // Focus input
  dom.convInput?.focus();
}

function closeConversation() {
  if (!dom.convOverlay) return;

  // Hide conversation overlay
  dom.convOverlay.classList.add("hidden");

  // Close mobile panels
  closeMobileSidebar();
  closeMobileCharacter();
  closeMobileMemory();

  // Cleanup mobile resources
  _cleanupMobileResources();

  setState({ conversationOpen: false, conversationAnima: null });
  setTalking(false);

  // Abort any active stream
  if (convStreamController) {
    convStreamController.abort();
    convStreamController = null;
  }
}

// ── Greeting on Character Click ──────────────────────

let _greetingInFlight = false;
const _GREET_COOLDOWN_MS = 3600 * 1000; // 1 hour — matches server-side cooldown
const _lastGreetTime = {}; // { animaName: timestamp_ms }

async function triggerGreeting(animaName) {
  if (_greetingInFlight) return;

  // Frontend cooldown: skip if greeted within the last hour
  const lastTs = _lastGreetTime[animaName];
  if (lastTs && Date.now() - lastTs < _GREET_COOLDOWN_MS) return;

  _greetingInFlight = true;
  try {
    const data = await greetAnima(animaName);
    if (!data.response) return;

    // If server returned cached response, skip display
    if (data.cached) return;

    _lastGreetTime[animaName] = Date.now();
    const now = new Date().toISOString();

    // Add visit marker + greeting message to chat
    const { chatMessages } = getState();
    const newMessages = [
      ...chatMessages,
      { role: "system", text: "デスクを訪問しました", timestamp: now },
      { role: "assistant", text: data.response, timestamp: now },
    ];
    setState({ chatMessages: newMessages });
    renderConvMessages();

    // Update bust-up expression
    if (data.emotion) {
      setExpression(data.emotion);
      setTimeout(() => setExpression("neutral"), 3000);
    }
  } catch (err) {
    logger.error("Failed to greet", { anima: animaName, error: err.message });
  } finally {
    _greetingInFlight = false;
  }
}

// ── Chat Rendering in Conversation Panel ──────────────────────

function renderConvBubble(msg) {
  const ts = msg.timestamp ? smartTimestamp(msg.timestamp) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "system") {
    return `<div class="chat-visit-marker">${escapeHtml(msg.text)}${tsHtml}</div>`;
  }
  if (msg.role === "user") {
    const imagesHtml = renderChatImages(msg.images);
    const textHtml = msg.text ? `<div class="chat-text">${escapeHtml(msg.text)}</div>` : "";
    return `<div class="chat-bubble user">${imagesHtml}${textHtml}${tsHtml}</div>`;
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
  return `<div class="chat-bubble assistant${streamClass}">${content}${toolHtml}${tsHtml}</div>`;
}

function renderConvMessages() {
  if (!dom.convMessages) return;
  const { chatMessages } = getState();
  if (chatMessages.length === 0) {
    dom.convMessages.innerHTML = '<div class="chat-empty">メッセージはまだありません</div>';
    return;
  }
  dom.convMessages.innerHTML = chatMessages.map(renderConvBubble).join("");
  dom.convMessages.scrollTop = dom.convMessages.scrollHeight;
}

async function loadAndRenderConvMessages(animaName) {
  if (!animaName) return;
  try {
    const data = await fetchConversationFull(animaName);
    if (data.turns && data.turns.length > 0) {
      const messages = data.turns.map((t) => ({
        role: t.role === "human" ? "user" : t.role === "system" ? "system" : "assistant",
        text: t.content || "",
        timestamp: t.timestamp || "",
      }));
      setState({ chatMessages: messages });
    } else {
      setState({ chatMessages: [] });
    }
  } catch (err) {
    logger.error("Failed to load conversation", { anima: animaName, error: err.message });
    setState({ chatMessages: [] });
  }
  renderConvMessages();

  // Check for active stream to resume after page reload
  resumeConversationStream(animaName);
}

// ── Stream Resume (page reload recovery) ──────────────────────

async function resumeConversationStream(animaName) {
  if (convStreamController) return; // Already streaming

  try {
    const active = await fetchActiveStream(animaName);
    if (!active || active.status !== "streaming") return;

    const progress = await fetchStreamProgress(animaName, active.response_id);
    if (!progress) return;

    // Show accumulated text in streaming bubble
    const { chatMessages } = getState();
    const streamingMsg = {
      role: "assistant",
      text: progress.full_text || "",
      streaming: true,
      activeTool: progress.active_tool || null,
    };
    setState({ chatMessages: [...chatMessages, streamingMsg] });
    renderConvMessages();

    dom.convInput.disabled = true;
    dom.convSend.disabled = true;
    convStreamController = new AbortController();

    const resumeBody = JSON.stringify({
      message: "",
      from_person: getCurrentUser() || "guest",
      resume: active.response_id,
      last_event_id: progress.last_event_id || "",
    });

    await streamChat(animaName, resumeBody, convStreamController.signal, {
      onTextDelta: (deltaText) => {
        streamingMsg.text += deltaText;
        scheduleStreamingUpdate(streamingMsg);
      },
      onToolStart: (toolName) => {
        streamingMsg.activeTool = toolName;
        setExpression("thinking");
        updateStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {
        streamingMsg.activeTool = null;
        setExpression("neutral");
        updateStreamingBubble(streamingMsg);
      },
      onDone: ({ summary, emotion }) => {
        if (summary) streamingMsg.text = summary;
        if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        setExpression(emotion);
        setTimeout(() => setExpression("neutral"), 3000);
        setState({ chatMessages: [...getState().chatMessages] });
        renderConvMessages();
      },
      onError: ({ message: errorMsg }) => {
        streamingMsg.text += `\n[エラー: ${errorMsg}]`;
        streamingMsg.streaming = false;
        setExpression("troubled");
        setState({ chatMessages: [...getState().chatMessages] });
        renderConvMessages();
      },
    });

    setTalking(false);
    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
      setState({ chatMessages: [...getState().chatMessages] });
      renderConvMessages();
    }
  } catch (err) {
    logger.error("Resume stream error", { anima: animaName, error: err.message });
  } finally {
    convStreamController = null;
    if (dom.convInput) dom.convInput.disabled = false;
    if (dom.convSend) dom.convSend.disabled = false;
  }
}

// ── SSE Streaming for Conversation ──────────────────────

async function sendConversationMessage() {
  const text = dom.convInput?.value?.trim();
  const images = convImageInputManager?.getPendingImages() || [];
  if (!text && images.length === 0) return;

  const animaName = getState().conversationAnima;
  if (!animaName) return;

  // Capture display images (with dataUrl for rendering)
  const displayImages = convImageInputManager?.getDisplayImages() || [];

  // Clear input
  dom.convInput.value = "";
  dom.convInput.disabled = true;
  dom.convSend.disabled = true;

  // Add user message + streaming assistant placeholder
  const { chatMessages } = getState();
  const sendTs = new Date().toISOString();
  const userMsg = { role: "user", text: text || "", images: displayImages, timestamp: sendTs };
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: sendTs };
  setState({ chatMessages: [...chatMessages, userMsg, streamingMsg] });
  renderConvMessages();

  // Clear images after capturing
  convImageInputManager?.clearImages();

  // Create AbortController for cancellable streaming
  convStreamController = new AbortController();

  try {
    const userName = getCurrentUser() || "guest";
    const bodyObj = { message: text || "", from_person: userName };
    if (images.length > 0) {
      bodyObj.images = images;
    }
    const body = JSON.stringify(bodyObj);

    let talkingStarted = false;

    await streamChat(animaName, body, convStreamController.signal, {
      onTextDelta: (deltaText) => {
        streamingMsg.afterHeartbeatRelay = false;
        if (!talkingStarted) {
          setTalking(true);
          setExpression("neutral");
          talkingStarted = true;
        }
        streamingMsg.text += deltaText;
        scheduleStreamingUpdate(streamingMsg);
      },
      onToolStart: (toolName) => {
        streamingMsg.activeTool = toolName;
        setExpression("thinking");
        updateStreamingBubble(streamingMsg);
      },
      onToolEnd: () => {
        streamingMsg.activeTool = null;
        setExpression("neutral");
        updateStreamingBubble(streamingMsg);
      },
      onHeartbeatRelayStart: ({ message }) => {
        streamingMsg.heartbeatRelay = true;
        streamingMsg.heartbeatText = "";
        streamingMsg.text = "";
        scheduleStreamingUpdate(streamingMsg);
      },
      onHeartbeatRelay: ({ text }) => {
        streamingMsg.heartbeatText = (streamingMsg.heartbeatText || "") + text;
        scheduleStreamingUpdate(streamingMsg);
      },
      onHeartbeatRelayDone: () => {
        streamingMsg.heartbeatRelay = false;
        streamingMsg.heartbeatText = "";
        streamingMsg.afterHeartbeatRelay = true;
        scheduleStreamingUpdate(streamingMsg);
      },
      onDone: ({ summary, emotion }) => {
        // Use clean summary (emotion tag already stripped server-side)
        if (summary) {
          streamingMsg.text = summary;
          updateStreamingBubble(streamingMsg);
        }
        setExpression(emotion);
        setTimeout(() => setExpression("neutral"), 3000);
      },
      onError: ({ message: errorMsg }) => {
        setExpression("troubled");
        streamingMsg.text += `\n[エラー: ${errorMsg}]`;
        updateStreamingBubble(streamingMsg);
      },
    });

    setTalking(false);

    // Finalize streaming message
    streamingMsg.streaming = false;
    if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
    setState({ chatMessages: [...getState().chatMessages] });
    renderConvMessages();
  } catch (err) {
    if (err.name === "AbortError") return;
    logger.error("Conversation stream error", { anima: animaName, error: err.message, name: err.name });
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
  if (msg.heartbeatRelay) {
    html = '<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>ハートビート処理中...</div>';
    if (msg.heartbeatText) {
      html += `<div class="heartbeat-relay-text">${escapeHtml(msg.heartbeatText)}</div>`;
    }
  } else if (msg.afterHeartbeatRelay && !msg.text) {
    html = '<div class="heartbeat-relay-indicator"><span class="tool-spinner"></span>応答を準備中...</div>';
  } else if (msg.text) {
    html = renderSimpleMarkdown(msg.text);
  } else {
    html = '<span class="cursor-blink"></span>';
  }
  if (msg.activeTool) {
    html += `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`;
  }
  bubble.innerHTML = html;
  dom.convMessages.scrollTop = dom.convMessages.scrollHeight;
}

// ── Board Tab ──────────────────────

let _boardInitialized = false;
let _boardChannels = [];
let _boardDMs = [];
let _boardSelectedChannel = null;
let _boardSelectedType = null; // "channel" | "dm"

async function initBoardTab() {
  if (!dom.paneBoard) return;

  if (!_boardInitialized) {
    _boardInitialized = true;
    dom.paneBoard.innerHTML = `
      <div class="ws-board-tab">
        <div class="ws-board-dropdown">
          <select class="ws-board-select" id="wsBoardSelect">
            <option value="">読み込み中...</option>
          </select>
        </div>
        <div class="ws-board-messages" id="wsBoardMessages">
          <div class="loading-placeholder">チャンネルを選択してください</div>
        </div>
        <div class="ws-board-input" id="wsBoardInputArea">
          <textarea class="ws-board-input-field" id="wsBoardInput" placeholder="メッセージを入力..." rows="1"></textarea>
          <button class="ws-board-send-btn" id="wsBoardSend">送信</button>
        </div>
      </div>`;

    const select = document.getElementById("wsBoardSelect");
    const sendBtn = document.getElementById("wsBoardSend");
    const input = document.getElementById("wsBoardInput");

    select?.addEventListener("change", () => {
      const val = select.value;
      if (!val) return;
      const [type, name] = val.split(":", 2);
      _boardSelectedType = type;
      _boardSelectedChannel = name;
      loadBoardMessages();
    });

    sendBtn?.addEventListener("click", sendBoardMessage);
    input?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        sendBoardMessage();
      }
    });
  }

  await loadBoardChannelList();
}

async function loadBoardChannelList() {
  const select = document.getElementById("wsBoardSelect");
  if (!select) return;

  try {
    const [chRes, dmRes] = await Promise.all([
      fetch("/api/channels"),
      fetch("/api/dm"),
    ]);

    _boardChannels = chRes.ok ? await chRes.json() : [];
    _boardDMs = dmRes.ok ? await dmRes.json() : [];

    let html = '<option value="">-- チャンネルを選択 --</option>';

    if (_boardChannels.length > 0) {
      html += '<optgroup label="Channels">';
      for (const ch of _boardChannels) {
        const count = ch.message_count || 0;
        html += `<option value="channel:${escapeHtml(ch.name)}">#${escapeHtml(ch.name)} (${count})</option>`;
      }
      html += "</optgroup>";
    }

    if (_boardDMs.length > 0) {
      html += '<optgroup label="DM">';
      for (const dm of _boardDMs) {
        const pair = dm.pair || dm.participants?.join(" & ") || "?";
        const count = dm.message_count || 0;
        html += `<option value="dm:${escapeHtml(pair)}">${escapeHtml(pair)} (${count})</option>`;
      }
      html += "</optgroup>";
    }

    if (_boardChannels.length === 0 && _boardDMs.length === 0) {
      html = '<option value="">チャンネルがありません</option>';
    }

    select.innerHTML = html;

    // Restore selection if previously selected
    if (_boardSelectedChannel && _boardSelectedType) {
      select.value = `${_boardSelectedType}:${_boardSelectedChannel}`;
    }
  } catch (err) {
    logger.error("Failed to load board channels", { error: err.message });
    select.innerHTML = '<option value="">読み込み失敗</option>';
  }
}

async function loadBoardMessages() {
  const messagesEl = document.getElementById("wsBoardMessages");
  if (!messagesEl || !_boardSelectedChannel) return;

  messagesEl.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    let url;
    if (_boardSelectedType === "channel") {
      url = `/api/channels/${encodeURIComponent(_boardSelectedChannel)}?limit=50&offset=0`;
    } else {
      url = `/api/dm/${encodeURIComponent(_boardSelectedChannel)}?limit=50`;
    }

    const res = await fetch(url);
    if (!res.ok) {
      messagesEl.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
      return;
    }

    const data = await res.json();
    const messages = data.messages || [];

    if (messages.length === 0) {
      messagesEl.innerHTML = '<div class="loading-placeholder">メッセージはまだありません</div>';
      return;
    }

    messagesEl.innerHTML = messages.map(renderBoardMessage).join("");
    messagesEl.scrollTop = messagesEl.scrollHeight;
  } catch (err) {
    logger.error("Failed to load board messages", { error: err.message });
    messagesEl.innerHTML = '<div class="loading-placeholder">読み込み失敗</div>';
  }
}

function renderBoardMessage(msg) {
  const ts = msg.ts ? smartTimestamp(msg.ts) : "";
  const from = escapeHtml(msg.from || "?");
  const text = escapeHtml(msg.text || "");
  const humanBadge = msg.source === "human" ? ' <span class="ws-board-human-badge">[human]</span>' : "";
  return `<div class="ws-board-msg">
    <span class="ws-board-msg-time">${escapeHtml(ts)}</span>
    <span class="ws-board-msg-from">[${from}]${humanBadge}</span>
    <span class="ws-board-msg-text">${text}</span>
  </div>`;
}

function appendBoardMessage(msg) {
  const messagesEl = document.getElementById("wsBoardMessages");
  if (!messagesEl) return;

  // Remove placeholder if present
  const placeholder = messagesEl.querySelector(".loading-placeholder");
  if (placeholder) placeholder.remove();

  const div = document.createElement("div");
  div.innerHTML = renderBoardMessage(msg);
  const el = div.firstElementChild;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendBoardMessage() {
  const input = document.getElementById("wsBoardInput");
  const text = input?.value?.trim();
  if (!text || !_boardSelectedChannel || _boardSelectedType !== "channel") return;

  const userName = getCurrentUser() || "guest";
  input.value = "";

  try {
    const res = await fetch(`/api/channels/${encodeURIComponent(_boardSelectedChannel)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, from_name: userName }),
    });
    if (!res.ok) {
      logger.error("Failed to send board message", { status: res.status });
    }
    // The message will appear via WebSocket board.post event
  } catch (err) {
    logger.error("Failed to send board message", { error: err.message });
  }
}

// ── System Status ──────────────────────

async function loadSystemStatus() {
  if (!dom.systemStatus) return;
  try {
    const data = await fetchSystemStatus();
    updateStatusDisplay(
      data.scheduler_running,
      `${data.scheduler_running ? "稼働中" : "停止"} (${data.animas}名)`
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
const lastAnimaStatus = {};

function setupWebSocket() {
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
    // Update 3D character animation
    if (getState().officeInitialized) {
      const animState = mapAnimaStatusToAnim(data.status);
      updateCharacterState(data.name, animState);
      setState({ characterStates: { ...getState().characterStates, [data.name]: animState } });
    }
    // Only log to activity feed when status actually changes
    if (lastAnimaStatus[data.name] !== data.status) {
      lastAnimaStatus[data.name] = data.status;
      addActivity("system", data.name, `Status: ${data.status}`);
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
  }));

  // ── board.post — shared channel message ──
  wsUnsubscribers.push(onEvent("board.post", (data) => {
    const from = data.from || "?";
    const channel = data.channel || "?";
    const text = data.text || "";

    // Update Board tab if matching channel is selected
    if (_boardSelectedType === "channel" && _boardSelectedChannel === channel) {
      appendBoardMessage({
        ts: data.ts || new Date().toISOString(),
        from,
        text,
        source: data.source || "",
      });
    }

    // Add to activity feed
    addActivity("chat", from, `[#${channel}] ${text}`);

    // Add to timeline
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
  wsUnsubscribers.push(onEvent("anima.notification", (data) => {
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

// ── Mobile Responsive Helpers ──────────────────────

/** @returns {boolean} */
function isMobileView() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function openMobileSidebar() {
  dom.convSidebar?.classList.add("mobile-open");
  dom.sidebarBackdrop?.classList.add("visible");
}

function closeMobileSidebar() {
  dom.convSidebar?.classList.remove("mobile-open");
  dom.sidebarBackdrop?.classList.remove("visible");
}

function toggleMobileCharacter() {
  dom.convCharacter?.classList.toggle("mobile-open");
}

function closeMobileCharacter() {
  dom.convCharacter?.classList.remove("mobile-open");
}

function openMobileMemory() {
  dom.memoryPanel?.classList.add("mobile-open");
}

function closeMobileMemory() {
  dom.memoryPanel?.classList.remove("mobile-open");
}

function initMobileControls() {
  // Sidebar toggle
  dom.mobileSidebarToggle?.addEventListener("click", () => {
    if (dom.convSidebar?.classList.contains("mobile-open")) {
      closeMobileSidebar();
    } else {
      closeMobileCharacter();
      openMobileSidebar();
    }
  });

  // Character toggle
  dom.mobileCharacterToggle?.addEventListener("click", () => {
    if (dom.convCharacter?.classList.contains("mobile-open")) {
      closeMobileCharacter();
    } else {
      closeMobileSidebar();
      toggleMobileCharacter();
    }
  });

  // Backdrop closes sidebar
  dom.sidebarBackdrop?.addEventListener("click", closeMobileSidebar);

  // Memory close button
  dom.mobileMemoryClose?.addEventListener("click", closeMobileMemory);

  // Update conversation input placeholder based on screen size
  updateConvInputPlaceholder();
  _mobileMediaQuery = window.matchMedia("(max-width: 768px)");
  _mobileMediaQuery.addEventListener("change", updateConvInputPlaceholder);
}

function updateConvInputPlaceholder() {
  if (!dom.convInput) return;
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  dom.convInput.placeholder = isMobileView()
    ? `${animaName} にメッセージ... (Enter)`
    : `メッセージを入力... (Ctrl+Enter)`;
}

function initTouchGestures() {
  if (!("ontouchstart" in window)) return;

  const overlay = dom.convOverlay;
  if (!overlay) return;

  _swiperInstance = new SwipeHandler(overlay);

  // Right swipe from left edge: open sidebar
  _swiperInstance.onSwipeRight((info) => {
    if (!isMobileView()) return;
    if (info.startX < 30) {
      openMobileSidebar();
    }
  });

  // Left swipe: close sidebar
  _swiperInstance.onSwipeLeft(() => {
    if (!isMobileView()) return;
    closeMobileSidebar();
  });
}

/** Set CSS custom property for viewport height fallback (svh-unsupported browsers). */
function initViewportHeightFallback() {
  if (CSS.supports && CSS.supports('height', '100svh')) return;

  function setVh() {
    const vh = window.visualViewport?.height || window.innerHeight;
    document.documentElement.style.setProperty('--vh-fallback', `${vh}px`);
  }
  setVh();
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', setVh);
  }
  window.addEventListener('resize', setVh);
}

/** Toggle timeline collapse on iPad-width viewports. */
function initTimelineCollapseToggle() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.timeline-toggle-btn');
    if (!btn) return;
    const timeline = btn.closest('.ws-timeline');
    if (timeline) {
      timeline.classList.toggle('collapsed');
    }
  });
}

function initMobileKeyboard() {
  const vv = window.visualViewport;

  function scrollInputIntoView() {
    const active = document.activeElement;
    if (!active?.matches(".ws-conv-input, .chat-input")) return;
    requestAnimationFrame(() => {
      active.scrollIntoView({ block: "nearest" });
    });
  }

  if (vv) {
    vv.addEventListener("resize", scrollInputIntoView);
  } else {
    // Fallback for browsers without visualViewport
    let lastHeight = window.innerHeight;
    window.addEventListener("resize", () => {
      const current = window.innerHeight;
      if (Math.abs(current - lastHeight) > 100) {
        scrollInputIntoView();
      }
      lastHeight = current;
    });
  }
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
    await loadAnimas();
    await loadSystemStatus();
    // Re-init office if needed
    initOfficeIfNeeded();
    return;
  }
  dashboardInitialized = true;

  // Initialize sub-modules
  initAnima(dom.animaSelector, dom.paneState, onAnimaSelected);
  initMemory(dom.memoryPanel);
  initSession(dom.paneHistory);

  // Bind right-panel tabs
  [dom.tabState, dom.tabActivity, dom.tabBoard, dom.tabHistory].forEach((btn) => {
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
    if (e.key === "Enter") {
      if (isMobileView()) {
        // Mobile: Enter sends, Shift+Enter inserts newline
        if (!e.shiftKey) {
          e.preventDefault();
          sendConversationMessage();
        }
      } else {
        // Desktop: Ctrl/Cmd+Enter sends
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          sendConversationMessage();
        }
      }
    }
  });

  // Auto-resize conversation input (100px max on mobile, 120px on desktop)
  dom.convInput?.addEventListener("input", () => {
    dom.convInput.style.height = "auto";
    const maxH = isMobileView() ? 100 : 120;
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
  });

  // ── Conversation Image Input Setup ────────────────
  if (dom.convAttachBtn && dom.convFileInput) {
    dom.convAttachBtn.addEventListener("click", () => dom.convFileInput.click());
    dom.convFileInput.addEventListener("change", () => {
      if (dom.convFileInput.files.length > 0) {
        convImageInputManager?.addFiles(dom.convFileInput.files);
        dom.convFileInput.value = "";
      }
    });
  }

  const convMain = document.querySelector(".ws-conv-main");
  if (convMain && dom.convInput && dom.convPreviewBar) {
    convImageInputManager = createImageInput({
      container: convMain,
      inputArea: dom.convInput,
      previewContainer: dom.convPreviewBar,
    });
  }

  // Initialize lightbox for image clicks
  initLightbox();

  // Close popups / conversation with Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (isMessagePopupVisible()) {
        hideMessagePopup();
      } else if (getState().conversationOpen) {
        closeConversation();
      }
    }
  });

  // Bind logout
  dom.logoutBtn?.addEventListener("click", () => {
    dom.dashboard.classList.add("hidden");
    logout();
  });

  // Load data
  await loadAnimas();
  await loadSystemStatus();

  // Connect WebSocket
  setupWebSocket();

  // Activate default right tab
  activateRightTab("state");

  // Auto-init 3D office (always visible now)
  initOfficeIfNeeded();

  // Viewport & mobile responsive features
  initViewportHeightFallback();
  initTimelineCollapseToggle();
  initMobileControls();
  initTouchGestures();
  initMobileKeyboard();
}

// ── Anima Selection Callback ──────────────────────

async function onAnimaSelected(name) {
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
