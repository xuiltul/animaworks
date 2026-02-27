// ── App Entry Point ──────────────────────
// Initialization, screen switching, and event delegation.

import { initI18n, applyTranslations } from "/shared/i18n.js";
import { getState, setState, subscribe } from "./state.js";
import { fetchSystemStatus, fetchConversationHistory, greetAnima } from "./api.js";
import { connect, onEvent } from "./websocket.js";
import { initLogin, getCurrentUser, logout } from "./login.js";
import { initAnima, loadAnimas, selectAnima, renderAnimaSelector, renderStatus } from "./anima.js";
import { initMemory, loadMemoryTab } from "./memory.js";
import { initSession, loadSessions } from "./session.js";
import { escapeHtml, renderSimpleMarkdown, smartTimestamp } from "./utils.js";
import { initOffice, disposeOffice, getDesks, highlightDesk, setCharacterClickHandler, getScene, registerClickTarget, setCharacterUpdateHook, getObstacles, getFloorDimensions } from "./office3d.js";
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
import { initVoiceUI, destroyVoiceUI } from "../../modules/voice-ui.js";
import { getIcon } from "../../shared/activity-types.js";
import { initOrgDashboard, disposeOrgDashboard, updateAnimaStatus, addActivityItem } from "./org-dashboard.js";

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
  dom.orgPanel = document.getElementById("wsOrgPanel");
  dom.viewToggle = document.getElementById("wsViewToggle");

  // Conversation overlay (3-column)
  dom.convOverlay = document.getElementById("wsConvOverlay");
  dom.convLayout = document.getElementById("wsConvLayout");
  dom.convBack = document.getElementById("wsConvBack");
  dom.convAnimaName = document.getElementById("wsConvAnimaName");
  dom.threadTabs = document.getElementById("wsThreadTabs");
  dom.convCanvas = document.getElementById("wsConvCanvas");
  dom.convMessages = document.getElementById("wsConvMessages");
  dom.convInput = document.getElementById("wsConvInput");
  dom.convSend = document.getElementById("wsConvSend");
  dom.convPreviewBar = document.getElementById("wsConvPreviewBar");
  dom.convAttachBtn = document.getElementById("wsConvAttachBtn");
  dom.convFileInput = document.getElementById("wsConvFileInput");
  dom.convPending = document.getElementById("wsConvPending");
  dom.convPendingList = document.getElementById("wsConvPendingList");
  dom.convPendingLabel = document.getElementById("wsConvPendingLabel");
  dom.convPendingCancel = document.getElementById("wsConvPendingCancel");
  dom.convQueueBtn = document.getElementById("wsConvQueueBtn");

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
  if (window.lucide) lucide.createIcons({ nodes: [entry] });

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

// ── View Switching ──────────────────────

let _currentView = null; // '3d' | 'org'

function getDefaultView() {
  const theme = localStorage.getItem("aw-theme") || "default";
  return theme === "business" ? "org" : "3d";
}

function getCurrentView() {
  return localStorage.getItem("aw-workspace-view") || getDefaultView();
}

async function switchView(view) {
  if (_currentView === view) return;
  _currentView = view;
  localStorage.setItem("aw-workspace-view", view);

  if (view === "org") {
    // Dispose 3D resources before switching
    disposeOffice();
    setState({ officeInitialized: false });

    dom.officePanel.classList.add("hidden");
    dom.orgPanel.classList.remove("hidden");

    const { animas } = getState();
    await initOrgDashboard(dom.orgPanel, animas, {
      onNodeClick: (name) => selectAnima(name),
    });
  } else {
    // Dispose org dashboard before switching
    dom.orgPanel.classList.add("hidden");
    dom.officePanel.classList.remove("hidden");
    disposeOrgDashboard();

    await initOfficeIfNeeded();
  }

  updateViewToggle();
}

function updateViewToggle() {
  if (!dom.viewToggle) return;
  const is3d = _currentView === "3d";
  dom.viewToggle.querySelector(".ws-view-toggle-3d").style.fontWeight = is3d ? "700" : "400";
  dom.viewToggle.querySelector(".ws-view-toggle-org").style.fontWeight = is3d ? "400" : "700";
}

// ── Status Mapping ──────────────────────

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
let convPendingQueue = [];      // Array<{ text, images, displayImages }>

// ── Conversation Draft Persistence ──────────

function _wsDraftKey(animaName) {
  const user = getCurrentUser() || "guest";
  const anima = animaName || "_";
  return `aw:draft:workspace-conv:${user}:${anima}`;
}

function _wsSaveDraft() {
  const animaName = getState().conversationAnima;
  if (!animaName || !dom.convInput) return;
  localStorage.setItem(_wsDraftKey(animaName), dom.convInput.value || "");
}

function _wsLoadDraft(animaName) {
  if (!animaName) return "";
  return localStorage.getItem(_wsDraftKey(animaName)) || "";
}

function _wsClearDraft(animaName) {
  if (!animaName) return;
  localStorage.removeItem(_wsDraftKey(animaName));
}

// ── History State (Session-Aware Rendering) ──────────
const _historyState = {};  // { [animaName]: { [threadId]: { sessions, hasMore, nextBefore, loading } } }
const HISTORY_PAGE_SIZE = 50;
const TOOL_RESULT_TRUNCATE = 500;
let _scrollObserver = null;

async function openConversation(animaName) {
  if (!dom.convOverlay) return;

  _wsSaveDraft();
  destroyVoiceUI();
  setState({ conversationOpen: true, conversationAnima: animaName, activeThreadId: "default" });

  const { threads } = getState();
  if (!threads[animaName]) {
    setState({ threads: { ...threads, [animaName]: [{ id: "default", label: "メイン", unread: false }] } });
  }

  // Show conversation overlay on top of office
  dom.convOverlay.classList.remove("hidden");

  // Update anima name + mobile placeholder
  if (dom.convAnimaName) dom.convAnimaName.textContent = animaName;
  updateConvInputPlaceholder();
  if (dom.convInput) {
    dom.convInput.value = _wsLoadDraft(animaName);
    dom.convInput.style.height = "auto";
    const maxH = isMobileView() ? 100 : 120;
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
  }

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
  _renderWsThreadTabs();
  triggerGreeting(animaName);

  // Focus input
  dom.convInput?.focus();

  // Initialize voice input for conversation
  const convInputArea = document.querySelector(".ws-conv-input-area");
  if (convInputArea && animaName) {
    initVoiceUI(convInputArea, animaName, _buildVoiceChatCallbacks(animaName));
  }
}

// ── Thread Tabs (Workspace) ──────────────────────

function _renderWsThreadTabs() {
  const container = dom.threadTabs;
  const animaName = getState().conversationAnima;
  if (!container || !animaName) return;

  const list = getState().threads[animaName] || [{ id: "default", label: "メイン", unread: false }];
  const activeThreadId = getState().activeThreadId || "default";

  let html = "";
  for (const t of list) {
    const activeClass = t.id === activeThreadId ? " active" : "";
    const closeBtn = t.id !== "default"
      ? ` <button type="button" class="thread-tab-close" data-thread="${escapeHtml(t.id)}" title="スレッドを閉じる" aria-label="閉じる">&times;</button>`
      : "";
    html += `<span class="thread-tab-wrap"><button type="button" class="thread-tab${activeClass}" data-thread="${escapeHtml(t.id)}">${escapeHtml(t.label)}</button>${closeBtn}</span>`;
  }
  html += `<button type="button" class="thread-tab-new" id="wsNewThreadBtn" title="新しいスレッド">＋</button>`;

  container.innerHTML = html;

  container.querySelectorAll(".thread-tab").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const tid = e.target.dataset.thread;
      if (tid) _selectWsThread(tid);
    });
  });
  container.querySelectorAll(".thread-tab-close").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const tid = e.target.dataset.thread;
      if (tid) _closeWsThread(tid);
    });
  });
  const newBtn = document.getElementById("wsNewThreadBtn");
  if (newBtn) newBtn.addEventListener("click", () => _createWsNewThread());
}

async function _selectWsThread(threadId) {
  const current = getState().activeThreadId;
  if (threadId === current) return;

  setState({ activeThreadId: threadId });
  _renderWsThreadTabs();

  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const hs = _historyState[animaName]?.[threadId];
  const needLoad = !hs || hs.sessions.length === 0;

  if (needLoad) {
    if (!_historyState[animaName]) _historyState[animaName] = {};
    _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: true };
    renderConvMessages();

    try {
      const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, null, threadId);
      if (data && data.sessions && data.sessions.length > 0) {
        _historyState[animaName][threadId] = {
          sessions: data.sessions,
          hasMore: data.has_more || false,
          nextBefore: data.next_before || null,
          loading: false,
        };
      } else {
        _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
      }
    } catch {
      _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
    }
  }

  renderConvMessages();
  _observeSentinel();
}

function _createWsNewThread() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const threadId = crypto.randomUUID().slice(0, 8);
  const { threads, chatMessagesByThread } = getState();
  const list = threads[animaName] || [{ id: "default", label: "メイン", unread: false }];
  list.push({ id: threadId, label: "新しいスレッド", unread: false });

  const nextThreads = { ...threads, [animaName]: list };
  const nextByThread = { ...chatMessagesByThread };
  if (!nextByThread[animaName]) nextByThread[animaName] = {};
  nextByThread[animaName][threadId] = [];

  setState({
    threads: nextThreads,
    chatMessagesByThread: nextByThread,
    activeThreadId: threadId,
  });

  if (!_historyState[animaName]) _historyState[animaName] = {};
  _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };

  _renderWsThreadTabs();
  renderConvMessages();
  _observeSentinel();
}

function _closeWsThread(threadId) {
  if (threadId === "default") return;

  const animaName = getState().conversationAnima;
  if (!animaName) return;

  const { threads, chatMessagesByThread, activeThreadId } = getState();
  const list = threads[animaName];
  if (!list) return;

  const idx = list.findIndex((t) => t.id === threadId);
  if (idx < 0) return;

  const nextList = list.filter((t) => t.id !== threadId);
  const nextThreads = { ...threads, [animaName]: nextList };
  const nextByThread = { ...chatMessagesByThread };
  if (nextByThread[animaName]) {
    const { [threadId]: _, ...rest } = nextByThread[animaName];
    nextByThread[animaName] = rest;
  }
  delete _historyState[animaName]?.[threadId];

  const switchToDefault = activeThreadId === threadId;
  setState({
    threads: nextThreads,
    chatMessagesByThread: nextByThread,
    ...(switchToDefault ? { activeThreadId: "default" } : {}),
  });

  _renderWsThreadTabs();
  renderConvMessages();
  _observeSentinel();
}

function closeConversation() {
  if (!dom.convOverlay) return;

  _wsSaveDraft();
  // Hide conversation overlay
  dom.convOverlay.classList.add("hidden");

  // Close mobile panels
  closeMobileSidebar();
  closeMobileCharacter();
  closeMobileMemory();

  // Cleanup mobile resources
  _cleanupMobileResources();

  // Cleanup infinite scroll observer
  if (_scrollObserver) {
    _scrollObserver.disconnect();
    _scrollObserver = null;
  }

  // Clear history state for this anima
  const animaName = getState().conversationAnima;
  if (animaName) {
    delete _historyState[animaName];
  }

  setState({ conversationOpen: false, conversationAnima: null });
  setTalking(false);

  // Clear pending queue
  convPendingQueue = [];
  _wsHidePendingIndicator();

  // Abort any active stream
  if (convStreamController) {
    convStreamController.abort();
    convStreamController = null;
  }

  // Destroy voice UI
  destroyVoiceUI();
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

    // Add visit marker + greeting message to chat (thread-aware)
    const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
    const threadId = activeThreadId || "default";
    const current = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
    const newMessages = [
      ...current,
      { role: "system", text: "デスクを訪問しました", timestamp: now },
      { role: "assistant", text: data.response, timestamp: now },
    ];
    const nextByThread = { ...chatMessagesByThread };
    if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
    nextByThread[conversationAnima][threadId] = newMessages;
    setState({ chatMessagesByThread: nextByThread });
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

// ── Tool Call Rendering ──────────────────────

function _renderToolCalls(toolCalls) {
  if (!toolCalls || toolCalls.length === 0) return "";

  return toolCalls.map((tc, idx) => {
    const errorClass = tc.is_error ? " tool-call-error" : "";
    const toolName = escapeHtml(tc.tool_name || "unknown");
    const errorLabel = tc.is_error ? " [ERROR]" : "";

    return `<div class="tool-call-row${errorClass}" data-tool-idx="${idx}">` +
      `<span class="tool-call-row-icon">\u25B6</span>` +
      `<span class="tool-call-row-name">${toolName}${errorLabel}</span>` +
      `</div>` +
      `<div class="tool-call-detail" data-tool-idx="${idx}" style="display:none;">` +
      _renderToolCallDetail(tc) +
      `</div>`;
  }).join("");
}

function _renderToolCallDetail(tc) {
  let html = "";

  const input = tc.input || "";
  if (input) {
    const inputStr = typeof input === "string" ? input : JSON.stringify(input, null, 2);
    html += `<div class="tool-call-label">\u5165\u529B</div>`;
    html += `<div class="tool-call-content">${escapeHtml(inputStr)}</div>`;
  }

  const result = tc.result || "";
  if (result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result, null, 2);
    html += `<div class="tool-call-label">\u7D50\u679C</div>`;
    if (resultStr.length > TOOL_RESULT_TRUNCATE) {
      const truncated = resultStr.slice(0, TOOL_RESULT_TRUNCATE);
      html += `<div class="tool-call-content" data-full-result="${escapeHtml(resultStr)}">${escapeHtml(truncated)}...</div>`;
      html += `<button class="tool-call-show-more">\u3082\u3063\u3068\u898B\u308B</button>`;
    } else {
      html += `<div class="tool-call-content">${escapeHtml(resultStr)}</div>`;
    }
  }

  return html;
}

function _bindToolCallHandlers(container) {
  if (!container) return;

  container.querySelectorAll(".tool-call-row").forEach(row => {
    row.addEventListener("click", () => {
      const idx = row.dataset.toolIdx;
      const detail = row.nextElementSibling;
      if (!detail || detail.dataset.toolIdx !== idx) return;

      const isExpanded = row.classList.contains("expanded");
      if (isExpanded) {
        row.classList.remove("expanded");
        detail.style.display = "none";
      } else {
        row.classList.add("expanded");
        detail.style.display = "";
      }
    });
  });

  container.querySelectorAll(".tool-call-show-more").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const contentEl = btn.previousElementSibling;
      if (!contentEl) return;
      const fullResult = contentEl.dataset.fullResult;
      if (fullResult) {
        contentEl.textContent = fullResult;
        delete contentEl.dataset.fullResult;
        btn.remove();
      }
    });
  });
}

// ── Session Divider Rendering ──────────────────────

function _renderSessionDivider(session, isFirst) {
  if (isFirst) return "";

  const trigger = session.trigger || "chat";
  let label = "";
  let extraClass = "";

  if (trigger === "heartbeat") {
    label = "\u2764 \u30CF\u30FC\u30C8\u30D3\u30FC\u30C8";
    extraClass = " session-divider-heartbeat";
  } else if (trigger === "cron") {
    label = "\u23F0 Cron\u30BF\u30B9\u30AF";
    extraClass = " session-divider-cron";
  } else {
    const ts = session.session_start ? smartTimestamp(session.session_start) : "";
    label = ts;
  }

  return `<div class="session-divider${extraClass}">` +
    `<span class="session-divider-label">${escapeHtml(label)}</span>` +
    `</div>`;
}

// ── History Message Rendering ──────────────────────

function _renderHistoryMessage(msg) {
  const ts = msg.ts ? smartTimestamp(msg.ts) : "";
  const tsHtml = ts ? `<span class="chat-ts">${escapeHtml(ts)}</span>` : "";

  if (msg.role === "system") {
    return `<div class="chat-bubble assistant" style="opacity:0.7; font-style:italic;">${escapeHtml(msg.content || "")}${tsHtml}</div>`;
  }

  if (msg.role === "assistant") {
    const content = msg.content ? renderSimpleMarkdown(msg.content) : "";
    const toolHtml = _renderToolCalls(msg.tool_calls);
    return `<div class="chat-bubble assistant">${content}${toolHtml}${tsHtml}</div>`;
  }

  // human / user
  const fromLabel = msg.from_person && msg.from_person !== "human"
    ? `<div style="font-size:0.72rem; opacity:0.7; margin-bottom:2px;">${escapeHtml(msg.from_person)}</div>`
    : "";
  return `<div class="chat-bubble user">${fromLabel}<div class="chat-text">${escapeHtml(msg.content || "")}</div>${tsHtml}</div>`;
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
  let thinkingHtml = "";
  if (msg.thinkingText) {
    const thSummary = `Thinking (${msg.thinkingText.length} chars)`;
    const thRendered = renderSimpleMarkdown(msg.thinkingText);
    thinkingHtml = `<details class="thinking-block"><summary class="thinking-summary"><span class="thinking-icon">💭</span> ${escapeHtml(thSummary)}</summary><div class="thinking-content">${thRendered}</div></details>`;
  }
  let content = "";
  if (msg.text) {
    content = renderSimpleMarkdown(msg.text);
  } else if (msg.streaming) {
    content = '<span class="cursor-blink"></span>';
  }
  const toolHtml = msg.activeTool
    ? `<div class="tool-indicator"><span class="tool-spinner"></span>${escapeHtml(msg.activeTool)} を実行中...</div>`
    : "";
  return `<div class="chat-bubble assistant${streamClass}">${thinkingHtml}${content}${toolHtml}${tsHtml}</div>`;
}

function renderConvMessages() {
  if (!dom.convMessages) return;

  const { activeThreadId, conversationAnima } = getState();
  const animaName = conversationAnima;
  const threadMessages = getState().chatMessagesByThread?.[animaName]?.[activeThreadId || "default"] || [];
  const hs = animaName ? _historyState[animaName]?.[activeThreadId || "default"] : null;

  // No history and no live messages
  if ((!hs || hs.sessions.length === 0) && threadMessages.length === 0) {
    if (hs && hs.loading) {
      dom.convMessages.innerHTML = '<div class="chat-empty"><span class="tool-spinner"></span> \u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
    } else {
      dom.convMessages.innerHTML = '<div class="chat-empty">\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u307E\u3060\u3042\u308A\u307E\u305B\u3093</div>';
    }
    return;
  }

  let html = "";

  // Sentinel + loading indicator for infinite scroll
  if (hs && hs.hasMore) {
    if (hs.loading) {
      html += '<div class="history-loading-more"><span class="tool-spinner"></span> \u904E\u53BB\u306E\u4F1A\u8A71\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
    }
    html += '<div class="chat-load-sentinel"></div>';
  }

  // Render sessions from history API
  if (hs && hs.sessions.length > 0) {
    for (let si = 0; si < hs.sessions.length; si++) {
      const session = hs.sessions[si];
      html += _renderSessionDivider(session, si === 0);

      if (session.messages) {
        for (const msg of session.messages) {
          html += _renderHistoryMessage(msg);
        }
      }
    }
  }

  // Render live chat messages (current streaming session)
  if (threadMessages.length > 0) {
    if (hs && hs.sessions.length > 0) {
      html += '<div class="session-divider"><span class="session-divider-label">\u73FE\u5728\u306E\u30BB\u30C3\u30B7\u30E7\u30F3</span></div>';
    }
    html += threadMessages.map(renderConvBubble).join("");
  }

  dom.convMessages.innerHTML = html;

  // Bind tool call handlers for history messages
  _bindToolCallHandlers(dom.convMessages);

  // Re-observe sentinel after render
  _observeSentinel();

  dom.convMessages.scrollTop = dom.convMessages.scrollHeight;
}

async function loadAndRenderConvMessages(animaName) {
  if (!animaName) return;

  const threadId = getState().activeThreadId || "default";

  // Initialize history state (loading) per-thread
  if (!_historyState[animaName]) _historyState[animaName] = {};
  _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: true };

  const { chatMessagesByThread } = getState();
  const nextByThread = { ...chatMessagesByThread };
  if (!nextByThread[animaName]) nextByThread[animaName] = {};
  nextByThread[animaName][threadId] = [];
  setState({ chatMessagesByThread: nextByThread });

  renderConvMessages();  // Shows loading indicator

  try {
    const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, null, threadId);
    if (data && data.sessions) {
      _historyState[animaName][threadId] = {
        sessions: data.sessions,
        hasMore: data.has_more || false,
        nextBefore: data.next_before || null,
        loading: false,
      };
    } else {
      _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
    }
  } catch (err) {
    logger.error("Failed to load conversation", { anima: animaName, error: err.message });
    _historyState[animaName][threadId] = { sessions: [], hasMore: false, nextBefore: null, loading: false };
  }

  renderConvMessages();

  // Set up infinite scroll observer
  _setupScrollObserver();

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

    // Show accumulated text in streaming bubble (thread-aware)
    const { activeThreadId, chatMessagesByThread } = getState();
    const threadId = activeThreadId || "default";
    const streamingMsg = {
      role: "assistant",
      text: progress.full_text || "",
      streaming: true,
      activeTool: progress.active_tool || null,
    };
    const current = chatMessagesByThread?.[animaName]?.[threadId] || [];
    const nextByThread = { ...chatMessagesByThread };
    if (!nextByThread[animaName]) nextByThread[animaName] = {};
    nextByThread[animaName][threadId] = [...current, streamingMsg];
    setState({ chatMessagesByThread: nextByThread });
    renderConvMessages();

    convStreamController = new AbortController();
    _wsUpdateSendButton(true);

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
      onThinkingStart: () => {
        streamingMsg.thinkingText = "";
        streamingMsg.thinking = true;
        updateStreamingBubble(streamingMsg);
      },
      onThinkingDelta: (text) => {
        streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text;
        scheduleStreamingUpdate(streamingMsg);
      },
      onThinkingEnd: () => {
        streamingMsg.thinking = false;
        updateStreamingBubble(streamingMsg);
      },
      onDone: ({ summary, emotion }) => {
        if (summary) streamingMsg.text = summary;
        if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
        streamingMsg.streaming = false;
        streamingMsg.activeTool = null;
        setExpression(emotion);
        setTimeout(() => setExpression("neutral"), 3000);
        const st = getState();
        const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
        const nextByThread = { ...st.chatMessagesByThread };
        if (!nextByThread[animaName]) nextByThread[animaName] = {};
        nextByThread[animaName][threadId] = [...arr];
        setState({ chatMessagesByThread: nextByThread });
        renderConvMessages();
      },
      onError: ({ message: errorMsg }) => {
        streamingMsg.text += `\n[エラー: ${errorMsg}]`;
        streamingMsg.streaming = false;
        setExpression("troubled");
        const st = getState();
        const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
        const nextByThread = { ...st.chatMessagesByThread };
        if (!nextByThread[animaName]) nextByThread[animaName] = {};
        nextByThread[animaName][threadId] = [...arr];
        setState({ chatMessagesByThread: nextByThread });
        renderConvMessages();
      },
    });

    setTalking(false);
    if (streamingMsg.streaming) {
      streamingMsg.streaming = false;
      if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
      const st = getState();
      const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
      const nextByThread = { ...st.chatMessagesByThread };
      if (!nextByThread[animaName]) nextByThread[animaName] = {};
      nextByThread[animaName][threadId] = [...arr];
      setState({ chatMessagesByThread: nextByThread });
      renderConvMessages();
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      logger.error("Resume stream error", { anima: animaName, error: err.message });
    }
  } finally {
    convStreamController = null;
    _wsUpdateSendButton(false);
    dom.convInput?.focus();

    if (convPendingQueue.length > 0) {
      const next = convPendingQueue.shift();
      _wsShowPendingIndicator();
      if (convPendingQueue.length === 0) _wsHidePendingIndicator();
      setTimeout(() => {
        _sendConversation(next.text, { images: next.images, displayImages: next.displayImages });
      }, 150);
    }
  }
}

// ── Infinite Scroll (Upward) ──────────────────────

function _setupScrollObserver() {
  if (!dom.convMessages) return;

  if (_scrollObserver) _scrollObserver.disconnect();

  _scrollObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          _loadMoreHistory();
        }
      }
    },
    { root: dom.convMessages, rootMargin: "200px 0px 0px 0px" },
  );

  _observeSentinel();
}

function _observeSentinel() {
  if (!_scrollObserver || !dom.convMessages) return;
  const sentinel = dom.convMessages.querySelector(".chat-load-sentinel");
  if (sentinel) _scrollObserver.observe(sentinel);
}

async function _loadMoreHistory() {
  const animaName = getState().conversationAnima;
  const threadId = getState().activeThreadId || "default";
  if (!animaName) return;

  const hs = _historyState[animaName]?.[threadId];
  if (!hs || !hs.hasMore || hs.loading) return;

  hs.loading = true;
  // Show loading indicator
  const existingIndicator = dom.convMessages.querySelector(".history-loading-more");
  if (!existingIndicator) {
    const indicator = document.createElement("div");
    indicator.className = "history-loading-more";
    indicator.innerHTML = '<span class="tool-spinner"></span> \u904E\u53BB\u306E\u4F1A\u8A71\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...';
    dom.convMessages.insertBefore(indicator, dom.convMessages.firstChild);
  }

  try {
    const data = await fetchConversationHistory(animaName, HISTORY_PAGE_SIZE, hs.nextBefore, threadId);
    if (data && data.sessions && data.sessions.length > 0) {
      hs.sessions = [...data.sessions, ...hs.sessions];
      hs.hasMore = data.has_more || false;
      hs.nextBefore = data.next_before || null;
    } else {
      hs.hasMore = false;
    }
  } catch (err) {
    logger.error("Failed to load more history", { anima: animaName, error: err.message });
    hs.hasMore = false;
  }
  hs.loading = false;

  // Re-render preserving scroll position
  const prevScrollHeight = dom.convMessages.scrollHeight;
  renderConvMessages();
  const newScrollHeight = dom.convMessages.scrollHeight;
  dom.convMessages.scrollTop += (newScrollHeight - prevScrollHeight);
}

// ── SSE Streaming for Conversation ──────────────────────

function _wsSubmitConversation() {
  const text = dom.convInput?.value?.trim();
  const hasImages = convImageInputManager && convImageInputManager.getImageCount() > 0;
  const isStreaming = !!convStreamController;

  // ── Not streaming ──
  if (!isStreaming) {
    if (text || hasImages) {
      convPendingQueue.push({
        text: text || "",
        images: convImageInputManager?.getPendingImages() || [],
        displayImages: convImageInputManager?.getDisplayImages() || [],
      });
      convImageInputManager?.clearImages();
    }
    if (convPendingQueue.length === 0) return;
    const next = convPendingQueue.shift();
    _wsShowPendingIndicator();
    if (convPendingQueue.length === 0) _wsHidePendingIndicator();
    _sendConversation(next.text, { images: next.images, displayImages: next.displayImages });
    return;
  }

  // ── Streaming + has input → add to queue ──
  if (text || hasImages) {
    convPendingQueue.push({
      text: text || "",
      images: convImageInputManager?.getPendingImages() || [],
      displayImages: convImageInputManager?.getDisplayImages() || [],
    });
    dom.convInput.value = "";
    dom.convInput.style.height = "auto";
    convImageInputManager?.clearImages();
    _wsShowPendingIndicator();
    _wsUpdateSendButton(true);
    return;
  }

  // ── Streaming + empty input + has queue → interrupt & drain queue ──
  if (convPendingQueue.length > 0) {
    _wsInterruptAndSendPending();
    return;
  }

  // ── Streaming + empty input + no queue → just stop ──
  _wsStopStreaming();
}

async function sendConversationMessage() {
  _wsSubmitConversation();
}

async function _sendConversation(text, overrideImages = null) {
  const images = overrideImages?.images || convImageInputManager?.getPendingImages() || [];
  const displayImages = overrideImages?.displayImages || convImageInputManager?.getDisplayImages() || [];
  if (!text && images.length === 0) return;

  const animaName = getState().conversationAnima;
  const threadId = getState().activeThreadId || "default";
  if (!animaName) return;

  // Capture display images (with dataUrl for rendering)
  const displayImages = convImageInputManager?.getDisplayImages() || [];

  // Clear input
  dom.convInput.value = "";
  dom.convInput.disabled = true;
  dom.convSend.disabled = true;

  // Add user message + streaming assistant placeholder (thread-aware)
  const { chatMessagesByThread } = getState();
  const current = chatMessagesByThread?.[animaName]?.[threadId] || [];
  const sendTs = new Date().toISOString();
  const userMsg = { role: "user", text: text || "", images: displayImages, timestamp: sendTs };
  const streamingMsg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: sendTs, thinkingText: "", thinking: false };
  const nextByThread = { ...chatMessagesByThread };
  if (!nextByThread[animaName]) nextByThread[animaName] = {};
  nextByThread[animaName][threadId] = [...current, userMsg, streamingMsg];
  setState({ chatMessagesByThread: nextByThread });
  renderConvMessages();

  if (!overrideImages) {
    convImageInputManager?.clearImages();
  }

  convStreamController = new AbortController();
  _wsUpdateSendButton(true);

  try {
    let sendSucceeded = false;
    const userName = getCurrentUser() || "guest";
    const bodyObj = { message: text || "", from_person: userName, thread_id: threadId };
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
      onThinkingStart: () => {
        streamingMsg.thinkingText = "";
        streamingMsg.thinking = true;
        updateStreamingBubble(streamingMsg);
      },
      onThinkingDelta: (text) => {
        streamingMsg.thinkingText = (streamingMsg.thinkingText || "") + text;
        scheduleStreamingUpdate(streamingMsg);
      },
      onThinkingEnd: () => {
        streamingMsg.thinking = false;
        updateStreamingBubble(streamingMsg);
      },
      onDone: ({ summary, emotion }) => {
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

    // Finalize streaming message (thread-aware) + update thread label on first message
    streamingMsg.streaming = false;
    if (!streamingMsg.text) streamingMsg.text = "(空の応答)";
    const st = getState();
    const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
    const nextByThread = { ...st.chatMessagesByThread };
    if (!nextByThread[animaName]) nextByThread[animaName] = {};
    nextByThread[animaName] = { ...nextByThread[animaName], [threadId]: [...arr] };
    setState({ chatMessagesByThread: nextByThread });
    const threadList = st.threads[animaName] || [];
    const entry = threadList.find((t) => t.id === threadId);
    if (entry && entry.label === "新しいスレッド" && (text || "").trim()) {
      const firstLine = (text || "").trim().slice(0, 20) + ((text || "").trim().length > 20 ? "..." : "");
      const nextThreads = { ...st.threads, [animaName]: threadList.map((t) => t.id === threadId ? { ...t, label: firstLine } : t) };
      setState({ threads: nextThreads });
      _renderWsThreadTabs();
    }
    renderConvMessages();
    sendSucceeded = true;

    if (sendSucceeded && dom.convInput && dom.convInput.value.trim() === text.trim()) {
      dom.convInput.value = "";
      dom.convInput.style.height = "auto";
      _wsClearDraft(animaName);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      streamingMsg.streaming = false;
      streamingMsg.activeTool = null;
      if (!streamingMsg.text) streamingMsg.text = "(中断されました)";
      const abortSt = getState();
      const abortArr = abortSt.chatMessagesByThread?.[animaName]?.[threadId] || [];
      const abortNext = { ...abortSt.chatMessagesByThread };
      if (!abortNext[animaName]) abortNext[animaName] = {};
      abortNext[animaName] = { ...abortNext[animaName], [threadId]: [...abortArr] };
      setState({ chatMessagesByThread: abortNext });
      renderConvMessages();
    } else {
      logger.error("Conversation stream error", { anima: animaName, error: err.message, name: err.name });
      streamingMsg.text = `[エラー] ${err.message}`;
      streamingMsg.streaming = false;
      streamingMsg.activeTool = null;
      const st = getState();
      const arr = st.chatMessagesByThread?.[animaName]?.[threadId] || [];
      const nextByThread = { ...st.chatMessagesByThread };
      if (!nextByThread[animaName]) nextByThread[animaName] = {};
      nextByThread[animaName] = { ...nextByThread[animaName], [threadId]: [...arr] };
      setState({ chatMessagesByThread: nextByThread });
      renderConvMessages();
      setExpression("troubled");
    }
    setTalking(false);
  } finally {
    convStreamController = null;
    _wsUpdateSendButton(false);
    _wsSaveDraft();
    dom.convInput?.focus();

    if (convPendingQueue.length > 0) {
      const next = convPendingQueue.shift();
      _wsShowPendingIndicator();
      if (convPendingQueue.length === 0) _wsHidePendingIndicator();
      setTimeout(() => {
        _sendConversation(next.text, { images: next.images, displayImages: next.displayImages });
      }, 150);
    }
  }
}

// ── Workspace Queue Helpers ──────────────────────

function _wsAddToQueue() {
  const text = dom.convInput?.value?.trim();
  const hasImages = convImageInputManager && convImageInputManager.getImageCount() > 0;
  if (!text && !hasImages) return;
  convPendingQueue.push({
    text: text || "",
    images: convImageInputManager?.getPendingImages() || [],
    displayImages: convImageInputManager?.getDisplayImages() || [],
  });
  if (dom.convInput) { dom.convInput.value = ""; dom.convInput.style.height = "auto"; }
  convImageInputManager?.clearImages();
  _wsShowPendingIndicator();
  _wsUpdateSendButton(!!convStreamController);
}

function _wsShowPendingIndicator() {
  if (!dom.convPending || !dom.convPendingList) return;
  if (convPendingQueue.length === 0) { dom.convPending.style.display = "none"; return; }
  if (dom.convPendingLabel) dom.convPendingLabel.textContent = `キュー (${convPendingQueue.length})`;
  dom.convPendingList.innerHTML = convPendingQueue.map((p, i) => {
    const txt = escapeHtml(p.text.length > 50 ? p.text.slice(0, 50) + "…" : p.text);
    const img = p.images?.length ? ` <span style="opacity:0.6">(+${p.images.length}画像)</span>` : "";
    return `<div class="pending-queue-item" data-idx="${i}">` +
      `<span class="pending-queue-item-num">${i + 1}.</span>` +
      `<span class="pending-queue-item-text">${txt || "(画像のみ)"}${img}</span>` +
      `<button class="pending-queue-item-del" data-idx="${i}" type="button">✕</button>` +
      `</div>`;
  }).join("");
  dom.convPending.style.display = "";

  dom.convPendingList.onclick = (e) => {
    const delBtn = e.target.closest(".pending-queue-item-del");
    if (delBtn) {
      e.stopPropagation();
      const idx = parseInt(delBtn.dataset.idx, 10);
      convPendingQueue.splice(idx, 1);
      _wsShowPendingIndicator();
      _wsUpdateSendButton(!!convStreamController);
      return;
    }
    const item = e.target.closest(".pending-queue-item");
    if (item) {
      const idx = parseInt(item.dataset.idx, 10);
      const removed = convPendingQueue.splice(idx, 1)[0];
      if (removed && dom.convInput) {
        dom.convInput.value = removed.text;
        dom.convInput.style.height = "auto";
        const maxH = isMobileView() ? 100 : 120;
        dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
        dom.convInput.focus();
      }
      _wsShowPendingIndicator();
      _wsUpdateSendButton(!!convStreamController);
    }
  };
}

function _wsHidePendingIndicator() {
  if (dom.convPending) dom.convPending.style.display = "none";
}

function _wsUpdateSendButton(isStreaming) {
  const hasInput = (dom.convInput?.value?.trim() || "").length > 0;

  if (dom.convQueueBtn) {
    dom.convQueueBtn.disabled = !hasInput;
  }

  if (!dom.convSend) return;
  dom.convSend.classList.remove("stop", "interrupt");
  if (!isStreaming) {
    dom.convSend.textContent = (convPendingQueue.length > 0 || hasInput) ? "↑" : "↑";
    dom.convSend.disabled = !hasInput && convPendingQueue.length === 0;
  } else if (hasInput) {
    dom.convSend.textContent = "↑";
    dom.convSend.disabled = false;
  } else if (convPendingQueue.length > 0) {
    dom.convSend.textContent = "■↑";
    dom.convSend.classList.add("interrupt");
    dom.convSend.disabled = false;
  } else {
    dom.convSend.textContent = "■";
    dom.convSend.classList.add("stop");
    dom.convSend.disabled = false;
  }
}

function _wsStopStreaming() {
  const animaName = getState().conversationAnima;
  if (!animaName) return;
  if (convStreamController) {
    convStreamController.abort();
  }
  fetch(`/api/animas/${encodeURIComponent(animaName)}/interrupt`, {
    method: "POST",
  }).catch(() => {});
}

function _wsInterruptAndSendPending() {
  _wsStopStreaming();
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
  if (msg.thinkingText) {
    const open = msg.thinking ? " open" : "";
    const summary = msg.thinking ? "Thinking..." : `Thinking (${msg.thinkingText.length} chars)`;
    const thinkingRendered = renderSimpleMarkdown(msg.thinkingText);
    html += `<details class="thinking-block"${open}><summary class="thinking-summary"><span class="thinking-icon">💭</span> ${escapeHtml(summary)}</summary><div class="thinking-content">${thinkingRendered}</div></details>`;
  }
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

function _buildVoiceChatCallbacks(animaName) {
  return {
    addUserBubble(text) {
      const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
      const threadId = activeThreadId || "default";
      const current = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
      const ts = new Date().toISOString();
      const nextByThread = { ...chatMessagesByThread };
      if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
      nextByThread[conversationAnima][threadId] = [...current, { role: "user", text, timestamp: ts }];
      setState({ chatMessagesByThread: nextByThread });
      renderConvMessages();
    },
    addStreamingBubble() {
      const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
      const threadId = activeThreadId || "default";
      const current = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
      const ts = new Date().toISOString();
      const msg = { role: "assistant", text: "", streaming: true, activeTool: null, timestamp: ts, thinkingText: "", thinking: false };
      const nextByThread = { ...chatMessagesByThread };
      if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
      nextByThread[conversationAnima][threadId] = [...current, msg];
      setState({ chatMessagesByThread: nextByThread });
      renderConvMessages();
      return msg;
    },
    updateStreamingBubble(msg) {
      updateStreamingBubble(msg);
    },
    finalizeStreamingBubble(msg) {
      const { conversationAnima, activeThreadId, chatMessagesByThread } = getState();
      const threadId = activeThreadId || "default";
      const arr = chatMessagesByThread?.[conversationAnima]?.[threadId] || [];
      const nextByThread = { ...chatMessagesByThread };
      if (!nextByThread[conversationAnima]) nextByThread[conversationAnima] = {};
      nextByThread[conversationAnima][threadId] = [...arr];
      setState({ chatMessagesByThread: nextByThread });
      renderConvMessages();
    },
  };
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
    if (_currentView === "org") {
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
    if (_currentView === "org") {
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
    if (_currentView === "org") {
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
    if (_currentView === "org") {
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
    if (_currentView === "org") {
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
    const initialView = getCurrentView();
    await switchView(initialView);
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
  dom.convSend?.addEventListener("click", () => _wsSubmitConversation());
  dom.convQueueBtn?.addEventListener("click", () => _wsAddToQueue());
  dom.convInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.altKey) {
      e.preventDefault();
      _wsAddToQueue();
    } else if (e.key === "Enter") {
      if (isMobileView()) {
        if (!e.shiftKey) {
          e.preventDefault();
          _wsSubmitConversation();
        }
      } else {
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          _wsSubmitConversation();
        }
      }
    }
  });

  // Pending queue cancel all
  dom.convPendingCancel?.addEventListener("click", () => {
    convPendingQueue = [];
    _wsHidePendingIndicator();
    _wsUpdateSendButton(!!convStreamController);
  });

  // Auto-resize conversation input + dynamic button update
  dom.convInput?.addEventListener("input", () => {
    dom.convInput.style.height = "auto";
    const maxH = isMobileView() ? 100 : 120;
    dom.convInput.style.height = Math.min(dom.convInput.scrollHeight, maxH) + "px";
    _wsSaveDraft();
    _wsUpdateSendButton(!!convStreamController);
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

  // View switching: 3D office or org dashboard
  const initialView = getCurrentView();
  await switchView(initialView);

  // View toggle button
  if (dom.viewToggle) {
    dom.viewToggle.addEventListener("click", () => {
      const next = _currentView === "3d" ? "org" : "3d";
      switchView(next);
    });
  }

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

function applyTheme() {
  const theme = localStorage.getItem("aw-theme") || "default";
  document.body.classList.toggle("theme-business", theme === "business");
}

export async function init() {
  await initI18n();
  applyTranslations();

  applyTheme();
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
