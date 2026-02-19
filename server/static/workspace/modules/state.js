// ── State Management ──────────────────────
// Simple Pub/Sub state store. No framework needed.

let state = {
  currentUser: localStorage.getItem("animaworks_user") || null,
  animas: [],
  selectedAnima: null,
  animaDetail: null,
  chatMessages: [],
  chatPagination: { totalRaw: 0, hasMore: false, loading: false },
  wsConnected: false,
  activeRightTab: "state",
  activeMemoryTab: "episodes",
  sessionList: null,
  officeInitialized: false,      // Whether 3D office has been initialized
  conversationOpen: false,       // Whether conversation panel is open in right sidebar
  conversationAnima: null,      // Anima name shown in conversation panel
  characterStates: {},           // Map: animaName → animationState (idle/working/thinking/error/sleeping)
};

const listeners = new Set();

export function getState() {
  return state;
}

export function setState(partial) {
  state = { ...state, ...partial };
  listeners.forEach((fn) => fn(state));
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
