// ── State Management ──────────────────────
// Simple Pub/Sub state store. No framework needed.

let state = {
  currentUser: localStorage.getItem("animaworks_user") || null,
  animas: [],
  selectedAnima: null,
  wsConnected: false,
  officeInitialized: false,      // Whether 3D office has been initialized
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
