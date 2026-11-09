import { setState } from "./state.js";
import { selectOfficeRenderer } from "./office-renderer-selection.js";
import { mountOffice3d } from "./renderers/office-3d-renderer.js";
import { mountOfficePixel } from "./renderers/office-pixel-renderer.js";

const RENDERER_STORAGE_KEY = "aw-workspace-renderer";

let _selectedRenderer = null;
let _activeRenderer = null;
let _activeInstance = null;
let _pendingMount = null;
let _mountGeneration = 0;
let _replyWaitingNames = new Set();

function selectedRenderer() {
  if (_selectedRenderer) return _selectedRenderer;

  let storedRenderer = null;
  try {
    storedRenderer = localStorage.getItem(RENDERER_STORAGE_KEY);
  } catch { /* storage may be disabled */ }

  const preference = selectOfficeRenderer(window.location.search, storedRenderer);
  _selectedRenderer = preference.renderer;
  if (preference.source === "url") {
    try {
      localStorage.setItem(RENDERER_STORAGE_KEY, preference.renderer);
    } catch { /* URL selection still works without storage */ }
  }
  return _selectedRenderer;
}

selectedRenderer();

function disposeActiveRenderer() {
  const instance = _activeInstance;
  _activeInstance = null;
  _activeRenderer = null;
  try {
    instance?.dispose();
  } finally {
    setState({ officeInitialized: false });
  }
}

/** Initialize the selected renderer in the workspace canvas. */
export async function mountOffice(dom) {
  const renderer = selectedRenderer();
  if (_activeRenderer === renderer && _activeInstance) return;

  while (_pendingMount) {
    const pending = _pendingMount;
    await pending.promise;
    if (_activeRenderer === renderer && _activeInstance) return;
  }

  disposeActiveRenderer();
  const generation = ++_mountGeneration;
  let instance = null;
  const promise = (async () => {
    try {
      instance = renderer === "3d"
        ? await mountOffice3d(dom)
        : mountOfficePixel(dom);
      if (!instance) {
        if (generation === _mountGeneration) setState({ officeInitialized: false });
        return;
      }
      if (generation !== _mountGeneration) {
        instance.dispose();
        return;
      }

      _activeInstance = instance;
      _activeRenderer = renderer;
      // Renderers opt in to the reply-waiting bubble; 3D simply ignores it.
      if (typeof instance.setReplyWaiting === "function") {
        instance.setReplyWaiting(_replyWaitingNames);
      }
      if (instance.ready) {
        const runtime = await instance.ready;
        if (generation !== _mountGeneration || _activeInstance !== instance) {
          instance.dispose();
          return;
        }
        if (runtime == null) disposeActiveRenderer();
      }
    } catch (error) {
      instance?.dispose();
      if (generation === _mountGeneration) {
        _activeInstance = null;
        _activeRenderer = null;
        setState({ officeInitialized: false });
      }
      console.error(`[workspace] Failed to mount ${renderer} office renderer`, error);
    }
  })();
  const pending = { promise };
  _pendingMount = pending;
  try {
    await promise;
  } finally {
    if (_pendingMount === pending) _pendingMount = null;
  }
}

export function disposeOfficeRenderer() {
  _mountGeneration += 1;
  disposeActiveRenderer();
}

export function highlightAnima(name) {
  _activeInstance?.highlight(name == null ? null : name);
}

/** Show a persistent "reply waiting" bubble on the given Animas' desks. */
export function setReplyWaiting(names) {
  _replyWaitingNames = new Set(names || []);
  if (typeof _activeInstance?.setReplyWaiting === "function") {
    _activeInstance.setReplyWaiting(_replyWaitingNames);
  }
}

export function getActiveRenderer() {
  return _activeRenderer;
}
