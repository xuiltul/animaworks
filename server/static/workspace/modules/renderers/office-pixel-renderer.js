import { t } from "/shared/i18n.js";
import { getState } from "../state.js";
import { selectAnima } from "../anima.js";
import { createPixelOffice } from "../../pixel/modules/office.js";

const SCALE_STORAGE_KEY = "aw-workspace-pixel-display-scale";
const SCALE_OPTIONS = ["auto", "1", "2", "3", "4"];

function savedScale() {
  try {
    const value = localStorage.getItem(SCALE_STORAGE_KEY);
    return SCALE_OPTIONS.includes(value) ? value : "auto";
  } catch {
    return "auto";
  }
}

function createHud(parent) {
  const root = document.createElement("div");
  root.className = "ws-pixel-hud";

  const scaleLabel = document.createElement("label");
  scaleLabel.className = "ws-pixel-scale-label";
  scaleLabel.textContent = t("ws.pixel.display_scale");

  const displayScale = document.createElement("select");
  displayScale.className = "ws-pixel-scale-select";
  displayScale.setAttribute("aria-label", t("ws.pixel.display_scale"));
  for (const value of SCALE_OPTIONS) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "auto" ? t("ws.pixel.scale_auto") : `${value}×`;
    displayScale.appendChild(option);
  }
  displayScale.value = savedScale();
  scaleLabel.appendChild(displayScale);

  const dayNightButton = document.createElement("button");
  dayNightButton.className = "ws-pixel-day-night";
  dayNightButton.type = "button";
  dayNightButton.setAttribute("aria-label", t("ws.pixel.toggle_day_night"));
  dayNightButton.title = t("ws.pixel.toggle_day_night");

  const dayNightIcon = document.createElement("img");
  dayNightIcon.className = "ws-pixel-day-night-icon";
  dayNightIcon.alt = "";
  const dayNightLabel = document.createElement("span");
  dayNightLabel.className = "ws-pixel-day-night-label";
  dayNightButton.append(dayNightIcon, dayNightLabel);

  const dateBoard = document.createElement("span");
  dateBoard.className = "ws-pixel-date-board";
  const dateText = document.createElement("span");
  dateText.className = "ws-pixel-date";
  const demoMark = document.createElement("small");
  demoMark.className = "ws-pixel-demo";
  dateBoard.append(dateText, demoMark);

  const connectionDot = document.createElement("span");
  connectionDot.className = "ws-pixel-connection";
  connectionDot.setAttribute("aria-hidden", "true");

  root.append(scaleLabel, dayNightButton, dateBoard, connectionDot);
  parent.appendChild(root);

  return {
    root,
    displayScale,
    dayNightButton,
    dayNightIcon,
    dayNightLabel,
    dayLabel: t("ws.pixel.day"),
    nightLabel: t("ws.pixel.night"),
    dateText,
    demoMark,
    demoLabel: t("ws.pixel.demo"),
    connectionDot,
    scaleStorageKey: SCALE_STORAGE_KEY,
  };
}

function resolveAnimaName(id) {
  return getState().animas.find((anima) =>
    String(anima.name || "").toLowerCase() === String(id || "").toLowerCase(),
  )?.name || null;
}

/** Mount the pixel office in the existing workspace shell. */
export function mountOfficePixel(dom) {
  const stage = document.createElement("div");
  stage.className = "ws-pixel-stage";
  const canvas = document.createElement("canvas");
  canvas.className = "ws-pixel-canvas";
  canvas.setAttribute("aria-label", t("ws.pixel.canvas_label"));
  stage.appendChild(canvas);
  dom.officeCanvas.appendChild(stage);

  const hud = createHud(dom.officePanel);
  const office = createPixelOffice({
    canvas,
    hud: { ...hud, fitContainer: stage },
    onAnimaClick: (id) => {
      // The human actor and other non-Anima sprites are not chat targets.
      const name = resolveAnimaName(id);
      if (name) selectAnima(name);
    },
  });

  let disposed = false;
  const replyWaiting = new Set();

  function applyReplyWaiting() {
    const actors = office.actors;
    if (!actors) return;
    for (const actor of actors.values()) {
      actor.replyWaiting = replyWaiting.has(String(actor.id).toLowerCase());
    }
  }

  function setReplyWaiting(names) {
    replyWaiting.clear();
    for (const name of names || []) replyWaiting.add(String(name).toLowerCase());
    applyReplyWaiting();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    office.dispose();
    hud.root.remove();
    stage.remove();
  }

  const ready = office.ready.then((runtime) => {
    if (disposed || !runtime) return null;
    applyReplyWaiting();
    const selectedAnima = getState().selectedAnima;
    if (selectedAnima) office.highlight(selectedAnima);
    return runtime;
  }).catch((error) => {
    dispose();
    throw error;
  });

  return {
    ready,
    highlight(name) {
      office.highlight(name);
    },
    setReplyWaiting,
    dispose,
  };
}
