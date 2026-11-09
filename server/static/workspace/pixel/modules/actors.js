import { t } from "/shared/i18n.js";
import { SpriteSheet } from "./sprite.js";
import { drawPixelText, measurePixelText } from "./pixel-text.js";

const STATE_MAP = Object.freeze({
  idle: { animation: "idle", bubble: "bubble_break" },
  working: { animation: "working", bubble: "bubble_working" },
  working_scheduled: { animation: "working", bubble: "bubble_cron" },
  thinking: { animation: "thinking", bubble: "bubble_thinking" },
  talking: { animation: "talking", bubble: "bubble_meeting" },
  reporting: { animation: "talking", bubble: "bubble_reporting" },
  sleeping: { animation: "sleeping", bubble: "bubble_sleeping" },
  success: { animation: "success", bubble: "" },
  error: { animation: "error", bubble: "bubble_error" },
  walking: { animation: "walk_down", bubble: "" },
});

const COMPACT_STATUS_COLORS = Object.freeze({
  idle: "#c98d9c",
  working: "#72b98e",
  success: "#e7bd5f",
  walking: "#6fa4bc",
  sleeping: "#8f8ab8",
});

// Seconds without any runtime event before the displayed state decays toward
// sleeping. Scheduled work keeps a longer grace because LLM turns can be
// silent between tool calls.
const STATE_DECAY_SECONDS = Object.freeze({
  working_scheduled: 300,
  working: 300,
  thinking: 600,
  idle: 180,
  success: 90,
});

// Short "何をしているか" labels derived from the tool an anima just used.
const TOOL_ACTIVITY_LABELS = [
  [/^bash$/, "コマンド実行中"],
  [/^(read|glob|grep)$/, "コード読解中"],
  [/^(edit|write|apply_patch|machine)$/, "コーディング中"],
  [/post_channel|broadcast/, "掲示板に投稿中"],
  [/send_message/, "メッセージ対応中"],
  [/call_human/, "報告準備中"],
  [/delegate_task/, "委任手配中"],
  [/search_memory|read_memory/, "調べ物中"],
  [/write_memory|archive_memory/, "記録整理中"],
  [/report_knowledge|report_procedure/, "知識整理中"],
  [/list_tasks|update_task|^goal$/, "タスク管理中"],
  [/skill/, "スキル整備中"],
  [/web_search|web_fetch|browser|fetch_url/, "調査中"],
];

function labelForTool(tool) {
  const name = String(tool || "").toLowerCase();
  if (!name) return "";
  for (const [pattern, label] of TOOL_ACTIVITY_LABELS) {
    if (pattern.test(name)) return label;
  }
  return "";
}

// Single source of truth for work-kind visuals (motion, bubble color, FX, speed).
// workKind layers on top of working / working_scheduled without replacing STATE_MAP.
const WORK_KINDS = Object.freeze({
  cron: {
    label: "定時作業中",
    legend: "定時",
    border: "#3f6fa8",
    text: "#2f5e93",
    accent: "clock",
    speed: 1.0,
  },
  task: {
    label: "タスク遂行中",
    legend: "タスク",
    border: "#3f7a38",
    text: "#356b2e",
    accent: "",
    speed: 1.5,
  },
  workers: {
    label: "フル稼働中",
    legend: "フル稼働",
    border: "#c05f2a",
    text: "#a84e1f",
    accent: "flash",
    speed: 2.0,
    shake: true,
    badge: true,
    microExpression: "error",
  },
  inbox: {
    label: "連絡対応中",
    legend: "連絡",
    border: "#7a5aa8",
    text: "#674a92",
    accent: "envelope",
    speed: 1.0,
    animation: "talking",
  },
  goal: {
    label: "目標に没頭中",
    legend: "没頭",
    border: "#b8862f",
    text: "#9c711f",
    accent: "sparkle",
    speed: 1.5,
    microExpression: "thinking",
  },
  consolidation: {
    label: "記憶整理中",
    legend: "記憶整理",
    border: "#4a4f8f",
    text: "#3a3f7a",
    accent: "moon",
    speed: 0.7,
    animation: "thinking",
  },
  chat: {
    label: "考え中",
    border: "#4a86a8",
    text: "#3a7292",
    accent: "",
    speed: 1.0,
  },
});

const DYNAMIC_BUBBLE_FILL = "#fbf0e4";
const DEFAULT_WORK_BORDER = "#3f7a38";
const DEFAULT_WORK_TEXT = "#356b2e";
// Persistent "reply waiting" badge; unlike activity labels it has no timer.
const REPLY_WAITING_COLORS = Object.freeze({ border: "#c27f00", text: "#9b720f" });

// Labels formerly baked into fx/bubbles.png rows — keep wording identical.
const STATE_BUBBLE_LABELS = Object.freeze({
  working: "作業中",
  working_scheduled: "定時作業中",
  thinking: "思考中",
  talking: "打合せ中",
  reporting: "報告中",
  sleeping: "居眠り中",
  error: "エラー",
  idle: "休憩中",
});

// Transient bubble overrides (setTransientBubble) keyed by former fx names.
const BUBBLE_FX_LABELS = Object.freeze({
  bubble_working: "作業中",
  bubble_thinking: "思考中",
  bubble_meeting: "打合せ中",
  bubble_sleeping: "居眠り中",
  bubble_reporting: "報告中",
  bubble_error: "エラー",
  bubble_break: "休憩中",
  bubble_instruction: "指示受領",
  bubble_delivery: "届け物中",
  bubble_cron: "定時作業中",
});

// Default border/text colors matching former bubbles.png rows.
const STATE_BUBBLE_COLORS = Object.freeze({
  working: { border: "#397b30", text: "#397b30" },
  working_scheduled: { border: "#397b30", text: "#397b30" },
  thinking: { border: "#3565a8", text: "#3565a8" },
  talking: { border: "#248a9b", text: "#248a9b" },
  reporting: { border: "#b85e18", text: "#b85e18" },
  error: { border: "#b93c46", text: "#b93c46" },
  idle: { border: "#bd4f72", text: "#bd4f72" },
  sleeping: { border: "#7d3ca1", text: "#7d3ca1" },
});

const BUBBLE_FX_COLORS = Object.freeze({
  bubble_instruction: { border: "#9b720f", text: "#9b720f" },
  bubble_delivery: { border: "#805028", text: "#805028" },
  bubble_meeting: { border: "#248a9b", text: "#248a9b" },
  bubble_working: { border: "#397b30", text: "#397b30" },
  bubble_thinking: { border: "#3565a8", text: "#3565a8" },
  bubble_sleeping: { border: "#7d3ca1", text: "#7d3ca1" },
  bubble_reporting: { border: "#b85e18", text: "#b85e18" },
  bubble_error: { border: "#b93c46", text: "#b93c46" },
  bubble_break: { border: "#bd4f72", text: "#bd4f72" },
  bubble_cron: { border: "#397b30", text: "#397b30" },
});

const NAME_PLATE_H = 14;

function workKindFromContext(ctx) {
  const context = String(ctx || "").toLowerCase();
  if (!context) return "";
  if (context.startsWith("cron") || context.startsWith("heartbeat")) return "cron";
  if (context.startsWith("task")) return "task";
  if (context.startsWith("workers")) return "workers";
  if (context.startsWith("inbox")) return "inbox";
  if (context.startsWith("goal")) return "goal";
  if (context.startsWith("consolidation")) return "consolidation";
  if (context === "chat" || context.startsWith("message")) return "chat";
  return "";
}

function labelForContext(ctx) {
  const kind = workKindFromContext(ctx);
  return kind && WORK_KINDS[kind] ? WORK_KINDS[kind].label : "";
}

function workKindConfig(kind) {
  return (kind && WORK_KINDS[kind]) || null;
}

function labelForBubbleName(name) {
  if (!name) return "";
  if (BUBBLE_FX_LABELS[name]) return BUBBLE_FX_LABELS[name];
  return "";
}

const DYNAMIC_BUBBLE_TEXT = Object.freeze({
  scale: 1,
  bold: true,
});

const NAME_TEXT_OPTIONS = Object.freeze({
  scale: 1,
  bold: true,
});

function normalizeState(value) {
  const raw = typeof value === "object" ? value?.state || value?.status : value;
  const state = String(raw || "idle").toLowerCase();
  if (STATE_MAP[state]) return state;
  if (state === "not_found" || state === "stopped") return "sleeping";
  // A running process with no observed activity is just waiting for work.
  if (state === "running" || state === "starting") return "sleeping";
  if (state.startsWith("cron") || state.startsWith("heartbeat") || state.startsWith("task")) {
    return "working_scheduled";
  }
  if (state.includes("bootstrap") || state.includes("think") || state.includes("process")) return "thinking";
  if (state.includes("work") || state.includes("busy")) return "working";
  if (state.includes("error") || state.includes("fail")) return "error";
  if (state.includes("sleep") || state.includes("stop") || state.includes("inactive")) return "sleeping";
  if (state.includes("talk") || state.includes("chat")) return "talking";
  if (state.includes("report")) return "reporting";
  if (state.includes("success") || state.includes("complete")) return "success";
  return "idle";
}

function tileKey(x, y) {
  return `${x},${y}`;
}

// Initial state for an actor at page load. The busy sidecar from /api/animas
// reports in-progress work, so active animas start awake instead of waiting
// for the next live event.
function initialActorState(anima) {
  const busy = anima?.busy;
  if (busy?.is_busy) {
    const progressAt = Date.parse(busy.last_progress_at || busy.busy_since || "");
    const fresh = Number.isFinite(progressAt) && Date.now() - progressAt < 15 * 60 * 1000;
    if (fresh) {
      const lanes = (busy.lanes || []).join(",");
      return lanes.includes("chat") ? "thinking" : "working_scheduled";
    }
  }
  return anima?.status || "idle";
}

function buildBlocked(scene) {
  const blocked = new Set();
  for (const desk of Object.values(scene.desks)) {
    const [x, y] = desk.tile;
    const half = desk.wide ? 2 : 1;
    for (let dx = -half; dx <= half; dx += 1) {
      blocked.add(tileKey(x + dx, y));
      blocked.add(tileKey(x + dx, y + 1));
    }
  }
  for (const [name, prop] of Object.entries(scene.props)) {
    if (name === "plants" || name === "welcome_mat" || name === "cat" || prop.under) continue;
    for (let dx = 0; dx < prop.w; dx += 1) {
      for (let dy = 0; dy < prop.h; dy += 1) {
        blocked.add(tileKey(prop.tile[0] + dx, prop.tile[1] + dy));
      }
    }
  }
  for (const [x, y] of scene.props.plants || []) blocked.add(tileKey(x, y));
  return blocked;
}

function waypointCandidates(scene) {
  const points = [];
  for (const prop of Object.values(scene.props || {})) {
    const sprite = prop.sprite || "";
    if (prop.kind === "meeting") {
      points.push(
        [prop.tile[0] - 1, prop.tile[1]],
        [prop.tile[0] + prop.w, prop.tile[1]],
        [prop.tile[0], prop.tile[1] + prop.h],
        [prop.tile[0] + prop.w - 1, prop.tile[1] + prop.h],
      );
    } else if (sprite === "sofa" || sprite === "coffee_corner") {
      const frontY = prop.tile[1] + prop.h + (sprite === "coffee_corner" ? 1 : 0);
      points.push([prop.tile[0] + Math.floor(prop.w / 2), frontY]);
    }
  }
  const width = scene.canvas.w / scene.canvas.tile;
  const height = scene.canvas.h / scene.canvas.tile;
  return points.filter(([x, y]) => x >= 1 && x < width - 1 && y >= 4 && y < height - 1);
}

function findPath(start, goal, blocked, scene) {
  const width = scene.canvas.w / scene.canvas.tile;
  const height = scene.canvas.h / scene.canvas.tile;
  const minimumY = 4;
  const startKey = tileKey(...start);
  const goalKey = tileKey(...goal);
  const open = [{ point: start, f: 0 }];
  const cameFrom = new Map();
  const gScore = new Map([[startKey, 0]]);
  const seen = new Set();
  const heuristic = ([x, y]) => Math.abs(x - goal[0]) + Math.abs(y - goal[1]);

  while (open.length) {
    open.sort((a, b) => a.f - b.f);
    const current = open.shift().point;
    const currentKey = tileKey(...current);
    if (currentKey === goalKey) {
      const path = [current];
      let cursor = currentKey;
      while (cameFrom.has(cursor)) {
        const previous = cameFrom.get(cursor);
        path.push(previous);
        cursor = tileKey(...previous);
      }
      return path.reverse();
    }
    if (seen.has(currentKey)) continue;
    seen.add(currentKey);
    const neighbors = [
      [current[0] + 1, current[1]], [current[0] - 1, current[1]],
      [current[0], current[1] + 1], [current[0], current[1] - 1],
    ];
    for (const next of neighbors) {
      const [x, y] = next;
      if (x < 1 || x >= width - 1 || y < minimumY || y >= height - 1) continue;
      const key = tileKey(x, y);
      if (key !== goalKey && blocked.has(key)) continue;
      const score = (gScore.get(currentKey) ?? Infinity) + 1;
      if (score >= (gScore.get(key) ?? Infinity)) continue;
      cameFrom.set(key, current);
      gScore.set(key, score);
      open.push({ point: next, f: score + heuristic(next) });
    }
  }
  return [start, goal];
}

class Actor {
  constructor(id, definition, homeTile, tileSize, seatPosition, assets, metadata = {}) {
    this.id = id;
    this.company = metadata.company || "default";
    this.isHuman = Boolean(metadata.isHuman);
    this.homeTile = [...homeTile];
    this.tileSize = tileSize;
    this.seatPosition = { ...seatPosition };
    this.x = seatPosition.x;
    this.y = seatPosition.y;
    this.sprite = new SpriteSheet(definition);
    this.assets = assets;
    this.state = "idle";
    this.bubble = "";
    this.bubbleOverride = "";
    this.bubbleOverrideRemaining = 0;
    this.replyWaiting = false;
    this.fxTime = 0;
    this.motion = null;
    this.isSeated = true;
    this.idleSeconds = 0;
    this.activityLabel = "";
    this.activityLabelRemaining = 0;
    this.contextLabel = "";
    this.contextLabelRemaining = 0;
    this.workKind = "";
    this.workKindRemaining = 0;
    this.laneCount = 0;
    this.burstRemaining = 0;
    this.burstAnimation = "";
    this.microExpressionRemaining = 0;
    this.microExpressionNextIn = 20 + Math.random() * 20;
  }

  setState(value) {
    const next = normalizeState(value);
    const changed = next !== this.state;
    this.state = next;
    this.idleSeconds = 0;
    // noteActivity often re-asserts working_scheduled; only cancel micro-expressions
    // when the state actually changes (walk / sleep / error / ...).
    if (changed) this.cancelMicroExpression();
    if (this.state === "sleeping") {
      this.setActivityLabel("");
      this.setContextLabel("");
      this.setWorkKind("");
    }
    const mapping = STATE_MAP[this.state];
    this.bubble = mapping.bubble;
    if (!this.motion) {
      // resolveSeatedAnimation keeps an in-flight micro-expression row.
      this.sprite.setAnimation(this.isSeated ? this.resolveSeatedAnimation() : "walk_down");
    }
  }

  noteActivity() {
    this.idleSeconds = 0;
  }

  setActivityLabel(label, duration = 30) {
    this.activityLabel = label || "";
    this.activityLabelRemaining = this.activityLabel ? duration : 0;
  }

  // Kind-of-work label (cron / task / inbox ...). Outlives individual tool
  // labels so silent stretches still show what the anima is broadly doing.
  setContextLabel(label, duration = 120) {
    this.contextLabel = label || "";
    this.contextLabelRemaining = this.contextLabel ? duration : 0;
  }

  setWorkKind(kind, duration = 120) {
    const next = WORK_KINDS[kind] ? kind : "";
    this.workKind = next;
    this.workKindRemaining = next ? duration : 0;
    if (!next) this.laneCount = 0;
    if (this.isSeated && !this.motion && this.burstRemaining <= 0) {
      this.sprite.setAnimation(this.resolveSeatedAnimation());
    }
  }

  // Seated pose animation: burst > micro-expression > workKind override > STATE_MAP.
  resolveSeatedAnimation() {
    if (this.burstRemaining > 0) return this.burstAnimation || "success";
    if (this.microExpressionRemaining > 0) {
      const micro = workKindConfig(this.workKind)?.microExpression;
      if (micro) return micro;
    }
    if (this.state === "working" || this.state === "working_scheduled") {
      const override = workKindConfig(this.workKind)?.animation;
      if (override) return override;
    }
    return STATE_MAP[this.state]?.animation || "idle";
  }

  // Temporary success (or other) row + sparkle without changing state.
  playBurst(animation = "success", duration = 3) {
    if (this.motion) return;
    if (this.state === "talking" || this.state === "reporting") return;
    this.cancelMicroExpression();
    this.burstAnimation = animation || "success";
    this.burstRemaining = duration;
    if (this.isSeated) this.sprite.setAnimation(this.burstAnimation);
  }

  cancelMicroExpression() {
    const wasActive = this.microExpressionRemaining > 0;
    this.microExpressionRemaining = 0;
    this.microExpressionNextIn = 20 + Math.random() * 20;
    if (wasActive && this.isSeated && !this.motion && this.burstRemaining <= 0) {
      this.sprite.setAnimation(this.resolveSeatedAnimation());
    }
  }

  dynamicLabel() {
    if (this.state !== "working_scheduled" && this.state !== "working") return "";
    let label = this.activityLabel || this.contextLabel;
    if (!label) return "";
    if (this.workKind === "workers" && this.laneCount >= 2) {
      label = `${label} ×${this.laneCount}`;
    }
    return label;
  }

  workBubbleColors() {
    if (this.replyWaiting) {
      return { ...REPLY_WAITING_COLORS, fill: DYNAMIC_BUBBLE_FILL };
    }
    const config = workKindConfig(this.workKind);
    if (config) {
      return {
        border: config.border,
        text: config.text,
        fill: DYNAMIC_BUBBLE_FILL,
      };
    }
    if (this.bubbleOverride && BUBBLE_FX_COLORS[this.bubbleOverride]) {
      return { ...BUBBLE_FX_COLORS[this.bubbleOverride], fill: DYNAMIC_BUBBLE_FILL };
    }
    const stateColors = STATE_BUBBLE_COLORS[this.state];
    if (stateColors) {
      return { ...stateColors, fill: DYNAMIC_BUBBLE_FILL };
    }
    return {
      border: DEFAULT_WORK_BORDER,
      text: DEFAULT_WORK_TEXT,
      fill: DYNAMIC_BUBBLE_FILL,
    };
  }

  // Activity label, transient override, or state default — all draw via the
  // same programmatic bubble path (no baked text in bubbles.png).
  fullBubbleLabel() {
    if (this.replyWaiting) return t("ws.reply_waiting");
    const activity = this.dynamicLabel();
    if (activity) return activity;
    if (this.bubbleOverride) return labelForBubbleName(this.bubbleOverride);
    return STATE_BUBBLE_LABELS[this.state] || labelForBubbleName(this.currentBubble());
  }

  animationSpeed() {
    if (this.state !== "working" && this.state !== "working_scheduled") return 1;
    const config = workKindConfig(this.workKind);
    if (config) return config.speed;
    return this.state === "working" ? 1.5 : 1;
  }

  shakeOffsetX() {
    const config = workKindConfig(this.workKind);
    if (!config?.shake || this.motion) return 0;
    if (this.state !== "working" && this.state !== "working_scheduled") return 0;
    // ~8Hz rectangularized sine, ±1px draw-only offset.
    return Math.sin(this.fxTime * Math.PI * 2 * 8) >= 0 ? 1 : -1;
  }

  renderScale() {
    return this.sprite.animation.startsWith("walk_") ? 1.5 : 1;
  }

  walk(path, speed = 120) {
    const includeHomeStep = this.isSeated;
    const route = includeHomeStep ? path : path.slice(1);
    const points = route.map(([x, y]) => ({
      x: (x + 0.5) * this.tileSize,
      y: (y + 1) * this.tileSize,
    }));
    return this.walkPixels(points, speed);
  }

  walkToSeat(speed = 120) {
    return this.walkPixels([{
      x: this.seatPosition.x,
      y: this.seatPosition.y,
    }], speed);
  }

  walkPixels(points, speed = 120) {
    if (this.motion?.resolve) this.motion.resolve(false);
    if (!points.length) return Promise.resolve(true);
    return new Promise((resolve) => {
      if (this.isSeated) this.y = this.visualY();
      this.isSeated = false;
      this.cancelMicroExpression();
      const first = points[0];
      const dx = first.x - this.x;
      const dy = first.y - this.y;
      this.sprite.setAnimation(
        Math.abs(dx) > Math.abs(dy) ? "walk_side" : (dy < 0 ? "walk_up" : "walk_down"),
        Math.abs(dx) > Math.abs(dy) && dx > 0,
      );
      this.motion = { points, speed, resolve };
      this.bubble = "";
    });
  }

  update(deltaSeconds) {
    this.fxTime += deltaSeconds;
    this.idleSeconds += deltaSeconds;
    const decayLimit = STATE_DECAY_SECONDS[this.state];
    if (decayLimit && !this.isHuman && this.isSeated && !this.motion &&
        this.idleSeconds > decayLimit) {
      this.setState("sleeping");
    }
    if (this.bubbleOverrideRemaining > 0) {
      this.bubbleOverrideRemaining = Math.max(0, this.bubbleOverrideRemaining - deltaSeconds);
      if (this.bubbleOverrideRemaining === 0) this.bubbleOverride = "";
    }
    if (this.activityLabelRemaining > 0) {
      this.activityLabelRemaining = Math.max(0, this.activityLabelRemaining - deltaSeconds);
      if (this.activityLabelRemaining === 0) this.activityLabel = "";
    }
    if (this.contextLabelRemaining > 0) {
      this.contextLabelRemaining = Math.max(0, this.contextLabelRemaining - deltaSeconds);
      if (this.contextLabelRemaining === 0) this.contextLabel = "";
    }
    if (this.workKindRemaining > 0) {
      this.workKindRemaining = Math.max(0, this.workKindRemaining - deltaSeconds);
      if (this.workKindRemaining === 0) {
        this.workKind = "";
        this.laneCount = 0;
        if (this.isSeated && !this.motion && this.burstRemaining <= 0) {
          this.sprite.setAnimation(this.resolveSeatedAnimation());
        }
      }
    }
    if (this.burstRemaining > 0) {
      this.burstRemaining = Math.max(0, this.burstRemaining - deltaSeconds);
      if (this.burstRemaining === 0) {
        this.burstAnimation = "";
        if (this.isSeated && !this.motion) {
          this.sprite.setAnimation(this.resolveSeatedAnimation());
        }
      }
    }
    this.updateMicroExpression(deltaSeconds);
    this.sprite.update(deltaSeconds * this.animationSpeed());
    if (!this.motion) return;
    let remaining = this.motion.speed * deltaSeconds;
    while (remaining > 0 && this.motion) {
      const target = this.motion.points[0];
      const dx = target.x - this.x;
      const dy = target.y - this.y;
      const distance = Math.hypot(dx, dy);
      if (distance <= remaining) {
        this.x = target.x;
        this.y = target.y;
        remaining -= distance;
        this.motion.points.shift();
        if (!this.motion.points.length) {
          const resolve = this.motion.resolve;
          this.motion = null;
          const mapping = STATE_MAP[this.state];
          this.sprite.setAnimation(this.isSeated ? this.resolveSeatedAnimation() : "walk_down");
          this.bubble = mapping.bubble;
          resolve(true);
        }
        continue;
      }
      const horizontal = Math.abs(dx) > Math.abs(dy);
      if (horizontal) this.sprite.setAnimation("walk_side", dx > 0);
      else this.sprite.setAnimation(dy < 0 ? "walk_up" : "walk_down");
      this.x += (dx / distance) * remaining;
      this.y += (dy / distance) * remaining;
      remaining = 0;
    }
  }

  updateMicroExpression(deltaSeconds) {
    const config = workKindConfig(this.workKind);
    const eligible = this.isSeated && !this.motion && this.burstRemaining <= 0 &&
      (this.state === "working" || this.state === "working_scheduled") &&
      Boolean(config?.microExpression);
    if (!eligible) {
      if (this.microExpressionRemaining > 0) this.cancelMicroExpression();
      return;
    }
    if (this.microExpressionRemaining > 0) {
      this.microExpressionRemaining = Math.max(0, this.microExpressionRemaining - deltaSeconds);
      if (this.microExpressionRemaining === 0) {
        this.microExpressionNextIn = 20 + Math.random() * 20;
        this.sprite.setAnimation(this.resolveSeatedAnimation());
      }
      return;
    }
    this.microExpressionNextIn -= deltaSeconds;
    if (this.microExpressionNextIn <= 0) {
      this.microExpressionRemaining = 1.5 + Math.random() * 0.5;
      this.sprite.setAnimation(config.microExpression);
    }
  }

  sit() {
    this.x = this.seatPosition.x;
    this.y = this.seatPosition.y;
    this.isSeated = true;
    const mapping = STATE_MAP[this.state];
    this.sprite.setAnimation(this.resolveSeatedAnimation());
    this.bubble = mapping.bubble;
  }

  setTransientBubble(name, duration = 1.5) {
    this.bubbleOverride = name;
    this.bubbleOverrideRemaining = duration;
  }

  clearTransientBubble() {
    this.bubbleOverride = "";
    this.bubbleOverrideRemaining = 0;
  }

  visualY() {
    return this.y;
  }

  spriteY() {
    return this.visualY();
  }

  draw(ctx) {
    if (!this.sprite.image) return false;
    const spriteY = this.spriteY();
    const scale = this.renderScale();
    const drawX = this.x + this.shakeOffsetX();
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = "#1b1218";
    ctx.beginPath();
    ctx.ellipse(
      Math.round(drawX),
      Math.round(spriteY - 2),
      Math.max(12, Math.round(this.sprite.frameW * scale * (this.isSeated ? 0.22 : 0.27))),
      Math.max(3, Math.round(this.sprite.frameH * scale * 0.065)),
      0,
      0,
      Math.PI * 2,
    );
    ctx.fill();
    ctx.restore();
    return this.sprite.draw(ctx, drawX, spriteY, scale);
  }

  drawStatusOverlay(ctx, options = {}) {
    const spriteY = this.spriteY();
    if (this.hasFullBubble()) {
      this.drawBubble(ctx, this.currentBubble(), spriteY, {
        offsetY: options.bubbleOffsetY || 0,
        offsetX: options.bubbleOffsetX || 0,
      });
      // workKind accents / burst sparkles still sit beside the dynamic bubble.
      if (this.burstRemaining > 0 ||
          this.state === "working" || this.state === "working_scheduled") {
        this.drawStateAccent(ctx, spriteY);
      }
      return;
    }
    this.drawStateAccent(ctx, spriteY);
  }

  drawNameOverlay(ctx, options = {}) {
    const position = {
      ...(options.namePosition || {}),
      offsetY: options.nameOffsetY || 0,
    };
    this.drawName(ctx, position);
    this.drawStatusDot(ctx, position);
  }

  drawOverlay(ctx, options = {}) {
    this.drawStatusOverlay(ctx, options);
    this.drawNameOverlay(ctx, options);
  }

  currentBubble() {
    if (this.bubbleOverride) return this.bubbleOverride;
    return STATE_MAP[this.state]?.bubble ?? "bubble_break";
  }

  isCompactStatus() {
    if (this.replyWaiting || this.dynamicLabel()) return false;
    return !this.bubbleOverride && Object.hasOwn(COMPACT_STATUS_COLORS, this.state);
  }

  hasFullBubble() {
    return !this.isCompactStatus() && Boolean(this.fullBubbleLabel());
  }

  nameBounds(position = {}) {
    const width = Math.max(28, measurePixelText(this.id, NAME_TEXT_OPTIONS) + 6);
    const x = Math.round(position.x ?? this.x);
    const y = Math.round((position.y ?? this.y + 3) + (position.offsetY || 0));
    return {
      x: x - width / 2 - 1,
      y,
      width: width + 12,
      height: NAME_PLATE_H,
    };
  }

  dynamicBubbleSize(label = this.fullBubbleLabel()) {
    const textWidth = measurePixelText(label || " ", DYNAMIC_BUBBLE_TEXT);
    // 10px PixelMplus raster is 12px tall; body + tail ≈ 24.
    return { width: Math.max(40, textWidth + 18), height: 24 };
  }

  bubbleBounds(name = this.currentBubble(), spriteY = this.spriteY(), offsetY = 0, offsetX = 0) {
    const label = this.replyWaiting
      ? t("ws.reply_waiting")
      : (this.dynamicLabel() || labelForBubbleName(name) ||
        STATE_BUBBLE_LABELS[this.state] || "");
    const size = this.dynamicBubbleSize(label);
    const x = Math.round(this.x - size.width / 2 + offsetX);
    const headTop = this.headTop(spriteY);
    const headGap = 5;
    const bob = Math.sin(this.fxTime * 3);
    const y = Math.round(headTop - headGap - (size.height - 3) + bob + offsetY);
    return { x, y, width: size.width, height: size.height };
  }

  drawName(ctx, position = {}) {
    ctx.save();
    const width = Math.max(28, measurePixelText(this.id, NAME_TEXT_OPTIONS) + 6);
    const x = Math.round(position.x ?? this.x);
    const y = Math.round((position.y ?? this.y + 3) + (position.offsetY || 0));
    ctx.fillStyle = "#573722";
    ctx.fillRect(x - width / 2 - 1, y, width + 2, NAME_PLATE_H);
    ctx.fillStyle = "#f3dfb5";
    ctx.fillRect(x - width / 2, y + 1, width, NAME_PLATE_H - 2);
    drawPixelText(ctx, this.id, x, y + Math.round(NAME_PLATE_H / 2), {
      ...NAME_TEXT_OPTIONS,
      align: "center",
      baseline: "middle",
      color: "#493020",
    });
    ctx.restore();
  }

  drawStatusDot(ctx, position = {}) {
    const width = Math.max(28, measurePixelText(this.id, NAME_TEXT_OPTIONS) + 6);
    const x = Math.round(position.x ?? this.x) + Math.ceil(width / 2) + 3;
    const y = Math.round((position.y ?? this.y + 3) + (position.offsetY || 0)) + 1;
    ctx.save();
    ctx.fillStyle = "#573722";
    ctx.fillRect(x, y, 8, 8);
    ctx.fillStyle = this.hasFullBubble()
      ? "#3a2f33"
      : (COMPACT_STATUS_COLORS[this.state] || "#8b7a70");
    ctx.fillRect(x + 1, y + 1, 6, 6);
    ctx.restore();
  }

  drawBubble(ctx, name, spriteY, options = {}) {
    const label = this.replyWaiting
      ? t("ws.reply_waiting")
      : (this.dynamicLabel() || labelForBubbleName(name) ||
        STATE_BUBBLE_LABELS[this.state] || "");
    if (!label) return;
    this.drawDynamicBubble(ctx, label, spriteY, options);
  }

  // Hand-drawn speech bubble for activity / state labels, styled to match the
  // former baked fx/bubbles.png rows. Colors prefer workKind, else state default.
  drawDynamicBubble(ctx, label, spriteY, options = {}) {
    const bounds = this.bubbleBounds(
      this.currentBubble(),
      spriteY,
      options.offsetY || 0,
      options.offsetX || 0,
    );
    const { x, y, width } = bounds;
    const bodyHeight = bounds.height - 6;
    const { border, text, fill } = this.workBubbleColors();
    ctx.save();
    if (options.quiet) ctx.globalAlpha = 0.55;
    ctx.fillStyle = border;
    ctx.fillRect(x + 2, y, width - 4, bodyHeight);
    ctx.fillRect(x, y + 2, width, bodyHeight - 4);
    ctx.fillStyle = fill;
    ctx.fillRect(x + 3, y + 2, width - 6, bodyHeight - 4);
    ctx.fillRect(x + 2, y + 3, width - 4, bodyHeight - 6);
    // tail
    const tailX = Math.round(x + width / 2);
    ctx.fillStyle = border;
    ctx.fillRect(tailX - 4, y + bodyHeight, 8, 2);
    ctx.fillRect(tailX - 2, y + bodyHeight + 2, 4, 2);
    ctx.fillStyle = fill;
    ctx.fillRect(tailX - 3, y + bodyHeight - 1, 6, 2);
    drawPixelText(ctx, label, tailX, y + Math.round(bodyHeight / 2), {
      ...DYNAMIC_BUBBLE_TEXT,
      align: "center",
      baseline: "middle",
      color: text,
    });
    ctx.restore();
  }

  headTop(spriteY) {
    const headInset = {
      thinking: 3,
      talking: 4,
      reporting: 4,
      sleeping: 19,
      error: 4,
    }[this.state] || 7;
    const scale = this.renderScale();
    return spriteY - (this.sprite.frameH - headInset) * scale;
  }

  drawStateAccent(ctx, spriteY) {
    const headTop = this.headTop(spriteY);
    if (this.burstRemaining > 0) {
      this.drawFx(ctx, "sparkle", this.x - 23, headTop + 15, this.fxTime);
      this.drawFx(ctx, "sparkle", this.x + 22, headTop + 3, this.fxTime + 0.22);
      return;
    }
    if (this.state === "error") {
      this.drawFx(ctx, "smoke", this.x - 21, headTop + 16, this.fxTime);
      this.drawFx(ctx, "bubble_small_exclamation", this.x + 25, headTop + 5, this.fxTime);
    } else if (this.state === "thinking") {
      this.drawFx(ctx, "bubble_small_question", this.x + 25, headTop + 5, this.fxTime);
    } else if (this.state === "sleeping") {
      this.drawFx(ctx, "bubble_small_sleep", this.x + 26, headTop + 6, this.fxTime);
    } else if (this.state === "success") {
      this.drawFx(ctx, "sparkle", this.x - 23, headTop + 15, this.fxTime);
      this.drawFx(ctx, "sparkle", this.x + 22, headTop + 3, this.fxTime + 0.22);
    } else if (this.state === "talking") {
      this.drawFx(ctx, "bubble_small_music", this.x + 25, headTop + 6, this.fxTime);
    } else if (this.state === "working" || this.state === "working_scheduled") {
      this.drawWorkKindAccent(ctx, headTop);
    }
  }

  drawWorkKindAccent(ctx, headTop) {
    const config = workKindConfig(this.workKind);
    if (!config) return;
    const accent = config.accent || "";
    if (accent === "clock") {
      this.drawFx(ctx, "clock", this.x + 25, headTop + 5, this.fxTime);
    } else if (accent === "flash") {
      this.drawFx(ctx, "flash", this.x - 23, headTop + 7, this.fxTime);
      if (config.badge && this.laneCount >= 2) {
        this.drawLaneBadge(ctx, this.x + 22, headTop + 3, this.laneCount);
      }
    } else if (accent === "sparkle") {
      this.drawFx(ctx, "sparkle", this.x - 23, headTop + 15, this.fxTime);
      this.drawFx(ctx, "sparkle", this.x + 22, headTop + 3, this.fxTime + 0.22);
    } else if (accent === "envelope") {
      const bob = Math.sin(this.fxTime * 4) * 1.5;
      this.drawFx(ctx, "envelope", this.x + 1, headTop - 1 + bob, this.fxTime);
    } else if (accent === "moon") {
      this.drawFx(ctx, "moon", this.x + 25, headTop + 5, this.fxTime);
    }
  }

  drawLaneBadge(ctx, x, y, count) {
    const text = `×${count}`;
    const options = { scale: 1, bold: true };
    const width = measurePixelText(text, options) + 6;
    const height = 14;
    const left = Math.round(x - width / 2);
    const top = Math.round(y - height / 2);
    ctx.save();
    ctx.fillStyle = "rgba(40, 24, 18, 0.72)";
    ctx.fillRect(left, top, width, height);
    ctx.fillStyle = "#fbf0e4";
    ctx.fillRect(left + 1, top + 1, width - 2, height - 2);
    drawPixelText(ctx, text, Math.round(x), Math.round(y), {
      ...options,
      align: "center",
      baseline: "middle",
      color: "#a84e1f",
    });
    ctx.restore();
  }

  drawFx(ctx, name, x, y, time) {
    const definition = this.assets.fxDefinition(name, name);
    const frame = Math.floor(time * definition.fps) % definition.frames;
    ctx.drawImage(
      definition.image,
      frame * definition.frameW,
      definition.row * definition.frameH,
      definition.frameW,
      definition.frameH,
      Math.round(x - definition.frameW / 2),
      Math.round(y - definition.frameH / 2),
      definition.frameW,
      definition.frameH,
    );
  }
}

export class ActorManager {
  constructor(scene, assets) {
    this.scene = scene;
    this.assets = assets;
    this.tile = scene.canvas.tile;
    this.disposed = false;
    this.blocked = buildBlocked(scene);
    this.actors = new Map();
    this.destinationReservations = new Map();
    this.actorReservations = new Map();
    this.waypoints = waypointCandidates(scene);
  }

  initialize(animas = []) {
    const known = new Map(animas.map((anima) => [
      String(anima.name || anima.id).toLowerCase(),
      anima,
    ]));
    const deskEntries = Object.entries(this.scene.desks || {});
    for (const [id, desk] of deskEntries) {
      const isHuman = desk.is_human || id === (this.scene.human_id || "human");
      const deskAsset = this.assets.prop("desk64");
      const deskHeight = deskAsset.naturalHeight || deskAsset.height || 48;
      const deskRows = Math.max(1, Math.ceil(deskHeight / this.tile));
      const home = [desk.tile[0], desk.tile[1] + deskRows];
      const deskFootY = (desk.tile[1] + deskRows) * this.tile;
      const seatPosition = {
        x: desk.tile[0] * this.tile + this.tile / 2 + (desk.seat_offset_x ?? 0),
        y: deskFootY - deskHeight + (desk.seat_sink ?? 20),
      };
      const actor = new Actor(
        id,
        this.assets.character(isHuman ? "human" : id),
        home,
        this.tile,
        seatPosition,
        this.assets,
        {
          company: desk.company || known.get(id)?.company || "default",
          isHuman,
        },
      );
      actor.setState(isHuman ? "idle" : initialActorState(known.get(id)));
      this.actors.set(id, actor);
    }
  }

  values() {
    return this.actors.values();
  }

  get(id) {
    return this.actors.get(String(id || "").toLowerCase());
  }

  ids(options = {}) {
    return [...this.actors.values()]
      .filter((actor) => options.includeHuman !== false || !actor.isHuman)
      .map((actor) => actor.id);
  }

  companyOf(id) {
    return this.get(id)?.company || null;
  }

  isHuman(id) {
    return Boolean(this.get(id)?.isHuman);
  }

  setState(id, state) {
    this.get(id)?.setState(state);
  }

  // Runtime events (tool activity, heartbeats, cron work) keep the displayed
  // state alive; without them the actor decays to sleeping.
  // extra.laneCount (from busy polling) feeds the workers ×N badge.
  noteActivity(id, ctx = "", tool = "", extra = {}) {
    const actor = this.get(id);
    if (!actor) return;
    actor.noteActivity();
    const context = String(ctx || "").toLowerCase();
    const chatty = ["thinking", "talking", "reporting", "error"].includes(actor.state);
    const resting = actor.state === "sleeping" || actor.state === "idle";
    if (context.startsWith("cron") || context.startsWith("task") ||
        context.startsWith("heartbeat") || context.startsWith("workers") ||
        context.startsWith("inbox") || context.startsWith("consolidation") ||
        context.startsWith("goal")) {
      if (!chatty) actor.setState("working_scheduled");
    } else if (context === "chat" || context.startsWith("message")) {
      if (resting) actor.setState("thinking");
    } else if (resting) {
      // Unknown context but the runtime is clearly doing something.
      actor.setState("working_scheduled");
    }
    const kind = workKindFromContext(context);
    if (kind) actor.setWorkKind(kind);
    const contextLabel = labelForContext(context);
    if (contextLabel) actor.setContextLabel(contextLabel);
    const label = labelForTool(tool);
    if (label) actor.setActivityLabel(label);
    if (Number.isFinite(extra.laneCount)) {
      actor.laneCount = Math.max(0, Math.floor(extra.laneCount));
    }
  }

  update(deltaSeconds) {
    this.actors.forEach((actor) => actor.update(deltaSeconds));
  }

  movingCount() {
    return [...this.actors.values()].filter((actor) => !actor.isSeated).length;
  }

  reservedTile(id) {
    const key = this.actorReservations.get(String(id || "").toLowerCase());
    return key ? key.split(",").map(Number) : null;
  }

  releaseDestination(id) {
    const actorId = String(id || "").toLowerCase();
    const key = this.actorReservations.get(actorId);
    if (key && this.destinationReservations.get(key) === actorId) {
      this.destinationReservations.delete(key);
    }
    this.actorReservations.delete(actorId);
  }

  destinationIsFree(point, actorId) {
    for (const [key, owner] of this.destinationReservations) {
      if (owner === actorId) continue;
      const [x, y] = key.split(",").map(Number);
      if (Math.max(Math.abs(point[0] - x), Math.abs(point[1] - y)) < 2) return false;
    }
    for (const actor of this.actors.values()) {
      if (actor.id === actorId || actor.isSeated) continue;
      const x = Math.round(actor.x / this.tile - 0.5);
      const y = Math.round(actor.y / this.tile - 1);
      if (Math.max(Math.abs(point[0] - x), Math.abs(point[1] - y)) < 2) return false;
    }
    return true;
  }

  reserveDestination(id, targetTile, alternatives = []) {
    const actorId = String(id || "").toLowerCase();
    this.releaseDestination(actorId);
    const seen = new Set();
    const candidates = [targetTile, ...alternatives, ...this.waypoints].filter((point) => {
      if (!Array.isArray(point) || point.length < 2) return false;
      const key = tileKey(...point);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    const selected = candidates.find((point) => this.destinationIsFree(point, actorId));
    if (!selected) return null;
    const key = tileKey(...selected);
    this.destinationReservations.set(key, actorId);
    this.actorReservations.set(actorId, key);
    return [...selected];
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    for (const actor of this.actors.values()) {
      const resolve = actor.motion?.resolve;
      actor.motion = null;
      resolve?.(false);
    }
    this.destinationReservations.clear();
    this.actorReservations.clear();
  }

  async walkTo(id, targetTile, options = {}) {
    if (this.disposed) return false;
    const actor = this.get(id);
    if (!actor) return false;
    const finalTarget = options.reserveDestination === false
      ? targetTile
      : this.reserveDestination(id, targetTile, options.alternatives || []);
    if (!finalTarget) return false;
    const start = actor.isSeated ? [...actor.homeTile] : [
      Math.round(actor.x / this.tile - 0.5),
      Math.round(actor.y / this.tile - 1),
    ];
    const targetCompany = options.targetCompany || null;
    const mustCross = targetCompany && targetCompany !== actor.company &&
      actor.company !== "human" && targetCompany !== "human";
    const pathRect = this.scene.zones.path?.rect;
    const pathEntry = pathRect
      ? [Math.floor((pathRect[0] + pathRect[2]) / 2), pathRect[1] + 1]
      : null;
    const destinations = [
      ...(mustCross
        ? [this.scene.walk.cross_company_via, ...(pathEntry ? [pathEntry] : [])]
        : []),
      ...(options.via || []),
      finalTarget,
    ];
    let cursor = start;
    for (const destination of destinations) {
      const path = findPath(cursor, destination, this.blocked, this.scene);
      const completed = await actor.walk(path, options.speed || 150);
      if (!completed) {
        this.releaseDestination(id);
        return false;
      }
      cursor = destination;
    }
    return true;
  }

  async returnHome(id, options = {}) {
    if (this.disposed) return false;
    const actor = this.get(id);
    if (!actor) return false;
    this.releaseDestination(id);
    const completed = await this.walkTo(id, actor.homeTile, {
      speed: 155,
      ...options,
      reserveDestination: false,
    });
    if (!completed) return false;
    const seated = await actor.walkToSeat(options.speed || 155);
    if (seated) actor.sit();
    return seated;
  }
}

export { findPath, WORK_KINDS };
