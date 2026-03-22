/* ── Team Builder — Data Access Layer ─────────
   Handles templates, roles, team state.
   Future: replace localStorage calls with API calls.
   ─────────────────────────────────────────── */

import { t } from "/shared/i18n.js";

// ── Role Master ──────────────────────────────

export const ROLES = [
  {
    id: "secretary",
    nameKey: "tb.role.secretary",
    descKey: "tb.role.secretary.desc",
    defaultTools: ["gmail", "calendar", "slack", "notion"],
  },
  {
    id: "customer_support",
    nameKey: "tb.role.customer_support",
    descKey: "tb.role.customer_support.desc",
    defaultTools: ["gmail", "slack"],
  },
  {
    id: "back_office",
    nameKey: "tb.role.back_office",
    descKey: "tb.role.back_office.desc",
    defaultTools: ["notion", "calendar"],
  },
  {
    id: "sales_assist",
    nameKey: "tb.role.sales_assist",
    descKey: "tb.role.sales_assist.desc",
    defaultTools: ["gmail", "slack", "notion"],
  },
  {
    id: "pr_sns",
    nameKey: "tb.role.pr_sns",
    descKey: "tb.role.pr_sns.desc",
    defaultTools: ["slack", "notion"],
  },
  {
    id: "recruiter",
    nameKey: "tb.role.recruiter",
    descKey: "tb.role.recruiter.desc",
    defaultTools: ["gmail", "calendar", "notion"],
  },
  {
    id: "accounting",
    nameKey: "tb.role.accounting",
    descKey: "tb.role.accounting.desc",
    defaultTools: ["notion"],
  },
  {
    id: "project_manager",
    nameKey: "tb.role.project_manager",
    descKey: "tb.role.project_manager.desc",
    defaultTools: ["slack", "notion", "calendar"],
  },
  {
    id: "researcher",
    nameKey: "tb.role.researcher",
    descKey: "tb.role.researcher.desc",
    defaultTools: ["notion"],
  },
  {
    id: "content_writer",
    nameKey: "tb.role.content_writer",
    descKey: "tb.role.content_writer.desc",
    defaultTools: ["notion"],
  },
  {
    id: "engineer",
    nameKey: "tb.role.engineer",
    descKey: "tb.role.engineer.desc",
    defaultTools: ["notion"],
  },
  {
    id: "data_analyst",
    nameKey: "tb.role.data_analyst",
    descKey: "tb.role.data_analyst.desc",
    defaultTools: ["notion"],
  },
  {
    id: "designer",
    nameKey: "tb.role.designer",
    descKey: "tb.role.designer.desc",
    defaultTools: ["notion", "slack"],
  },
  {
    id: "marketing",
    nameKey: "tb.role.marketing",
    descKey: "tb.role.marketing.desc",
    defaultTools: ["slack", "notion"],
  },
  {
    id: "hr",
    nameKey: "tb.role.hr",
    descKey: "tb.role.hr.desc",
    defaultTools: ["gmail", "calendar", "notion"],
  },
  {
    id: "legal",
    nameKey: "tb.role.legal",
    descKey: "tb.role.legal.desc",
    defaultTools: ["notion"],
  },
  {
    id: "product_manager",
    nameKey: "tb.role.product_manager",
    descKey: "tb.role.product_manager.desc",
    defaultTools: ["slack", "notion", "calendar"],
  },
  {
    id: "finance",
    nameKey: "tb.role.finance",
    descKey: "tb.role.finance.desc",
    defaultTools: ["notion"],
  },
];

// ── Template Master ──────────────────────────

export const TEMPLATES = [
  {
    id: "secretary",
    nameKey: "tb.tpl.secretary",
    descKey: "tb.tpl.secretary.desc",
    recommended: true,
    members: [{ roleId: "secretary", count: 1 }],
  },
  {
    id: "customer_support",
    nameKey: "tb.tpl.customer_support",
    descKey: "tb.tpl.customer_support.desc",
    recommended: false,
    members: [{ roleId: "customer_support", count: 1 }],
  },
  {
    id: "sales_assist",
    nameKey: "tb.tpl.sales_assist",
    descKey: "tb.tpl.sales_assist.desc",
    recommended: false,
    members: [{ roleId: "sales_assist", count: 1 }],
  },
  {
    id: "back_office",
    nameKey: "tb.tpl.back_office",
    descKey: "tb.tpl.back_office.desc",
    recommended: false,
    members: [{ roleId: "back_office", count: 1 }],
  },
];

// ── Tool Display ─────────────────────────────

const TOOL_LABELS = {
  gmail: "Gmail",
  calendar: "Google Calendar",
  slack: "Slack",
  notion: "Notion",
};

export function getToolLabel(toolId) {
  return TOOL_LABELS[toolId] || toolId;
}

export function getAllTools() {
  return Object.entries(TOOL_LABELS).map(([id, label]) => ({ id, label }));
}

// ── Role Helpers ─────────────────────────────

export function getRoleById(id) {
  return ROLES.find((r) => r.id === id) || null;
}

export function getRoleName(roleId) {
  const role = getRoleById(roleId);
  return role ? t(role.nameKey) : roleId;
}

export function getTemplateById(id) {
  return TEMPLATES.find((tpl) => tpl.id === id) || null;
}

// ── Name Generator ───────────────────────────

const MEMBER_NAMES = [
  "Alice", "Ben", "Chloe", "David", "Emma",
  "Frank", "Grace", "Henry", "Ivy", "Jack",
  "Karen", "Leo", "Mia", "Noah", "Olivia",
  "Paul", "Quinn", "Ruby", "Sam", "Tina",
];

let _nameIndex = 0;

function generateName() {
  const name = MEMBER_NAMES[_nameIndex % MEMBER_NAMES.length];
  _nameIndex++;
  return name;
}

// ── Team State (localStorage backed) ─────────

const STORAGE_KEY = "aw_team_builder";

function _loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function _saveState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

/**
 * Create a team from selected roles.
 * @param {Array<{roleId: string, count: number}>} selections
 * @returns {{id: string, members: Array, department: string, reportTo: string}}
 */
export function createTeam(selections) {
  _nameIndex = 0;
  const members = [];
  for (const sel of selections) {
    const role = getRoleById(sel.roleId);
    if (!role) continue;
    for (let i = 0; i < sel.count; i++) {
      members.push({
        id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
        roleId: sel.roleId,
        displayName: generateName(),
        title: "",
        isLead: false,
        tools: [...role.defaultTools],
        model: "",
        credential: "",
      });
    }
  }

  // First member defaults to lead
  if (members.length > 0) {
    members[0].isLead = true;
  }

  const team = {
    id: crypto.randomUUID ? crypto.randomUUID() : `team-${Date.now()}`,
    department: "",
    reportTo: "",
    members,
    createdAt: new Date().toISOString(),
  };

  _saveState(team);
  return team;
}

/**
 * Get saved team (or null).
 */
export function getTeam() {
  return _loadState();
}

/**
 * Update team and persist.
 */
export function saveTeam(team) {
  _saveState(team);
}

/**
 * Update team-level settings (department, reportTo).
 */
export function updateTeamSettings(department, reportTo) {
  const team = getTeam();
  if (!team) return false;
  if (department !== undefined) team.department = department;
  if (reportTo !== undefined) team.reportTo = reportTo;
  _saveState(team);
  return true;
}

/**
 * Add a member to existing team.
 */
export function addMember(roleId) {
  const team = getTeam();
  if (!team) return null;
  const role = getRoleById(roleId);
  if (!role) return null;

  // Pick a name not yet used
  const usedNames = new Set(team.members.map((m) => m.displayName));
  let name = "";
  for (const n of MEMBER_NAMES) {
    if (!usedNames.has(n)) { name = n; break; }
  }
  if (!name) name = `Member-${team.members.length + 1}`;

  const member = {
    id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
    roleId,
    displayName: name,
    title: "",
    isLead: false,
    tools: [...role.defaultTools],
    model: "",
    credential: "",
  };
  team.members.push(member);
  _saveState(team);
  return member;
}

/**
 * Remove a member by id.
 */
export function removeMember(memberId) {
  const team = getTeam();
  if (!team) return false;
  team.members = team.members.filter((m) => m.id !== memberId);
  _saveState(team);
  return true;
}

/**
 * Update a member's role.
 */
export function updateMemberRole(memberId, newRoleId) {
  const team = getTeam();
  if (!team) return false;
  const member = team.members.find((m) => m.id === memberId);
  if (!member) return false;
  const role = getRoleById(newRoleId);
  if (!role) return false;
  member.roleId = newRoleId;
  member.tools = [...role.defaultTools];
  _saveState(team);
  return true;
}

/**
 * Update a member's display name.
 */
export function updateMemberName(memberId, newName) {
  const team = getTeam();
  if (!team) return false;
  const member = team.members.find((m) => m.id === memberId);
  if (!member) return false;
  member.displayName = newName.trim() || member.displayName;
  _saveState(team);
  return true;
}

/**
 * Update a member's model.
 */
export function updateMemberModel(memberId, modelId, credential) {
  const team = getTeam();
  if (!team) return false;
  const member = team.members.find((m) => m.id === memberId);
  if (!member) return false;
  member.model = modelId || "";
  member.credential = credential || "";
  _saveState(team);
  return true;
}

/**
 * Update a member's title.
 */
export function updateMemberTitle(memberId, newTitle) {
  const team = getTeam();
  if (!team) return false;
  const member = team.members.find((m) => m.id === memberId);
  if (!member) return false;
  member.title = (newTitle || "").trim();
  _saveState(team);
  return true;
}

/**
 * Set a member as team lead (only one lead at a time).
 */
export function setTeamLead(memberId) {
  const team = getTeam();
  if (!team) return false;
  for (const m of team.members) {
    m.isLead = (m.id === memberId);
  }
  _saveState(team);
  return true;
}

/**
 * Toggle a tool for a member.
 */
export function toggleMemberTool(memberId, toolId) {
  const team = getTeam();
  if (!team) return false;
  const member = team.members.find((m) => m.id === memberId);
  if (!member) return false;
  const idx = member.tools.indexOf(toolId);
  if (idx >= 0) {
    member.tools.splice(idx, 1);
  } else {
    member.tools.push(toolId);
  }
  _saveState(team);
  return true;
}
