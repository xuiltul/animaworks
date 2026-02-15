/* ── Step 1: Language Selection ────────────── */

import { t, setLocale, getLocale } from "../setup.js";

let container = null;
let selectedLang = "ja";
let dropdownOpen = false;
let filterText = "";

const LANGUAGES = [
  { code: "en", native: "English" },
  { code: "ja", native: "日本語" },
  { code: "zh-CN", native: "简体中文" },
  { code: "zh-TW", native: "繁體中文" },
  { code: "ko", native: "한국어" },
  { code: "es", native: "Español" },
  { code: "fr", native: "Français" },
  { code: "de", native: "Deutsch" },
  { code: "pt", native: "Português" },
  { code: "it", native: "Italiano" },
  { code: "ru", native: "Русский" },
  { code: "ar", native: "العربية" },
  { code: "hi", native: "हिन्दी" },
  { code: "tr", native: "Türkçe" },
  { code: "vi", native: "Tiếng Việt" },
  { code: "th", native: "ไทย" },
  { code: "id", native: "Bahasa Indonesia" },
];

const PREVIEWS = {
  en: "AnimaWorks is a framework that treats AI agents as \"autonomous persons\". Each Person has their own identity, memories, and decision-making criteria.",
  ja: "AnimaWorksは、AIエージェントを「自律的な人」として扱うフレームワークです。各Personは固有のアイデンティティ・記憶・判断基準を持ちます。",
  "zh-CN": "AnimaWorks是一个将AI代理视为\"自主个体\"的框架。每个Person拥有独特的身份、记忆和决策标准。",
  "zh-TW": "AnimaWorks是一個將AI代理視為「自主個體」的框架。每個Person擁有獨特的身份、記憶和決策標準。",
  ko: "AnimaWorks는 AI 에이전트를 '자율적인 사람'으로 다루는 프레임워크입니다. 각 Person은 고유한 정체성, 기억, 판단 기준을 갖습니다.",
  es: "AnimaWorks es un framework que trata a los agentes de IA como \"personas autónomas\". Cada Person tiene su propia identidad, memorias y criterios de decisión.",
  fr: "AnimaWorks est un framework qui traite les agents IA comme des « personnes autonomes ». Chaque Person possède sa propre identité, ses souvenirs et ses critères de décision.",
  de: "AnimaWorks ist ein Framework, das KI-Agenten als \"autonome Personen\" behandelt. Jede Person hat ihre eigene Identität, Erinnerungen und Entscheidungskriterien.",
  pt: "AnimaWorks é um framework que trata agentes de IA como \"pessoas autônomas\". Cada Person possui identidade, memórias e critérios de decisão próprios.",
  it: "AnimaWorks è un framework che tratta gli agenti AI come \"persone autonome\". Ogni Person ha la propria identità, i propri ricordi e i propri criteri decisionali.",
  ru: "AnimaWorks — это фреймворк, который относится к ИИ-агентам как к «автономным личностям». Каждый Person обладает собственной идентичностью, памятью и критериями принятия решений.",
  ar: "AnimaWorks هو إطار عمل يتعامل مع وكلاء الذكاء الاصطناعي باعتبارهم \"أشخاصًا مستقلين\". يمتلك كل Person هوية وذكريات ومعايير اتخاذ قرارات فريدة.",
  hi: "AnimaWorks एक ऐसा फ्रेमवर्क है जो AI एजेंटों को \"स्वायत्त व्यक्ति\" के रूप में मानता है। प्रत्येक Person की अपनी पहचान, स्मृतियाँ और निर्णय मानदंड होते हैं।",
  tr: "AnimaWorks, yapay zeka ajanlarını \"özerk kişiler\" olarak ele alan bir çerçevedir. Her Person kendine özgü kimliğe, anılara ve karar verme ölçütlerine sahiptir.",
  vi: "AnimaWorks là một framework coi các tác nhân AI là \"con người tự chủ\". Mỗi Person có danh tính, ký ức và tiêu chí ra quyết định riêng.",
  th: "AnimaWorks เป็นเฟรมเวิร์กที่ปฏิบัติต่อเอเจนต์ AI ในฐานะ \"บุคคลอิสระ\" แต่ละ Person มีอัตลักษณ์ ความทรงจำ และเกณฑ์การตัดสินใจเป็นของตนเอง",
  id: "AnimaWorks adalah framework yang memperlakukan agen AI sebagai \"pribadi otonom\". Setiap Person memiliki identitas, memori, dan kriteria pengambilan keputusan yang unik.",
};

export function initLanguageStep(el) {
  container = el;
  detectLocale();
  render();

  // Close dropdown when clicking outside
  document.addEventListener("click", (e) => {
    if (dropdownOpen && container && !container.querySelector(".lang-dropdown-wrapper")?.contains(e.target)) {
      dropdownOpen = false;
      filterText = "";
      render();
    }
  });
}

async function detectLocale() {
  try {
    const res = await fetch("/api/setup/detect-locale");
    if (res.ok) {
      const data = await res.json();
      if (data.detected && LANGUAGES.some((l) => l.code === data.detected)) {
        selectedLang = data.detected;
        await setLocale(selectedLang);
        render();
      }
    }
  } catch {
    // Use default
  }
}

function getFilteredLanguages() {
  if (!filterText) return LANGUAGES;
  const q = filterText.toLowerCase();
  return LANGUAGES.filter(
    (l) => l.native.toLowerCase().includes(q) || l.code.toLowerCase().includes(q)
  );
}

function render() {
  const currentLang = LANGUAGES.find((l) => l.code === selectedLang) || LANGUAGES[0];
  const filtered = getFilteredLanguages();
  const preview = PREVIEWS[selectedLang] || PREVIEWS.ja;

  const optionItems = filtered
    .map((lang) => {
      const selected = lang.code === selectedLang ? " lang-option-selected" : "";
      return `<div class="lang-option${selected}" data-lang="${lang.code}" lang="${lang.code}">
        <span class="lang-option-native">${lang.native}</span>
        <span class="lang-option-code">${lang.code}</span>
      </div>`;
    })
    .join("");

  const emptyMsg = filtered.length === 0 ? `<div class="lang-dropdown-empty">No results</div>` : "";

  container.innerHTML = `
    <h2 data-i18n="lang.title">${t("lang.title")}</h2>
    <p style="color: #8888aa; font-size: 0.85rem; margin-top: 4px;" data-i18n="lang.desc">${t("lang.desc")}</p>
    <div class="lang-dropdown-wrapper" style="margin-top: 16px;">
      <button class="lang-dropdown-trigger${dropdownOpen ? " open" : ""}" type="button" aria-haspopup="listbox" aria-expanded="${dropdownOpen}">
        <span class="lang-globe">🌐</span>
        <span class="lang-current" lang="${currentLang.code}">${currentLang.native}</span>
        <span class="lang-chevron">${dropdownOpen ? "▲" : "▼"}</span>
      </button>
      ${dropdownOpen ? `
      <div class="lang-dropdown-panel" role="listbox" aria-label="Language selection">
        <div class="lang-search-wrapper">
          <input class="lang-search-input" type="text" placeholder="Search..." value="${filterText}" autofocus />
        </div>
        <div class="lang-options-list">
          ${optionItems}
          ${emptyMsg}
        </div>
      </div>
      ` : ""}
    </div>
    <div class="language-preview">
      <div class="language-preview-title" data-i18n="lang.preview.title">${t("lang.preview.title")}</div>
      <div class="language-preview-text">${preview}</div>
    </div>
  `;

  // Bind trigger click
  const trigger = container.querySelector(".lang-dropdown-trigger");
  if (trigger) {
    trigger.addEventListener("click", (e) => {
      e.stopPropagation();
      dropdownOpen = !dropdownOpen;
      filterText = "";
      render();
    });
  }

  // Bind search input
  const searchInput = container.querySelector(".lang-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      filterText = e.target.value;
      renderOptions();
    });
    searchInput.addEventListener("click", (e) => e.stopPropagation());
    // Focus search input when dropdown opens
    requestAnimationFrame(() => searchInput.focus());
  }

  // Bind option clicks
  bindOptionClicks();
}

function renderOptions() {
  const listEl = container.querySelector(".lang-options-list");
  if (!listEl) return;

  const filtered = getFilteredLanguages();
  const optionItems = filtered
    .map((lang) => {
      const selected = lang.code === selectedLang ? " lang-option-selected" : "";
      return `<div class="lang-option${selected}" data-lang="${lang.code}" lang="${lang.code}">
        <span class="lang-option-native">${lang.native}</span>
        <span class="lang-option-code">${lang.code}</span>
      </div>`;
    })
    .join("");

  const emptyMsg = filtered.length === 0 ? `<div class="lang-dropdown-empty">No results</div>` : "";
  listEl.innerHTML = optionItems + emptyMsg;
  bindOptionClicks();
}

function bindOptionClicks() {
  container.querySelectorAll(".lang-option").forEach((opt) => {
    opt.addEventListener("click", async (e) => {
      e.stopPropagation();
      selectedLang = opt.dataset.lang;
      dropdownOpen = false;
      filterText = "";
      await setLocale(selectedLang);
      render();
    });
  });
}

export function getLanguageData() {
  return { locale: selectedLang };
}
