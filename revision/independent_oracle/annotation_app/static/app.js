"use strict";

const state = {
  data: null,
  currentId: null,
  currentCase: null,
  selectedPath: null,
  filtered: [],
  dirty: false,
  editVersion: 0,
  saving: false,
  autosaveTimer: null,
  autoAdvancePending: false,
  view: "code",
};

const el = Object.fromEntries([
  "workspace", "lockedView", "lockedMessage", "authorProgress", "roleLabel",
  "progressText", "progressPercent", "progressBar", "guideButton", "guideDialog",
  "roleSwitcher",
  "closeGuide", "guideContent", "searchInput", "categoryFilter", "stateFilter",
  "caseList", "filterCount", "caseCategory", "caseId", "casePosition", "ruleName",
  "ruleDescription", "findingList", "authorDecision", "fileTabs", "codeView",
  "diffView", "defectiveCode", "referenceCode", "candidateCode", "referenceDiff",
  "candidateDiff", "annotationForm", "labelControl", "rationaleInput", "rationaleHint",
  "saveStatus", "previousButton", "saveButton", "saveNextButton", "nextUnlabeledButton",
  "fatalError",
].map((id) => [id, document.getElementById(id)]));

const initialParams = new URLSearchParams(window.location.search);
let activeRole = initialParams.get("role") || "author_1";
const roleLocked = initialParams.get("lock_role") === "1";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const url = new URL(path, window.location.origin);
  if (url.pathname.startsWith("/api/")) url.searchParams.set("role", activeRole);
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function roleTitle(role) {
  return ({ author_1: "Author 1 · blinded annotation", author_2: "Author 2 · blinded annotation", adjudicator: "Third author · disagreement adjudication" })[role] || role;
}

function setProgress() {
  if (state.data.mode === "locked" && state.data.author_progress) {
    const progress = state.data.author_progress;
    const completed = progress.author_1 + progress.author_2;
    const total = progress.total * 2;
    const percent = total ? Math.round((completed / total) * 100) : 0;
    el.progressText.textContent = `Authors ${completed} / ${total}`;
    el.progressPercent.textContent = `${percent}%`;
    el.progressBar.style.width = `${percent}%`;
    return;
  }
  const completed = state.data.completed || 0;
  const total = state.data.total || 0;
  const percent = total ? Math.round((completed / total) * 100) : 0;
  el.progressText.textContent = `${completed} / ${total} completed`;
  el.progressPercent.textContent = `${percent}%`;
  el.progressBar.style.width = `${percent}%`;
}

function renderRoleSwitcher() {
  const roles = state.data.available_roles || [];
  if (!roles.length || roleLocked) {
    el.roleSwitcher.hidden = true;
    return;
  }
  const labels = { author_1: "Author 1", author_2: "Author 2", adjudicator: "Adjudicator" };
  el.roleSwitcher.hidden = false;
  el.roleSwitcher.innerHTML = "";
  for (const role of roles) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = labels[role];
    button.className = role === state.data.role ? "active" : "";
    button.setAttribute("aria-current", role === state.data.role ? "page" : "false");
    button.addEventListener("click", async () => {
      if (role === state.data.role || !(await flushDirty())) return;
      const url = new URL(window.location.href);
      url.searchParams.set("role", role);
      window.location.assign(url);
    });
    el.roleSwitcher.appendChild(button);
  }
}

function setSaveStatus(message, type = "") {
  el.saveStatus.textContent = message;
  el.saveStatus.className = `save-status ${type}`.trim();
}

function renderGuide(markdown) {
  const lines = String(markdown || "").split("\n");
  const nodes = [];
  let inList = false;
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("# ")) {
      if (inList) { nodes.push("</ul>"); inList = false; }
      nodes.push(`<h1>${formatInline(line.slice(2))}</h1>`);
    } else if (line.startsWith("- ")) {
      if (!inList) { nodes.push("<ul>"); inList = true; }
      nodes.push(`<li>${formatInline(line.slice(2))}</li>`);
    } else if (!line) {
      if (inList) { nodes.push("</ul>"); inList = false; }
    } else {
      if (inList) { nodes.push("</ul>"); inList = false; }
      nodes.push(`<p>${formatInline(line)}</p>`);
    }
  }
  if (inList) nodes.push("</ul>");
  el.guideContent.innerHTML = nodes.join("");
}

function formatInline(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function populateFilters() {
  const selected = el.categoryFilter.value;
  const categories = [...new Set(state.data.cases.map((item) => item.category))].sort();
  el.categoryFilter.innerHTML = '<option value="all">All categories</option>' + categories.map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join("");
  el.categoryFilter.value = categories.includes(selected) ? selected : "all";
}

function applyFilters() {
  const query = el.searchInput.value.trim().toLowerCase();
  const category = el.categoryFilter.value;
  const annotationState = el.stateFilter.value;
  state.filtered = state.data.cases.filter((item) => {
    const queryMatch = !query || item.blind_id.toLowerCase().includes(query) || item.rule.toLowerCase().includes(query);
    const categoryMatch = category === "all" || item.category === category;
    const isLabeled = Boolean(item.label);
    const stateMatch = annotationState === "all" || (annotationState === "labeled" ? isLabeled : !isLabeled);
    return queryMatch && categoryMatch && stateMatch;
  });
  renderCaseList();
}

function renderCaseList() {
  el.caseList.innerHTML = "";
  for (const item of state.filtered) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-item${item.blind_id === state.currentId ? " active" : ""}`;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", item.blind_id === state.currentId ? "true" : "false");
    button.innerHTML = `
      <span class="case-item-id">${escapeHtml(item.blind_id)}</span>
      <span class="state-dot ${item.label ? "done" : ""}" title="${item.label || "Unlabeled"}"></span>
      <span class="case-item-category">${escapeHtml(item.category)}</span>
      <span class="case-item-rule">${escapeHtml(item.rule)}</span>`;
    button.addEventListener("click", () => selectCase(item.blind_id));
    el.caseList.appendChild(button);
  }
  el.filterCount.textContent = `${state.filtered.length} of ${state.data.cases.length} cases`;
}

function annotationValues() {
  const checked = el.annotationForm.querySelector('input[name="label"]:checked');
  return { label: checked?.value || "", rationale: el.rationaleInput.value.trim() };
}

function validateAnnotation(showError = true) {
  const { label, rationale } = annotationValues();
  const valid = Boolean(label) && (label === "Correct" || Boolean(rationale));
  const rationaleMissing = Boolean(label && label !== "Correct" && !rationale);
  el.rationaleInput.classList.toggle("invalid", showError && rationaleMissing);
  if (showError && !label) setSaveStatus("Select a decision before saving.", "error");
  else if (showError && rationaleMissing) setSaveStatus("Rationale is required for this decision.", "error");
  return valid;
}

function markDirty() {
  state.dirty = true;
  state.editVersion += 1;
  const { label } = annotationValues();
  el.rationaleHint.textContent = label && label !== "Correct" ? "required" : "optional for Correct";
  setSaveStatus("Unsaved changes");
  clearTimeout(state.autosaveTimer);
  if (validateAnnotation(false)) {
    const delay = annotationValues().label === "Correct" ? 350 : 850;
    state.autosaveTimer = setTimeout(
      () => saveCurrent({ silent: true, advance: state.autoAdvancePending }),
      delay,
    );
  }
}

async function saveCurrent({ advance = false, silent = false } = {}) {
  clearTimeout(state.autosaveTimer);
  if (state.saving) return false;
  if (!validateAnnotation(!silent)) return false;
  const { label, rationale } = annotationValues();
  const savingId = state.currentId;
  const savingVersion = state.editVersion;
  state.saving = true;
  setSaveStatus("Saving...");
  try {
    const updated = await api(`/api/labels/${encodeURIComponent(savingId)}`, {
      method: "POST",
      body: JSON.stringify({ label, rationale }),
    });
    state.data = updated;
    populateFilters();
    applyFilters();
    setProgress();
    const savedCurrentRevision = state.currentId === savingId && state.editVersion === savingVersion;
    if (savedCurrentRevision) {
      state.dirty = false;
      state.autoAdvancePending = false;
      if (state.currentCase) state.currentCase.annotation = { ...state.currentCase.annotation, label, rationale };
      setSaveStatus("Saved", "saved");
    } else {
      setSaveStatus("Newer changes not yet saved");
    }
    if (advance && savedCurrentRevision) await selectRelative(1, true);
    return true;
  } catch (error) {
    setSaveStatus(error.message, "error");
    return false;
  } finally {
    state.saving = false;
  }
}

async function flushDirty() {
  if (!state.dirty) return true;
  return saveCurrent();
}

async function selectCase(blindId) {
  if (blindId === state.currentId && state.currentCase) return;
  if (!(await flushDirty())) return;
  clearTimeout(state.autosaveTimer);
  setSaveStatus("Loading...");
  try {
    const caseData = await api(`/api/cases/${encodeURIComponent(blindId)}`);
    state.currentId = blindId;
    state.currentCase = caseData;
    state.selectedPath = allPaths(caseData)[0] || null;
    state.dirty = false;
    state.autoAdvancePending = false;
    renderCase();
    renderCaseList();
    if (state.data.locked) {
      for (const control of el.annotationForm.querySelectorAll('input, textarea, button[type="submit"], #saveNextButton')) control.disabled = true;
      setSaveStatus("Annotations frozen for adjudication", "saved");
    } else {
      setSaveStatus("Ready");
    }
    const active = el.caseList.querySelector(".case-item.active");
    active?.scrollIntoView({ block: "nearest" });
  } catch (error) {
    setSaveStatus(error.message, "error");
  }
}

function allPaths(caseData) {
  const paths = [];
  for (const key of ["defective_files", "human_reference_files", "candidate_repair_files"]) {
    for (const file of caseData[key] || []) if (!paths.includes(file.path)) paths.push(file.path);
  }
  return paths;
}

function getFile(group, path) {
  return (state.currentCase[group] || []).find((file) => file.path === path)?.content || "";
}

function renderCase() {
  const item = state.currentCase;
  const allCasesIndex = state.data.cases.findIndex((entry) => entry.blind_id === item.blind_id);
  el.caseCategory.textContent = item.category;
  el.caseId.textContent = item.blind_id;
  el.casePosition.textContent = `${allCasesIndex + 1} of ${state.data.cases.length}`;
  el.ruleName.textContent = item.rule;
  el.ruleDescription.textContent = item.rule_description;
  el.findingList.innerHTML = item.target_findings.map((finding) => `<div>${escapeHtml(finding.file)}:${escapeHtml(finding.line)}:${escapeHtml(finding.column)} · ${escapeHtml(finding.severity)}</div>`).join("");

  const annotation = item.annotation || {};
  for (const input of el.annotationForm.querySelectorAll('input[name="label"]')) input.checked = input.value === annotation.label;
  el.rationaleInput.value = annotation.rationale || "";
  el.rationaleHint.textContent = annotation.label && annotation.label !== "Correct" ? "required" : "optional for Correct";
  el.rationaleInput.classList.remove("invalid");

  if (state.data.role === "adjudicator") {
    el.authorDecision.hidden = false;
    el.authorDecision.innerHTML = `<span>Author 1 <strong>${escapeHtml(annotation.author_1_label)}</strong></span><span>Author 2 <strong>${escapeHtml(annotation.author_2_label)}</strong></span>`;
  } else {
    el.authorDecision.hidden = true;
  }
  renderFileTabs();
  renderComparison();
  updateNavigation();
}

function renderFileTabs() {
  const paths = allPaths(state.currentCase);
  el.fileTabs.innerHTML = "";
  for (const path of paths) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = path;
    button.title = path;
    button.className = path === state.selectedPath ? "active" : "";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", path === state.selectedPath ? "true" : "false");
    button.addEventListener("click", () => {
      state.selectedPath = path;
      renderFileTabs();
      renderComparison();
    });
    el.fileTabs.appendChild(button);
  }
}

function renderCode(content, targetLines = new Set(), changedLines = new Set(), changeKind = "") {
  const lines = String(content).replace(/\n$/, "").split("\n");
  return lines.map((line, index) => {
    const number = index + 1;
    const classes = ["code-line"];
    if (targetLines.has(number)) classes.push("target");
    if (changedLines.has(number)) classes.push(`changed-${changeKind}`);
    return `<span class="${classes.join(" ")}"><span class="line-number">${number}</span>${escapeHtml(line) || " "}</span>`;
  }).join("");
}

function buildDiffRows(before, after) {
  const a = String(before).replace(/\n$/, "").split("\n");
  const b = String(after).replace(/\n$/, "").split("\n");
  const dp = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  }
  const rows = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      rows.push({ kind: "same", oldNo: ++i, newNo: ++j, text: a[i - 1] });
    } else if (j < b.length && (i === a.length || dp[i][j + 1] >= dp[i + 1][j])) {
      rows.push({ kind: "add", oldNo: "", newNo: ++j, text: b[j - 1] });
    } else {
      rows.push({ kind: "remove", oldNo: ++i, newNo: "", text: a[i - 1] });
    }
  }
  return rows;
}

function diffLines(rows) {
  return rows.map((row) => {
    const marker = row.kind === "add" ? "+" : row.kind === "remove" ? "−" : " ";
    return `<span class="code-line diff-line ${row.kind}"><span class="line-number">${row.oldNo}</span><span class="line-number">${row.newNo}</span><span class="diff-marker">${marker}</span>${escapeHtml(row.text) || " "}</span>`;
  }).join("");
}

function renderComparison() {
  const path = state.selectedPath;
  const defective = getFile("defective_files", path);
  const reference = getFile("human_reference_files", path);
  const candidate = getFile("candidate_repair_files", path);
  const targetLines = new Set(state.currentCase.target_findings.filter((finding) => finding.file === path).map((finding) => Number(finding.line)));
  const referenceRows = buildDiffRows(defective, reference);
  const candidateRows = buildDiffRows(defective, candidate);
  const defectiveChanged = new Set(
    [...referenceRows, ...candidateRows]
      .filter((row) => row.kind === "remove")
      .map((row) => Number(row.oldNo)),
  );
  const referenceChanged = new Set(referenceRows.filter((row) => row.kind === "add").map((row) => Number(row.newNo)));
  const candidateChanged = new Set(candidateRows.filter((row) => row.kind === "add").map((row) => Number(row.newNo)));
  el.defectiveCode.innerHTML = renderCode(defective, targetLines, defectiveChanged, "defective");
  el.referenceCode.innerHTML = renderCode(reference, new Set(), referenceChanged, "reference");
  el.candidateCode.innerHTML = renderCode(candidate, new Set(), candidateChanged, "candidate");
  el.referenceDiff.innerHTML = diffLines(referenceRows);
  el.candidateDiff.innerHTML = diffLines(candidateRows);
  for (const pane of [el.defectiveCode, el.referenceCode, el.candidateCode, el.referenceDiff, el.candidateDiff]) pane.scrollTop = pane.scrollLeft = 0;
}

function updateNavigation() {
  const index = state.data.cases.findIndex((item) => item.blind_id === state.currentId);
  el.previousButton.disabled = index <= 0;
  el.saveNextButton.disabled = index < 0 || index >= state.data.cases.length - 1;
  el.nextUnlabeledButton.disabled = !state.data.cases.some((item) => !item.label && item.blind_id !== state.currentId);
}

async function selectRelative(offset, skipFlush = false) {
  if (!skipFlush && !(await flushDirty())) return;
  const index = state.data.cases.findIndex((item) => item.blind_id === state.currentId);
  const target = state.data.cases[index + offset];
  if (target) await selectCase(target.blind_id);
}

async function selectNextUnlabeled() {
  if (!(await flushDirty())) return;
  const cases = state.data.cases;
  const start = cases.findIndex((item) => item.blind_id === state.currentId);
  for (let step = 1; step <= cases.length; step += 1) {
    const item = cases[(start + step) % cases.length];
    if (!item.label) { await selectCase(item.blind_id); return; }
  }
}

function setView(view) {
  state.view = view;
  el.codeView.hidden = view !== "code";
  el.diffView.hidden = view !== "diff";
  for (const button of document.querySelectorAll("[data-view]")) button.classList.toggle("active", button.dataset.view === view);
}

function bindEvents() {
  el.searchInput.addEventListener("input", applyFilters);
  el.categoryFilter.addEventListener("change", applyFilters);
  el.stateFilter.addEventListener("change", applyFilters);
  el.annotationForm.addEventListener("submit", (event) => { event.preventDefault(); saveCurrent(); });
  el.labelControl.addEventListener("change", () => {
    state.autoAdvancePending = true;
    markDirty();
    const { label } = annotationValues();
    if (label !== "Correct") el.rationaleInput.focus();
  });
  el.rationaleInput.addEventListener("input", markDirty);
  el.previousButton.addEventListener("click", () => selectRelative(-1));
  el.saveNextButton.addEventListener("click", () => saveCurrent({ advance: true }));
  el.nextUnlabeledButton.addEventListener("click", selectNextUnlabeled);
  el.guideButton.addEventListener("click", () => el.guideDialog.showModal());
  el.closeGuide.addEventListener("click", () => el.guideDialog.close());
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => setView(button.dataset.view));
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveCurrent();
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (state.dirty) { event.preventDefault(); event.returnValue = ""; }
  });

  let syncing = false;
  for (const pane of [el.defectiveCode, el.referenceCode, el.candidateCode]) {
    pane.addEventListener("scroll", () => {
      if (syncing || state.view !== "code") return;
      syncing = true;
      for (const other of [el.defectiveCode, el.referenceCode, el.candidateCode]) {
        if (other !== pane) { other.scrollTop = pane.scrollTop; other.scrollLeft = pane.scrollLeft; }
      }
      requestAnimationFrame(() => { syncing = false; });
    });
  }
}

async function start() {
  bindEvents();
  try {
    state.data = await api("/api/bootstrap");
    activeRole = state.data.role;
    el.roleLabel.textContent = roleTitle(state.data.role);
    renderRoleSwitcher();
    setProgress();
    renderGuide(state.data.guide);
    if (state.data.mode === "locked") {
      el.lockedView.hidden = false;
      el.lockedMessage.textContent = state.data.message;
      const progress = state.data.author_progress;
      el.authorProgress.innerHTML = `<span>Author 1: ${progress.author_1} / ${progress.total}</span><span>Author 2: ${progress.author_2} / ${progress.total}</span>`;
      setInterval(async () => {
        try {
          const latest = await api("/api/bootstrap");
          if (latest.mode !== "locked") window.location.reload();
          state.data = latest;
          setProgress();
          const next = latest.author_progress;
          el.authorProgress.innerHTML = `<span>Author 1: ${next.author_1} / ${next.total}</span><span>Author 2: ${next.author_2} / ${next.total}</span>`;
        } catch (_) {
          // Keep the locked screen stable during transient local-server errors.
        }
      }, 10000);
      return;
    }
    populateFilters();
    applyFilters();
    el.workspace.hidden = false;
    if (state.data.cases.length) {
      const firstUnlabeled = state.data.cases.find((item) => !item.label) || state.data.cases[0];
      await selectCase(firstUnlabeled.blind_id);
    }
  } catch (error) {
    el.fatalError.hidden = false;
    el.fatalError.textContent = `Unable to open the annotation package: ${error.message}`;
  }
}

start();
