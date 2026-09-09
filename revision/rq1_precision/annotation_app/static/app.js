"use strict";

const state = {
  data: null,
  currentId: null,
  currentCase: null,
  evidenceIndex: 0,
  filtered: [],
  dirty: false,
  autoAdvancePending: false,
  saveQueue: Promise.resolve(),
  pendingSaveCount: 0,
  saveVersions: new Map(),
  syncStatus: new Map(),
  failedSaves: new Map(),
  caseCache: new Map(),
  selectionToken: 0,
};

const elementIds = [
  "workspace", "lockedView", "lockedMessage", "authorProgress", "roleLabel",
  "roleSwitcher", "progressText", "progressPercent", "progressBar", "guideButton", "guideDialog",
  "closeGuide", "guideContent", "searchInput", "categoryFilter", "stateFilter",
  "caseList", "filterCount", "caseCategory", "caseId", "casePosition", "ruleName",
  "findingLocation", "findingMessage", "authorDecision", "sourcePath", "sourceCode",
  "jumpButton", "evidenceTabs", "evidenceContent", "annotationForm", "rationaleInput",
  "rationaleHint", "saveStatus", "previousButton", "saveButton", "saveNextButton",
  "nextUnlabeledButton", "fatalError",
];
const el = Object.fromEntries(elementIds.map((id) => [id, document.getElementById(id)]));
const initialParams = new URLSearchParams(window.location.search);
let activeRole = initialParams.get("role") || "author_1";

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
  if (!response.ok) {
    const error = new Error(payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function roleTitle(role) {
  return ({
    author_1: "Author 1 - independent blinded annotation",
    author_2: "Author 2 - independent blinded annotation",
    adjudicator: "Third author - disagreements only",
  })[role] || role;
}

function renderRoleSwitcher() {
  const roles = state.data.available_roles || [];
  if (!roles.length) {
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
      if (role === state.data.role || !(await flushDirty({ waitForBackground: true }))) return;
      const url = new URL(window.location.href);
      url.searchParams.set("role", role);
      window.location.assign(url);
    });
    el.roleSwitcher.appendChild(button);
  }
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
  const completed = state.data.cases.filter((item) => Boolean(item.label)).length;
  state.data.completed = completed;
  const total = state.data.total || 0;
  const percent = total ? Math.round((completed / total) * 100) : 0;
  el.progressText.textContent = `${completed} / ${total} completed`;
  el.progressPercent.textContent = `${percent}%`;
  el.progressBar.style.width = `${percent}%`;
}

function setSaveStatus(message, type = "") {
  el.saveStatus.textContent = message;
  el.saveStatus.className = `save-status ${type}`.trim();
}

function populateFilters() {
  const selected = el.categoryFilter.value;
  const categories = [...new Set(state.data.cases.map((item) => item.category))].sort();
  el.categoryFilter.innerHTML = '<option value="all">All categories</option>' + categories
    .map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`)
    .join("");
  el.categoryFilter.value = categories.includes(selected) ? selected : "all";
}

function applyFilters() {
  const query = el.searchInput.value.trim().toLowerCase();
  const category = el.categoryFilter.value;
  const annotationState = el.stateFilter.value;
  state.filtered = state.data.cases.filter((item) => {
    const haystack = `${item.blind_id} ${item.rule} ${item.location}`.toLowerCase();
    const queryMatch = !query || haystack.includes(query);
    const categoryMatch = category === "all" || item.category === category;
    const labeled = Boolean(item.label);
    const stateMatch = annotationState === "all" || (annotationState === "labeled" ? labeled : !labeled);
    return queryMatch && categoryMatch && stateMatch;
  });
  renderCaseList();
}

function renderCaseList() {
  el.caseList.innerHTML = "";
  for (const item of state.filtered) {
    const syncStatus = state.syncStatus.get(item.blind_id);
    const dotClass = syncStatus || (item.label ? "done" : "");
    const dotTitle = syncStatus === "pending"
      ? "Saving in background"
      : (syncStatus === "failed" ? "Save failed" : (item.label || "Unlabeled"));
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-item${item.blind_id === state.currentId ? " active" : ""}`;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", item.blind_id === state.currentId ? "true" : "false");
    button.innerHTML = `
      <span class="case-item-id">${escapeHtml(item.blind_id)}</span>
      <span class="state-dot ${dotClass}" title="${escapeHtml(dotTitle)}"></span>
      <span class="case-item-category">${escapeHtml(item.category)}</span>
      <span class="case-item-rule">${escapeHtml(item.rule)}</span>
      <span class="case-item-location">${escapeHtml(item.location)}</span>`;
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
  const rationaleMissing = Boolean(label && label !== "Correct" && !rationale);
  const valid = Boolean(label) && !rationaleMissing;
  el.rationaleInput.classList.toggle("invalid", showError && rationaleMissing);
  if (showError && !label) setSaveStatus("Select a decision before saving.", "error");
  else if (showError && rationaleMissing) setSaveStatus("Rationale is required for this decision.", "error");
  return valid;
}

function markDirty() {
  state.dirty = true;
  const { label } = annotationValues();
  el.rationaleHint.textContent = label && label !== "Correct" ? "required" : "optional for Correct";
  setSaveStatus("Unsaved changes");
  if (state.autoAdvancePending && label !== "Correct") {
    setSaveStatus("Enter the required rationale to continue.");
  }
}

function refreshSaveStatus() {
  if (state.pendingSaveCount > 0) {
    setSaveStatus(`${state.pendingSaveCount} saving in background...`);
  } else if (state.failedSaves.size > 0) {
    setSaveStatus(`${state.failedSaves.size} background save failed; revisit and save again.`, "error");
  } else {
    setSaveStatus("Saved", "saved");
  }
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function persistSave(record) {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      return await api(`/api/labels/${encodeURIComponent(record.blindId)}?compact=1`, {
        method: "POST",
        body: JSON.stringify({ label: record.label, rationale: record.rationale }),
      });
    } catch (error) {
      lastError = error;
      if ((error.status && error.status < 500) || attempt === 3) throw error;
      await wait(250 * attempt);
    }
  }
  throw lastError;
}

function enqueueSave(record) {
  const version = (state.saveVersions.get(record.blindId) || 0) + 1;
  state.saveVersions.set(record.blindId, version);
  state.syncStatus.set(record.blindId, "pending");
  state.failedSaves.delete(record.blindId);
  state.pendingSaveCount += 1;
  refreshSaveStatus();
  renderCaseList();

  const task = state.saveQueue.then(() => persistSave(record));
  state.saveQueue = task.catch(() => undefined);
  task.then((result) => {
    if (state.saveVersions.get(record.blindId) === version) {
      state.syncStatus.delete(record.blindId);
      state.failedSaves.delete(record.blindId);
    }
    if (result.locked) {
      state.data.locked = true;
      setFormLocked(true);
    }
  }).catch((error) => {
    if (state.saveVersions.get(record.blindId) === version) {
      state.syncStatus.set(record.blindId, "failed");
      state.failedSaves.set(record.blindId, { ...record, error: error.message });
    }
  }).finally(() => {
    state.pendingSaveCount -= 1;
    refreshSaveStatus();
    renderCaseList();
  });
  return task;
}

function nextUnlabeledId(afterId) {
  const cases = state.data.cases;
  const start = Math.max(0, cases.findIndex((item) => item.blind_id === afterId));
  for (let offset = 1; offset <= cases.length; offset += 1) {
    const item = cases[(start + offset) % cases.length];
    if (!item.label) return item.blind_id;
  }
  return null;
}

function updateOptimisticAnnotation(blindId, label, rationale) {
  const summary = state.data.cases.find((item) => item.blind_id === blindId);
  if (summary) {
    summary.label = label;
    summary.rationale = rationale;
  }
  if (state.currentCase?.blind_id === blindId) {
    state.currentCase.annotation = { ...state.currentCase.annotation, label, rationale };
  }
  state.dirty = false;
  state.autoAdvancePending = false;
  setProgress();
  applyFilters();
}

async function saveCurrent({ advance = false, waitForBackground = false } = {}) {
  if (!validateAnnotation()) return false;
  const { label, rationale } = annotationValues();
  const savingId = state.currentId;
  updateOptimisticAnnotation(savingId, label, rationale);
  const task = enqueueSave({ blindId: savingId, label, rationale });
  if (advance && !state.data.locked) {
    const nextId = nextUnlabeledId(savingId);
    if (nextId) void selectCase(nextId);
  }
  if (!waitForBackground) return true;
  try {
    await task;
    return true;
  } catch (_error) {
    return false;
  }
}

async function flushDirty({ waitForBackground = false } = {}) {
  if (state.dirty && !(await saveCurrent({ waitForBackground }))) return false;
  if (!waitForBackground) return true;
  await state.saveQueue;
  refreshSaveStatus();
  return state.failedSaves.size === 0;
}

function loadCase(blindId) {
  if (!state.caseCache.has(blindId)) {
    const request = api(`/api/cases/${encodeURIComponent(blindId)}`).catch((error) => {
      state.caseCache.delete(blindId);
      throw error;
    });
    state.caseCache.set(blindId, request);
  }
  return state.caseCache.get(blindId);
}

async function selectCase(blindId) {
  if (blindId === state.currentId && state.currentCase) return;
  if (!(await flushDirty())) return;
  const token = ++state.selectionToken;
  if (!state.pendingSaveCount) setSaveStatus("Loading...");
  try {
    const selectedCase = await loadCase(blindId);
    if (token !== state.selectionToken) return;
    state.currentCase = selectedCase;
    state.currentId = blindId;
    state.evidenceIndex = 0;
    state.dirty = false;
    state.autoAdvancePending = false;
    renderCase();
    renderCaseList();
    setFormLocked(Boolean(state.data.locked));
    if (state.pendingSaveCount || state.failedSaves.size) refreshSaveStatus();
    else setSaveStatus(state.data.locked ? "Author files frozen" : "Ready", state.data.locked ? "saved" : "");
    el.caseList.querySelector(".case-item.active")?.scrollIntoView({ block: "nearest" });
    const nextId = nextUnlabeledId(blindId);
    if (nextId) void loadCase(nextId).catch(() => undefined);
  } catch (error) {
    setSaveStatus(error.message, "error");
  }
}

function setFormLocked(locked) {
  for (const control of el.annotationForm.querySelectorAll("input, textarea")) control.disabled = locked;
  el.saveButton.disabled = locked;
  el.saveNextButton.disabled = locked;
}

function renderSource(content, reportedLine, displayRange) {
  const lines = String(content).replace(/\n$/, "").split("\n");
  return lines.map((line, index) => {
    const number = index + 1;
    const target = number === reportedLine;
    const nativeRange = Boolean(displayRange)
      && number >= displayRange.start_line
      && number <= displayRange.end_line;
    const classes = ["code-line"];
    if (nativeRange) classes.push("native-range");
    if (target) classes.push("target");
    const title = target
      ? "Frozen CodeLinter reported line"
      : (nativeRange ? "Native HomeCheck semantic range" : "");
    return `<span class="${classes.join(" ")}" data-line="${number}" title="${title}"><span class="line-number">${number}</span>${escapeHtml(line) || " "}</span>`;
  }).join("");
}

function jumpToFinding() {
  const target = el.sourceCode.querySelector(".code-line.target");
  if (!target) return;
  const paneBox = el.sourceCode.getBoundingClientRect();
  const targetBox = target.getBoundingClientRect();
  const currentOffset = targetBox.top - paneBox.top;
  const desiredOffset = (el.sourceCode.clientHeight - targetBox.height) / 2;
  el.sourceCode.scrollTo({
    top: Math.max(0, el.sourceCode.scrollTop + currentOffset - desiredOffset),
    left: 0,
    behavior: "auto",
  });
}

function renderCase() {
  const item = state.currentCase;
  const finding = item.finding;
  const index = state.data.cases.findIndex((entry) => entry.blind_id === item.blind_id);
  el.caseCategory.textContent = item.category;
  el.caseId.textContent = item.blind_id;
  el.casePosition.textContent = `${index + 1} of ${state.data.cases.length}`;
  el.ruleName.textContent = item.rule;
  el.findingLocation.textContent = `${finding.relative_path}:${finding.line}:${finding.column}`;
  el.findingMessage.textContent = finding.message;
  el.sourcePath.textContent = item.source.relative_path;
  el.sourcePath.title = item.source.relative_path;
  el.sourceCode.innerHTML = renderSource(item.source.content, finding.line, item.display_range);
  requestAnimationFrame(jumpToFinding);

  const annotation = item.annotation || {};
  for (const input of el.annotationForm.querySelectorAll('input[name="label"]')) {
    input.checked = input.value === annotation.label;
  }
  el.rationaleInput.value = annotation.rationale || "";
  el.rationaleHint.textContent = annotation.label && annotation.label !== "Correct" ? "required" : "optional for Correct";
  el.rationaleInput.classList.remove("invalid");

  if (state.data.role === "adjudicator") {
    el.authorDecision.hidden = false;
    el.authorDecision.innerHTML = `<span>Author 1 <strong>${escapeHtml(annotation.author_1_label)}</strong></span><span>Author 2 <strong>${escapeHtml(annotation.author_2_label)}</strong></span>`;
  } else {
    el.authorDecision.hidden = true;
    el.authorDecision.innerHTML = "";
  }
  renderEvidenceTabs();
  updateNavigation();
}

function renderEvidenceTabs() {
  const evidence = state.currentCase.rule_evidence || [];
  if (state.evidenceIndex >= evidence.length) state.evidenceIndex = 0;
  el.evidenceTabs.innerHTML = "";
  evidence.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item.name;
    button.title = item.name;
    button.className = index === state.evidenceIndex ? "active" : "";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", index === state.evidenceIndex ? "true" : "false");
    button.addEventListener("click", () => {
      state.evidenceIndex = index;
      renderEvidenceTabs();
    });
    el.evidenceTabs.appendChild(button);
  });
  el.evidenceContent.textContent = evidence[state.evidenceIndex]?.content || "No bundled rule evidence.";
  el.evidenceContent.scrollTop = 0;
}

function updateNavigation() {
  const index = state.filtered.findIndex((item) => item.blind_id === state.currentId);
  el.previousButton.disabled = index <= 0;
  el.nextUnlabeledButton.disabled = !state.data.cases.some((item) => !item.label && item.blind_id !== state.currentId);
}

async function selectRelative(offset, unlabeledOnly = false) {
  if (!(await flushDirty())) return;
  const pool = unlabeledOnly ? state.data.cases.filter((item) => !item.label) : state.filtered;
  if (!pool.length) return;
  const currentIndex = pool.findIndex((item) => item.blind_id === state.currentId);
  const nextIndex = currentIndex < 0 ? 0 : (currentIndex + offset + pool.length) % pool.length;
  await selectCase(pool[nextIndex].blind_id);
}

function showLockedView() {
  el.workspace.hidden = true;
  el.lockedView.hidden = false;
  el.lockedMessage.textContent = state.data.message;
  const progress = state.data.author_progress;
  el.authorProgress.innerHTML = `
    <span>Author 1 <strong>${progress.author_1} / ${progress.total}</strong></span>
    <span>Author 2 <strong>${progress.author_2} / ${progress.total}</strong></span>`;
}

async function initialize() {
  try {
    state.data = await api("/api/bootstrap");
    activeRole = state.data.role;
    el.roleLabel.textContent = roleTitle(state.data.role);
    renderRoleSwitcher();
    el.guideContent.textContent = state.data.guide || "";
    setProgress();
    if (state.data.mode === "locked") {
      showLockedView();
      return;
    }
    el.workspace.hidden = false;
    populateFilters();
    applyFilters();
    if (state.data.cases.length) await selectCase(state.data.cases[0].blind_id);
    else {
      el.workspace.hidden = true;
      el.lockedView.hidden = false;
      el.lockedView.querySelector("h2").textContent = "No disagreements require adjudication";
      el.lockedMessage.textContent = `The authors agreed on all ${state.data.agreement_count || 0} cases.`;
    }
  } catch (error) {
    el.fatalError.hidden = false;
    el.fatalError.textContent = `Unable to load annotation package: ${error.message}`;
  }
}

el.searchInput.addEventListener("input", applyFilters);
el.categoryFilter.addEventListener("change", applyFilters);
el.stateFilter.addEventListener("change", applyFilters);
el.annotationForm.addEventListener("input", (event) => {
  markDirty();
  const input = event.target.closest?.('input[name="label"]');
  if (input) {
    state.autoAdvancePending = true;
    if (input.value === "Correct") {
      saveCurrent({ advance: true });
    } else {
      setTimeout(() => el.rationaleInput.focus(), 0);
    }
  }
});
el.rationaleInput.addEventListener("blur", () => {
  if (state.autoAdvancePending && validateAnnotation(false)) {
    saveCurrent({ advance: true });
  }
});
el.annotationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  saveCurrent();
});
el.saveNextButton.addEventListener("click", () => saveCurrent({ advance: true }));
el.previousButton.addEventListener("click", () => selectRelative(-1));
el.nextUnlabeledButton.addEventListener("click", () => selectRelative(1, true));
el.jumpButton.addEventListener("click", jumpToFinding);
el.guideButton.addEventListener("click", () => el.guideDialog.showModal());
el.closeGuide.addEventListener("click", () => el.guideDialog.close());
el.guideDialog.addEventListener("click", (event) => {
  if (event.target === el.guideDialog) el.guideDialog.close();
});
window.addEventListener("beforeunload", (event) => {
  if (!state.dirty && state.pendingSaveCount === 0 && state.failedSaves.size === 0) return;
  event.preventDefault();
  event.returnValue = "";
});

initialize();
