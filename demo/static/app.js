/* dag-demo frontend: draw a DAG, assign A/B/C, ask Python (via the pywebview
   bridge) for the CMI. */

const COLORS = { A: "#e74c3c", B: "#3498db", C: "#27ae60", "": "#7f8c8d" };
const NAME_POOL = ["X", "Y", "Z", "U", "V", "W", "S", "T"];

const cy = cytoscape({
  container: document.getElementById("cy"),
  wheelSensitivity: 0.2,
  style: [
    {
      selector: "node",
      style: {
        label: (n) => n.data("id"),
        "background-color": (n) => COLORS[n.data("role") || ""],
        shape: (n) => (n.indegree(false) === 0 ? "round-rectangle" : "ellipse"),
        color: "#fff",
        "font-size": 14,
        "text-valign": "center",
        "text-halign": "center",
        width: 56, height: 44,
        "border-width": 0,
      },
    },
    {
      selector: "node:selected",
      style: { "border-width": 4, "border-color": "#f39c12" },
    },
    {
      selector: "node.connect-src",
      style: { "border-width": 4, "border-color": "#f39c12", "border-style": "dashed" },
    },
    {
      selector: "edge",
      style: {
        label: (e) => {
          const s = e.data("source");
          const pre = cy.getElementById(s).data("precoder");
          return `H_${s}${e.data("target")}` + (pre ? `·F_${s}` : "");
        },
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "arrow-scale": 1.4,
        width: 2.5,
        "line-color": "#aab4bd",
        "target-arrow-color": "#aab4bd",
        "font-size": 12,
        "text-background-color": "#fff",
        "text-background-opacity": 0.85,
        "text-background-padding": 2,
      },
    },
    {
      selector: "edge:selected",
      style: { "line-color": "#f39c12", "target-arrow-color": "#f39c12", width: 4 },
    },
  ],
});

/* ---- node / edge editing -------------------------------------------- */

function nextName() {
  const used = new Set(cy.nodes().map((n) => n.id()));
  for (const c of NAME_POOL) if (!used.has(c)) return c;
  let i = 1;
  while (used.has(`X${i}`)) i++;
  return `X${i}`;
}

function addNode(pos) {
  const name = prompt("Node name (alphanumeric)", nextName());
  if (!name) return;
  if (!/^[A-Za-z][A-Za-z0-9]*$/.test(name)) { setStatus("Node name must be alphanumeric and start with a letter.", "error"); return; }
  if (cy.getElementById(name).length) { setStatus(`Node ${name} already exists.`, "error"); return; }
  cy.add({ data: { id: name, role: null }, position: pos });
  refreshShapes();
}

cy.on("dbltap", (ev) => {
  if (ev.target === cy) addNode(ev.position);
});

/* connect mode: click parent, then child */
let connectMode = false;
let connectSrc = null;
const btnConnect = document.getElementById("btn-connect");

function setConnectMode(on) {
  connectMode = on;
  btnConnect.classList.toggle("active", on);
  if (!on && connectSrc) { connectSrc.removeClass("connect-src"); connectSrc = null; }
  cy.autoungrabify(on);
}
btnConnect.addEventListener("click", () => setConnectMode(!connectMode));

cy.on("tap", "node", (ev) => {
  if (!connectMode) return;
  const n = ev.target;
  if (!connectSrc) {
    connectSrc = n;
    n.addClass("connect-src");
  } else if (connectSrc.id() !== n.id()) {
    const id = `${connectSrc.id()}->${n.id()}`;
    if (!cy.getElementById(id).length) {
      cy.add({ data: { id, source: connectSrc.id(), target: n.id() } });
    }
    connectSrc.removeClass("connect-src");
    connectSrc = null;
    refreshShapes();
  }
});
cy.on("tap", (ev) => {
  if (connectMode && ev.target === cy && connectSrc) {
    connectSrc.removeClass("connect-src");
    connectSrc = null;
  }
});

function refreshShapes() { cy.style().update(); }

/* roles */
function assignRole(role) {
  const sel = cy.nodes(":selected");
  if (!sel.length) { setStatus("Select a node first.", "error"); return; }
  sel.forEach((n) => n.data("role", role));
  refreshShapes();
}
document.getElementById("btn-role-A").addEventListener("click", () => assignRole("A"));
document.getElementById("btn-role-B").addEventListener("click", () => assignRole("B"));
document.getElementById("btn-role-C").addEventListener("click", () => assignRole("C"));
document.getElementById("btn-role-none").addEventListener("click", () => assignRole(null));

function togglePrecoder() {
  const sel = cy.nodes(":selected");
  if (!sel.length) { setStatus("Select a node first.", "error"); return; }
  sel.forEach((n) => n.data("precoder", !n.data("precoder")));
  refreshShapes();
}
document.getElementById("btn-precoder").addEventListener("click", togglePrecoder);

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (k === "a" || k === "b" || k === "c") assignRole(k.toUpperCase());
  if (k === "n") assignRole(null);
  if (k === "p") togglePrecoder();
  if (k === "delete" || k === "backspace") deleteSelected();
  if (k === "escape") setConnectMode(false);
});

function deleteSelected() {
  cy.$(":selected").remove();
  refreshShapes();
}
document.getElementById("btn-add").addEventListener("click", () => {
  const ext = cy.extent();
  addNode({ x: (ext.x1 + ext.x2) / 2, y: (ext.y1 + ext.y2) / 2 });
});
document.getElementById("btn-delete").addEventListener("click", deleteSelected);
document.getElementById("btn-clear").addEventListener("click", () => {
  cy.elements().remove();
  document.getElementById("out").hidden = true;
  setStatus("Cleared. Double-click to add a node.");
});

/* ---- compute --------------------------------------------------------- */

const statusEl = document.getElementById("status");
function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

function katexInto(el, lines) {
  el.innerHTML = "";
  for (const s of lines) {
    const div = document.createElement("div");
    katex.render(s, div, { displayMode: true, throwOnError: false });
    el.appendChild(div);
  }
}

/* ---- output format switching (LaTeX rendered as math / Mathematica source) -- */

let outputFormat = "rendered"; // "rendered" (LaTeX, shown as math) | "mathematica"
let lastData = null; // last compute payload, re-rendered on a format switch

/* Per-block content as {latex: [...], math: [...]|null} (or {note} for the
   LMMSE multi-node case). Single source of truth shared by display and Copy. */
function blockContent(data, key) {
  switch (key) {
    case "model":
      return { latex: data.model || [], math: null };
    case "cmi":
      return { latex: [data.latex], math: data.mathematica ? [data.mathematica] : null };
    case "expanded": {
      const latex = [data.latex_expanded];
      if (data.latex_capacity) latex.push(data.latex_capacity);
      let math = null;
      if (data.mathematica_expanded) {
        math = [data.mathematica_expanded];
        if (data.mathematica_capacity) math.push(data.mathematica_capacity);
      }
      return { latex, math };
    }
    case "grad":
      return {
        latex: [data.gradient.latex],
        math: data.gradient.mathematica ? [data.gradient.mathematica] : null,
      };
    case "lmmse":
      if (data.lmmse.note) return { note: data.lmmse.note };
      return {
        latex: [data.lmmse.W, data.lmmse.E],
        math:
          data.lmmse.W_mathematica && data.lmmse.E_mathematica
            ? [data.lmmse.W_mathematica, data.lmmse.E_mathematica]
            : null,
      };
    default:
      return { latex: [], math: null };
  }
}

/* Pick the lines for the active format (Mathematica falls back to LaTeX source
   when no Wolfram form exists, so a block is never blank). */
function activeLines(content) {
  return outputFormat === "mathematica" && content.math && content.math.length
    ? content.math
    : content.latex;
}

/* Fill a math block either as rendered KaTeX or as a raw-source <pre>. */
function renderBlock(el, content) {
  if (outputFormat === "rendered") {
    katexInto(el, content.latex);
    return;
  }
  el.innerHTML = "";
  const pre = document.createElement("pre");
  pre.className = "code-block scroll";
  pre.textContent = activeLines(content).join("\n\n");
  el.appendChild(pre);
}

/* Reach the symbolic engine through the pywebview bridge (desktop.py's Api).
   The result payload matches core.run_compute; errors arrive as {_error}. */
async function callCompute(body) {
  if (!(window.pywebview && window.pywebview.api && window.pywebview.api.compute)) {
    throw new Error("Python bridge not found (launch with: uv run python desktop.py).");
  }
  const data = await window.pywebview.api.compute(body);
  if (data && data._error) throw new Error(data._error);
  return data;
}

function collectGraph() {
  const nodes = cy.nodes().map((n) => ({
    id: n.id(),
    role: n.data("role") || null,
    precoder: !!n.data("precoder"),
  }));
  const edges = cy.edges().map((e) => ({ source: e.data("source"), target: e.data("target") }));
  return { nodes, edges };
}

async function compute() {
  const { nodes, edges } = collectGraph();
  const selEdge = cy.edges(":selected");
  const selPreNode = cy.nodes(":selected").filter((n) => n.data("precoder"));
  const body = {
    nodes, edges,
    expand: document.getElementById("opt-expand").checked,
    check: document.getElementById("opt-check").checked,
    grad_edge: selEdge.length
      ? { source: selEdge[0].data("source"), target: selEdge[0].data("target") }
      : null,
    grad_node: !selEdge.length && selPreNode.length ? selPreNode[0].id() : null,
    lmmse: document.getElementById("opt-lmmse").checked,
  };

  setStatus("Computing… (symbolic derivation + verification)", "busy");
  document.getElementById("btn-compute").disabled = true;
  try {
    const data = await callCompute(body);
    render(data);
    setStatus("Done.");
  } catch (err) {
    document.getElementById("out").hidden = true;
    setStatus(err.message, "error");
  } finally {
    document.getElementById("btn-compute").disabled = false;
  }
}
document.getElementById("btn-compute").addEventListener("click", compute);

/* ---- code export ----------------------------------------------------- */

let codeLevel = "high";

async function callExport(body, level) {
  if (!(window.pywebview && window.pywebview.api && window.pywebview.api.export)) {
    throw new Error("Python bridge not found (launch with: uv run python desktop.py).");
  }
  const data = await window.pywebview.api.export(body, level);
  if (data && data._error) throw new Error(data._error);
  return data;
}

async function exportCode() {
  try {
    const data = await callExport(collectGraph(), codeLevel);
    document.getElementById("out").hidden = false;
    document.getElementById("code-wrap").hidden = false;
    document.getElementById("code").textContent = data.code;
    document.getElementById("btn-code-copy").textContent = "Copy";
    setStatus("Code generated.");
  } catch (err) {
    setStatus(err.message, "error");
  }
}

function setCodeLevel(level) {
  codeLevel = level;
  document.getElementById("btn-code-high").classList.toggle("active", level === "high");
  document.getElementById("btn-code-low").classList.toggle("active", level === "low");
  exportCode();
}

document.getElementById("btn-export").addEventListener("click", exportCode);
document.getElementById("btn-code-high").addEventListener("click", () => setCodeLevel("high"));
document.getElementById("btn-code-low").addEventListener("click", () => setCodeLevel("low"));
document.getElementById("btn-code-copy").addEventListener("click", async () => {
  const code = document.getElementById("code").textContent;
  try {
    await navigator.clipboard.writeText(code);
    document.getElementById("btn-code-copy").textContent = "Copied ✓";
  } catch {
    document.getElementById("btn-code-copy").textContent = "Copy failed";
  }
});

function render(data) {
  lastData = data;
  document.getElementById("out").hidden = false;
  resetCopyButtons();

  const badges = document.getElementById("badges");
  badges.innerHTML = "";
  const ind = document.createElement("span");
  if (data.independent) {
    ind.className = "badge ok";
    ind.textContent = "I = 0 — conditionally independent (symbolically proved)";
  } else {
    ind.className = "badge no";
    ind.textContent = "I ≠ 0 (independence not proved)";
  }
  badges.appendChild(ind);

  if (data.check) {
    const b = document.createElement("span");
    if (data.check.passed === true) {
      b.className = "badge ok";
      b.textContent = `PyTorch check ✓  max|err| = ${data.check.max_abs_err.toExponential(1)}`;
    } else if (data.check.passed === false) {
      b.className = "badge fail";
      b.textContent = `PyTorch check ✗  max|err| = ${data.check.max_abs_err.toExponential(1)}`;
    } else {
      b.className = "badge fail";
      b.textContent = `Check error: ${data.check.error}`;
    }
    badges.appendChild(b);
  }

  applyFormat(data);
}

/* Fill the format-sensitive blocks (model/CMI/expanded/gradient/LMMSE) in the
   current outputFormat. Badges are format-agnostic and stay as render() set them.
   Also toggles the optional wraps' visibility and disables Copy on absent blocks. */
function applyFormat(data) {
  renderBlock(document.getElementById("model"), blockContent(data, "model"));
  renderBlock(document.getElementById("cmi"), blockContent(data, "cmi"));

  const expWrap = document.getElementById("expanded-wrap");
  expWrap.hidden = !data.latex_expanded;
  if (data.latex_expanded) {
    renderBlock(document.getElementById("expanded"), blockContent(data, "expanded"));
  }

  const gradWrap = document.getElementById("grad-wrap");
  gradWrap.hidden = !data.gradient;
  if (data.gradient) {
    renderBlock(document.getElementById("grad"), blockContent(data, "grad"));
  }

  const lmWrap = document.getElementById("lmmse-wrap");
  lmWrap.hidden = !data.lmmse;
  if (data.lmmse) {
    const el = document.getElementById("lmmse");
    const content = blockContent(data, "lmmse");
    if (content.note) {
      // a note has no LaTeX/Mathematica variant — plain text in every mode
      el.innerHTML = "";
      const n = document.createElement("div");
      n.className = "status";
      n.textContent = content.note;
      el.appendChild(n);
    } else {
      renderBlock(el, content);
    }
  }
}

/* ---- format toggle ------------------------------------------------------- */

function setFormat(fmt) {
  outputFormat = fmt;
  for (const [id, f] of [
    ["btn-fmt-rendered", "rendered"],
    ["btn-fmt-math", "mathematica"],
  ]) {
    document.getElementById(id).classList.toggle("active", f === fmt);
  }
  resetCopyButtons();
  if (lastData) applyFormat(lastData); // re-render only, no recompute
}
document.getElementById("btn-fmt-rendered").addEventListener("click", () => setFormat("rendered"));
document.getElementById("btn-fmt-math").addEventListener("click", () => setFormat("mathematica"));

/* ---- per-block copy ------------------------------------------------------- */

function resetCopyButtons() {
  document.querySelectorAll(".copy-block").forEach((b) => (b.textContent = "Copy"));
}

/* The text a block contributes to the clipboard in the active format. */
function copyTextFor(key) {
  const content = blockContent(lastData, key);
  if (content.note) return content.note;
  return activeLines(content).join("\n\n");
}

document.getElementById("out").addEventListener("click", async (ev) => {
  const btn = ev.target.closest(".copy-block");
  if (!btn || !lastData) return;
  resetCopyButtons();
  try {
    await navigator.clipboard.writeText(copyTextFor(btn.dataset.copy));
    btn.textContent = "Copied ✓";
  } catch {
    btn.textContent = "Copy failed";
  }
});

/* ---- resizable results panel ----------------------------------------- */

const splitter = document.getElementById("splitter");
splitter.addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  splitter.classList.add("dragging");
  splitter.setPointerCapture(ev.pointerId);
  const onMove = (mv) => {
    const w = Math.max(320, window.innerWidth - mv.clientX);
    document.documentElement.style.setProperty("--panel-width", `${w}px`);
    cy.resize();
  };
  const onUp = () => {
    splitter.classList.remove("dragging");
    splitter.removeEventListener("pointermove", onMove);
    splitter.removeEventListener("pointerup", onUp);
    cy.resize();
  };
  splitter.addEventListener("pointermove", onMove);
  splitter.addEventListener("pointerup", onUp);
});

/* ---- preloaded example: two-hop relay X -> Y -> Z -------------------- */

cy.add([
  { data: { id: "X", role: "A" }, position: { x: 160, y: 220 } },
  { data: { id: "Y", role: "C" }, position: { x: 360, y: 160 } },
  { data: { id: "Z", role: "B" }, position: { x: 560, y: 220 } },
  { data: { id: "X->Y", source: "X", target: "Y" } },
  { data: { id: "Y->Z", source: "Y", target: "Z" } },
]);
cy.fit(undefined, 120);
