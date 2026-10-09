"use strict";
/* SplitSnap front end: vanilla JS, no framework, no third-party requests (only our own API).
   Flow: 1 Photo -> 2 Check the bill (correction UI) -> 3 People -> 4 Who had what -> 5 The split (explanation + overrides). */

window.__errs = [];                                      // read by the end-to-end test: any script error lands here
window.addEventListener("error", (e) => window.__errs.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => window.__errs.push(String(e.reason)));
const $ = (s, r = document) => r.querySelector(s);
function h(tag, attrs, ...kids) {                       // tiny DOM builder: text is always textContent (no HTML injection)
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}

function fill(parent, ...kids) {                          // replaceChildren, but skips null/false (conditional pieces)
  parent.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false));
}

/* ---------- money: Indian grouping, true minus ---------- */
const nf = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const rupees = (n) => (n < 0 ? "−" : "") + "₹" + nf.format(Math.abs(n));
const fromPaise = (p) => rupees(p / 100);
const toNum = (s) => { const x = parseFloat(String(s ?? "").replace(/Rs\.?|[₹,\s]/gi, "").replace("−", "-")); return Number.isFinite(x) ? x : null; };

/* ---------- state ---------- */
const MARKS_LIGHT = ["#8A2D22", "#2F3E6B", "#2A6168", "#7A5C0F", "#5E3A5C", "#4A5560", "#A4552B", "#5B6322"];
const MARKS_DARK = ["#E0735F", "#8FA3E8", "#6FC0C8", "#D9B23A", "#C08DC0", "#9AA8B5", "#E39560", "#A9BB5A"];   // lighter inks for the dark theme
const darkTheme = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const TYPE_LABEL = { CGST: "CGST", SGST: "SGST", IGST: "IGST", GST: "GST", VAT: "VAT", TAX: "Tax", SERVICE_CHARGE: "Service charge",
  PACKAGING: "Packaging", DISCOUNT: "Discount", ROUND_OFF: "Round off", OTHER: "Other" };
const STEPS = ["Photo", "Check", "People", "Assign", "Split"];
let uidN = 0; const uid = () => "u" + (++uidN);
const newItem = (o = {}) => ({ uid: uid(), name: "", qty: "1", unit_price: "", total: "", low: {}, ...o });
const newCharge = (o = {}) => ({ uid: uid(), type: "OTHER", amount: "", low: {}, ...o });
const S = {
  step: 1, hasBill: false, file: null, photoUrl: null, quality: null, busy: false, error: "", threshold: 0.7, demoServer: false,
  bill: { items: [], charges: [], subtotal: "", grand_total: "", low: {} }, issues: [], summary: null,
  people: [], removing: null, personErr: "", assign: {}, overrides: {}, adjusting: null, rebalance: false,
  result: null, expl: [], splitErr: "",
};
const cleanBill = () => ({
  items: S.bill.items.map(({ name, qty, unit_price, total }) => ({ name, qty, unit_price, total })),
  charges: S.bill.charges.map(({ type, amount }) => ({ type, amount })),
  subtotal: S.bill.subtotal, grand_total: S.bill.grand_total,
});
const markColor = (name) => (darkTheme() ? MARKS_DARK : MARKS_LIGHT)[Math.max(S.people.indexOf(name), 0) % 8];
const initial = (name) => ([...name.trim()][0] || "?").toUpperCase();
const markEl = (name, hollow) => h("span", { class: "mark" + (hollow ? " hollow" : ""), style: `--mk:${markColor(name)}`, "aria-hidden": "true" }, initial(name));

async function api(path, body) {
  const init = body instanceof FormData ? { method: "POST", body }
    : body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  let r;
  try { r = await fetch(path, init); } catch { throw new Error("Can't reach the server. Check the connection and try again."); }
  let data = null; try { data = await r.json(); } catch { /* not json */ }
  if (!r.ok) {
    const d = data && data.detail;
    const e = new Error(typeof d === "string" ? d : "That didn't work. Check the entries and try again.");
    e.status = r.status; throw e;
  }
  return data;
}

/* ---------- adopt a bill from the server (model output or sample) ---------- */
function adopt(bill, fields, threshold) {
  const low = new Set((fields || []).filter((f) => f.p_min < threshold).map((f) => f.path));
  const flags = (prefix, keys) => Object.fromEntries(keys.filter((k) => low.has(`${prefix}.${k}`)).map((k) => [k, true]));
  S.bill = {
    items: bill.items.map((it, i) => newItem({ ...it, low: flags(`items[${i}]`, ["name", "qty", "unit_price", "total"]) })),
    charges: bill.charges.map((c, i) => newCharge({ ...c, low: flags(`charges[${i}]`, ["type", "amount"]) })),
    subtotal: bill.subtotal, grand_total: bill.grand_total,
    low: { subtotal: low.has("subtotal"), grand_total: low.has("grand_total") },
  };
  S.hasBill = true; S.assign = {}; S.overrides = {}; S.result = null; S.issues = []; S.summary = null;
}

/* ---------- navigation ---------- */
function unassigned() { return S.bill.items.filter((it) => !S.assign[it.uid] || !Object.values(S.assign[it.uid]).some((u) => u > 0)); }
function reachable(n) {
  if (n === 1) return true;
  if (n === 2 || n === 3) return S.hasBill;
  if (n === 4) return S.hasBill && S.people.length > 0;
  return S.hasBill && S.people.length > 0 && unassigned().length === 0;
}
function go(n) {
  if (!reachable(n)) return;
  S.step = n; render();
  const m = $("#main"); m.focus({ preventScroll: true }); window.scrollTo(0, 0);
}

function renderSteps() {
  fill($("#steps"), h("div", { class: "seg" }, ...STEPS.map((label, i) =>
    h("button", { type: "button", class: i + 1 < S.step ? "done" : "", "aria-current": S.step === i + 1 ? "step" : false,
      "aria-label": `Step ${i + 1}: ${label}`, disabled: !reachable(i + 1), onclick: () => go(i + 1) }))));
  $("#where").textContent = `${STEPS[S.step - 1]} · ${S.step} of 5`;
  renderSlip();
}

/* the bill, kept in view beside the step on a wide screen (hidden on a phone) */
function sampleSlip() {
  const row = (t, a, c) => h("div", { class: "row" + (c ? " " + c : "") }, h("span", { class: "t" }, t), h("span", { class: "dots" }), h("span", { class: "amt" }, a));
  const pal = darkTheme() ? MARKS_DARK : MARKS_LIGHT;
  return h("div", { class: "sample" }, h("h3", {}, "How it works"),
    h("p", { class: "cap2" }, "A sample bill split three ways. Taxes, service charge and discounts follow each person's share of the items, not the headcount."),
    row("Paneer Butter Masala", "₹220.00"), row("3 × Garlic Naan", "₹120.00"), row("Veg Biryani", "₹260.00"), row("Cold Coffee", "₹140.00"),
    h("div", { style: "height:8px" }), row("CGST + SGST", "₹37.00"), row("Service charge", "₹40.00"), row("Discount", "−₹50.00"), row("Total", "₹767.00", "total"),
    h("div", { class: "shares" }, ...[["Riya", "₹269.48"], ["Aman", "₹310.95"], ["Sara", "₹186.57"]].map(([n, a], i) =>
      h("div", { class: "row" }, h("span", { class: "t" }, h("span", { class: "mark", style: `--mk:${pal[i]}` }, n[0]), n), h("span", { class: "dots" }), h("span", { class: "amt" }, a)))));
}
function renderSlip() {
  const el = $("#slip"); if (!el) return;
  if (S.step === 1) { el.classList.add("on"); fill(el, sampleSlip()); return; }
  const show = S.hasBill && S.step >= 2;
  el.classList.toggle("on", show);
  const photo = show && S.step === 2 && S.photoUrl;
  document.querySelector(".page").classList.toggle("with-photo", !!photo);
  if (photo) {
    fill(el, h("h3", {}, "Your photo"),
      h("a", { href: S.photoUrl, target: "_blank", rel: "noopener", class: "photo-link", "aria-label": "Open the photo full size" }, h("img", { src: S.photoUrl, alt: "The bill photo you added" })),
      h("p", { class: "note" }, "Opens full size when you click it."));
    return;
  }
  if (!show) return;
  const row = (t, amt, cls) => h("div", { class: "row" + (cls ? " " + cls : "") }, h("span", { class: "t" }, t), h("span", { class: "dots" }), h("span", { class: "amt" }, amt));
  const items = S.bill.items.filter((it) => it.name || it.total);
  fill(el, h("h3", {}, "The bill"),
    ...items.map((it) => row((it.qty && it.qty !== "1" ? it.qty + " × " : "") + (it.name || "Item"), rupees(toNum(it.total) ?? 0))),
    S.bill.charges.length ? h("div", { style: "height:12px" }) : null,
    ...S.bill.charges.map((c) => row(TYPE_LABEL[c.type] || c.type, rupees(toNum(c.amount) ?? 0))),
    row("Total", rupees(toNum(S.bill.grand_total) ?? 0), "total"));
}

function setBar(left, right, above) {
  const bar = $("#bar");
  if (!left && !right) { bar.hidden = true; return; }
  bar.hidden = false;
  fill($("#bar-in"), above ? h("div", { class: "bar-above" }, above) : null, h("div", { class: "bar-main" }, left || h("span"), right || h("span")));
}

function render() {
  renderSteps();
  ({ 1: step1, 2: step2, 3: step3, 4: step4, 5: step5 })[S.step]();
}

/* ======================================================================= step 1: photo */
const PROBLEMS = {
  low_resolution: "The photo is small. Move closer so the bill fills the frame.",
  blurry: "The photo is blurry. Hold the phone steady and tap to focus.",
  too_dark: "The photo is too dark to read. Move to the light.",
  washed_out: "The photo is washed out. Tilt the bill to avoid glare.",
  low_contrast: "The print is faint in this photo. Try more light.",
};

function step1() {
  const cam = h("input", { type: "file", accept: "image/*", capture: "environment", hidden: true, id: "cam", onchange: (e) => pickFile(e.target.files[0]) });
  const pick = h("input", { type: "file", accept: "image/*", hidden: true, id: "pick", onchange: (e) => pickFile(e.target.files[0]) });
  const q = S.quality;
  fill($("#main"),
    h("h1", {}, "Add the bill"),
    h("p", { class: "lede" }, "Take a photo of the printed bill, or choose one from your phone."),
    cam, pick,
    h("button", { class: "btn block", type: "button", id: "take", onclick: () => cam.click() }, "Take a photo"),
    h("div", { class: "actions" }, h("button", { class: "text", type: "button", id: "choose", onclick: () => pick.click() }, "or choose a file")),
    S.photoUrl && h("img", { class: "preview", src: S.photoUrl, alt: "The bill you added" }),
    S.busy && h("div", { class: "reading", role: "status" }, h("span", { class: "sweep" }), "Reading the bill"),
    S.error && h("p", { class: "note bad", role: "alert" }, S.error),
    S.partial && S.error && h("div", { class: "actions" }, h("button", { class: "btn", type: "button", id: "hand-partial", onclick: () => enterByHand(String(S.partial.total || "")) }, "Enter the items by hand")),
    q && q.retake && h("div", { class: "retake", role: "status" },
      h("p", {}, (q.problems || []).map((p) => PROBLEMS[p]).filter(Boolean).join(" ")),
      h("div", { class: "actions", style: "margin:0" },
        h("button", { class: "text", type: "button", onclick: () => cam.click() }, "Retake"),
        h("button", { class: "text", type: "button", onclick: () => { S.quality = null; go(2); } }, "Use it anyway"))),
    S.file && !S.busy && !(q && q.retake) && h("div", { class: "actions" },
      h("button", { class: "btn", type: "button", id: "read", onclick: readBill }, "Read the bill")),
    S.examples && S.examples.length ? h("div", { class: "examples-box" }, h("h2", {}, "Or try an example photo"),
      h("div", { class: "examples", id: "examples" }, ...S.examples.map((f, i) =>
        h("button", { class: "ex", type: "button", "aria-label": "Example bill " + (i + 1), onclick: () => loadExample(f) }, h("img", { src: "/example/" + f, alt: "", loading: "lazy" }))))) : null,
    h("h2", {}, "No bill to hand?"),
    h("ul", { class: "samples" },
      h("li", {}, h("button", { class: "text", type: "button", id: "sample", onclick: () => loadSample("clean") }, "Try a sample"), h("span", { class: "note" }, "Everything adds up")),
      h("li", {}, h("button", { class: "text", type: "button", id: "sample2", onclick: () => loadSample("messy") }, "Try a messy sample"), h("span", { class: "note" }, "Shows how corrections work")),
      h("li", {}, h("button", { class: "text", type: "button", id: "hand", onclick: enterByHand }, "Enter the bill by hand"), h("span", { class: "note" }, "No photo needed"))));
  setBar(null, null); renderSteps();
}

function dropPhoto() { if (S.photoUrl) URL.revokeObjectURL(S.photoUrl); S.photoUrl = null; S.file = null; }
async function loadExamples() {
  try { const r = await fetch("/examples"); if (r.ok) { S.examples = (await r.json()).files; if (S.step === 1) step1(); } } catch (e) { /* the list is a convenience */ }
}
async function loadExample(name) {
  try { const b = await (await fetch("/example/" + name)).blob(); pickFile(new File([b], name, { type: b.type || "image/jpeg" })); }
  catch (e) { S.error = "Couldn't open that example."; step1(); }
}
function pickFile(f) {
  if (!f) return;
  if (S.photoUrl) URL.revokeObjectURL(S.photoUrl);
  S.file = f; S.photoUrl = URL.createObjectURL(f); S.error = ""; S.quality = null; step1();
  readBill();
}
async function readBill() {
  if (!S.file || S.busy) return;
  S.busy = true; S.error = ""; step1();
  try {
    const fd = new FormData(); fd.append("file", S.file);
    const r = await api("/extract", fd);
    S.threshold = r.conf_threshold; S.quality = r.quality; adopt(r.bill, r.fields, r.conf_threshold); S.issues = r.issues;
    S.busy = false;
    if (!S.bill.items.length) {                                // the model found nothing: say so, offer the alternatives
      S.hasBill = false; S.partial = r.partial || null;
      const pr = S.partial, shop = pr && pr.company, tot = pr && pr.total;
      S.error = pr && (shop || tot)
        ? "We couldn't read the items on this photo" + (shop ? ", but we could read the shop (" + shop + ")" : "") + (tot ? (shop ? " and" : ", but we could read") + " a total of " + tot : "") + ". You can enter the items by hand" + (tot ? " and the total will be filled in." : ".")
        : "We couldn't read any items in this photo. Try a flatter, closer photo with the whole bill in view and good light, or enter the bill by hand.";
      step1(); return;
    }
    if (r.quality && r.quality.retake) { step1(); return; }    // show the retake prompt first; "Use it anyway" continues
    go(2);
  } catch (e) { S.busy = false; S.error = e.message + (e.status === 503 ? "" : ""); step1(); }
}
async function loadSample(kind) {
  dropPhoto();
  try { const r = await api("/demo?kind=" + kind); S.threshold = r.conf_threshold; adopt(r.bill, r.fields, r.conf_threshold); S.issues = r.issues; S.quality = null; go(2); }
  catch (e) { S.error = e.message; step1(); }
}
function enterByHand(total) { const keep = typeof total === "string"; if (!keep) dropPhoto(); S.bill = { items: [newItem()], charges: [], subtotal: "", grand_total: keep ? total : "", low: {} }; S.hasBill = true; S.assign = {}; S.overrides = {}; S.result = null; S.quality = null; go(2); }

/* ======================================================================= step 2: check the bill */
let vTimer = 0, vSeq = 0;

function scheduleValidate() { clearTimeout(vTimer); renderSlip(); vTimer = setTimeout(runValidate, 250); }

async function runValidate() {
  const seq = ++vSeq;
  try { const r = await api("/validate", { bill: cleanBill() }); if (seq !== vSeq) return; S.issues = r.issues; S.summary = r.summary; paintBill(); }
  catch { /* keep editing; the split step re-checks */ }
}

function field(label, key, value, o = {}) {
  const input = h("input", { class: (o.num ? "num " : "") + (o.cls || ""), type: "text", inputmode: o.mode || "text", value: value ?? "", placeholder: o.ph || "",
    autocomplete: "off", autocapitalize: o.cap || "off", "data-path": o.path, "aria-label": label + (o.aria ? " " + o.aria : ""),
    oninput: (e) => { o.set(e.target.value); if (o.clearLow) o.clearLow(); const f = e.target.closest(".field"); f && f.classList.remove("low"); scheduleValidate(); } });
  return h("label", { class: "field" + (o.low ? " low" : ""), "data-fpath": o.path }, h("span", { class: "cap" }, label), input);
}


function itemRow(it, i) {
  const set = (k) => (v) => { it[k] = v; }, clear = (k) => () => { delete it.low[k]; };
  return h("div", { class: "item" + (i === 0 ? " first" : ""), "data-row": `items[${i}]` },
    field("Item", "name", it.name, { path: `items[${i}].name`, set: set("name"), clearLow: clear("name"), low: it.low.name, ph: "Item name", cap: "words", aria: i + 1, cls: "nm" }),
    h("div", { class: "nums" },
      field("Qty", "qty", it.qty, { path: `items[${i}].qty`, set: set("qty"), clearLow: clear("qty"), low: it.low.qty, mode: "numeric", num: true, aria: i + 1 }),
      field("Rate", "unit_price", it.unit_price, { path: `items[${i}].unit_price`, set: set("unit_price"), clearLow: clear("unit_price"), low: it.low.unit_price, mode: "decimal", num: true, aria: i + 1 }),
      field("Amount", "total", it.total, { path: `items[${i}].total`, set: set("total"), clearLow: clear("total"), low: it.low.total, mode: "decimal", num: true, aria: i + 1 }),
      h("button", { class: "x danger", type: "button", "aria-label": `Remove item ${i + 1}`, onclick: () => { S.bill.items.splice(i, 1); step2(); scheduleValidate(); } }, "×")),
    h("p", { class: "note", "data-note": `items[${i}]` }));
}


function chargeRow(c, i) {
  const sel = h("select", { "aria-label": "Charge type " + (i + 1), onchange: (e) => { c.type = e.target.value; delete c.low.type; scheduleValidate(); } },
    ...Object.entries(TYPE_LABEL).map(([v, l]) => h("option", { value: v, selected: v === c.type }, l)));
  return h("div", { class: "charge" + (i === 0 ? " first" : ""), "data-row": `charges[${i}]` },
    h("div", { class: "nums" },
      h("label", { class: "field" + (c.low.type ? " low" : "") }, h("span", { class: "cap" }, "Charge"), sel),
      field("Amount", "amount", c.amount, { path: `charges[${i}].amount`, set: (v) => { c.amount = v; }, clearLow: () => { delete c.low.amount; }, low: c.low.amount, mode: "decimal", num: true, aria: i + 1 }),
      h("button", { class: "x danger", type: "button", "aria-label": `Remove charge ${i + 1}`, onclick: () => { S.bill.charges.splice(i, 1); step2(); scheduleValidate(); } }, "×")),
    h("p", { class: "note", "data-note": `charges[${i}]` }));
}


function step2() {
  const b = S.bill;
  fill($("#main"),
    h("h1", {}, "Check what we read"),
    h("p", { class: "legend" }, "Fix anything that's wrong. ", h("span", { class: "hl" }, "Highlighted"), " means we're unsure; ",
      h("span", { class: "rp" }, "underlined in red"), " means it doesn't add up."),
    S.photoUrl && h("details", { class: "peek" }, h("summary", {}, "Compare with your photo"),
      h("a", { href: S.photoUrl, target: "_blank", rel: "noopener" }, h("img", { src: S.photoUrl, alt: "The bill photo you added" }))),
    h("div", { class: "status", id: "status", role: "status" }),
    h("div", {}, ...b.items.map(itemRow), h("div", { class: "list-end" })),
    h("div", { class: "actions" }, h("button", { class: "text", type: "button", id: "add-item", onclick: () => { b.items.push(newItem()); step2(); const l = document.querySelectorAll(".item input"); l[l.length - 4]?.focus(); } }, "Add an item")),
    h("h2", {}, "Charges"),
    b.charges.length ? h("div", {}, ...b.charges.map(chargeRow), h("div", { class: "list-end" })) : h("p", { class: "note" }, "No charges on this bill."),
    h("div", { class: "actions" }, h("button", { class: "text", type: "button", id: "add-charge", onclick: () => { b.charges.push(newCharge()); step2(); } }, "Add a charge")),
    h("h2", {}, "Totals on the bill"),
    h("div", { class: "pair" },
      field("Items subtotal", "subtotal", b.subtotal, { path: "subtotal", set: (v) => { b.subtotal = v; }, clearLow: () => { b.low.subtotal = false; }, low: b.low.subtotal, mode: "decimal", num: true }),
      field("Total payable", "grand_total", b.grand_total, { path: "grand_total", set: (v) => { b.grand_total = v; updateBar2(); }, clearLow: () => { b.low.grand_total = false; }, low: b.low.grand_total, mode: "decimal", num: true })),
    h("p", { class: "note", "data-note": "subtotal" }), h("p", { class: "note", "data-note": "grand_total" }),
    h("h2", {}, "How the bill breaks down"),
    h("div", { class: "ledger", id: "ledger" }),
    h("p", { class: "note", id: "tick" }));
  paintBill(); updateBar2(); scheduleValidate(); renderSteps();
}

function applyFix(sg) {                                     // a suggested value from the validator: one tap, never applied on its own
  const m = sg.path.match(/^items\[(\d+)\]\.(\w+)$/); if (!m) return;
  const row = S.bill.items[+m[1]]; row[m[2]] = sg.value; delete row.low[m[2]];
  const y = window.scrollY; step2(); window.scrollTo(0, y); scheduleValidate();
}

function paintBill() {
  const byPath = {};
  for (const i of S.issues) (byPath[i.path] ||= []).push(i);
  document.querySelectorAll("[data-fpath]").forEach((f) => { const bad = (byPath[f.dataset.fpath] || []).length > 0; f.classList.toggle("bad", bad); });
  document.querySelectorAll("[data-note]").forEach((n) => {
    const key = n.dataset.note;
    const msgs = S.issues.filter((i) => i.path === key || i.path.startsWith(key + ".")).map((i) => i.message);
    let low = false;
    const idx = key.match(/^(items|charges)\[(\d+)\]$/);
    if (idx) { const row = S.bill[idx[1]][+idx[2]]; low = row && Object.values(row.low).some(Boolean); }
    else if (S.bill.low[key]) low = true;
    n.className = "note" + (msgs.length ? " bad" : "");
    n.textContent = msgs.length ? msgs.join(" ") : low ? "We're not sure about the highlighted part. Check it against the bill." : "";
    const sg = (S.issues.find((i) => i.suggestion && (i.path === key || i.path.startsWith(key + "."))) || {}).suggestion;
    if (sg && msgs.length) n.append(" ", h("button", { class: "text fix", type: "button", onclick: () => applyFix(sg) }, sg.label));
  });
  const st = $("#status");
  if (st) {
    const bad = S.issues.length, lows = [...S.bill.items, ...S.bill.charges].reduce((n, r) => n + Object.values(r.low).filter(Boolean).length, 0) + (S.bill.low.subtotal ? 1 : 0) + (S.bill.low.grand_total ? 1 : 0);
    const nI = S.bill.items.length, nC = S.bill.charges.length, todo = bad + lows;
    fill(st, h("b", {}, `${nI} item${nI === 1 ? "" : "s"}`), h("span", {}, "and"), h("b", {}, `${nC} charge${nC === 1 ? "" : "s"}`), h("span", {}, "read."),
      todo ? h("b", { class: "chk" }, `${todo} to check`) : h("b", { class: "ok" }, "✓ Looks right"));
  }
  const L = $("#ledger");
  if (L) {
    fill(L);
    for (const [k, v] of Object.entries(S.summary || {})) L.append(h("div", { class: "row" + (k === "Total" ? " total" : "") },
      h("span", {}, k), h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(v))));
    const t = $("#tick"); const errs = S.issues.filter((i) => i.severity === "error");
    t.className = "note " + (errs.length ? "bad" : "ok");
    t.textContent = !S.summary ? "" : errs.length ? errs.length + (errs.length === 1 ? " thing doesn't add up." : " things don't add up.") + " You can still continue." : "✓ Adds up to the total on the bill.";
  }
  updateBar2();
}
function updateBar2() {
  if (S.step !== 2) return;
  const g = toNum(S.bill.grand_total);
  setBar(h("div", {}, h("span", { class: "tot" }, g == null ? "₹—" : rupees(g)), h("span", { class: "sub" }, "Total on the bill")),
    h("button", { class: "btn", type: "button", id: "to-people", disabled: S.bill.items.length === 0, onclick: () => go(3) }, "Add people"));
}

/* ======================================================================= step 3: people */
function addPerson(raw) {
  const name = raw.trim().replace(/\s+/g, " ").slice(0, 40);
  if (!name) return;
  if (S.people.some((p) => p.toLowerCase() === name.toLowerCase())) { S.personErr = `${name} is already on the list.`; return step3(); }
  S.people.push(name); S.personErr = ""; S.result = null; step3(); $("#pname")?.focus();
}
function removePerson(name) {
  const has = S.bill.items.filter((it) => (S.assign[it.uid] || {})[name] > 0).length;
  if (has && S.removing !== name) { S.removing = name; return step3(); }
  S.people = S.people.filter((p) => p !== name);
  for (const k of Object.keys(S.assign)) delete S.assign[k][name];
  delete S.overrides[name]; S.removing = null; S.result = null;
  S.step === 4 ? step4() : step3();
}
function step3() {
  const input = h("input", { id: "pname", type: "text", autocomplete: "off", autocapitalize: "words", placeholder: "Name", "aria-label": "Name" });
  fill($("#main"), 
    h("h1", {}, "Who's eating"),
    h("p", { class: "lede" }, "Add everyone who shared the bill. You can add or remove people later."),
    h("form", { class: "add-person", onsubmit: (e) => { e.preventDefault(); addPerson(input.value); } },
      h("label", { class: "field" }, h("span", {}, "Name"), input), h("button", { class: "btn", type: "submit", id: "add-person" }, "Add")),
    S.personErr && h("p", { class: "note bad", role: "alert" }, S.personErr),
    h("ul", { class: "people" }, ...S.people.map((n) => {
      const has = S.bill.items.filter((it) => (S.assign[it.uid] || {})[n] > 0).length;
      return h("li", {}, markEl(n), h("span", { class: "nm" }, n),
        h("button", { class: "text danger", type: "button", "aria-label": "Remove " + n, onclick: () => removePerson(n) }, S.removing === n ? `Remove (${has} item${has === 1 ? "" : "s"})?` : "Remove"));
    })),
    S.people.length === 0 && h("p", { class: "note" }, "Nobody yet."));
  setBar(h("div", {}, h("span", { class: "tot" }, rupees(toNum(S.bill.grand_total) ?? 0)), h("span", { class: "sub" }, S.people.length + (S.people.length === 1 ? " person" : " people"))),
    h("button", { class: "btn", type: "button", id: "to-assign", disabled: S.people.length === 0, onclick: () => go(4) }, "Who had what"));
  renderSteps();
}

/* ======================================================================= step 4: who had what */
function setUnits(it, name, units) {
  const a = (S.assign[it.uid] ||= {});
  if (units <= 0) delete a[name]; else a[name] = Math.min(units, 99);
  S.result = null; step4(true);
}
function tallyNow() {                                            // each person's pre-tax share of the items assigned so far
  const t = Object.fromEntries(S.people.map((p) => [p, 0]));
  for (const it of S.bill.items) {
    const a = S.assign[it.uid] || {}, units = Object.values(a).reduce((x, y) => x + y, 0), amt = toNum(it.total) || 0;
    if (units) for (const [p, u] of Object.entries(a)) if (p in t) t[p] += (amt * u) / units;
  }
  return t;
}

function step4(keepScroll) {
  const y = window.scrollY;
  const left = unassigned().length;
  fill($("#main"), 
    h("h1", {}, "Who had what"),
    h("p", { class: "lede" }, "Tap the names that shared each item. Use + to give someone more than one share."),
    h("div", { class: "actions", style: "margin-top:0" },
      h("button", { class: "text", type: "button", id: "equal-all", onclick: () => { S.bill.items.forEach((it) => { S.assign[it.uid] = Object.fromEntries(S.people.map((p) => [p, 1])); }); S.result = null; step4(true); } }, "Split every item equally")),
    ...S.bill.items.map((it) => {
      const a = S.assign[it.uid] || {};
      return h("div", { class: "assign-item", "data-item": it.uid },
        h("div", { class: "row" }, h("span", {}, it.name || "Item"), h("span", { class: "dots" }), h("span", { class: "amt" }, rupees(toNum(it.total) ?? 0))),
        h("div", { class: "chips" }, ...S.people.map((p) => {
          const on = a[p] > 0;
          return h("span", { style: "display:inline-flex;align-items:center" },
            h("button", { class: "chip", type: "button", "aria-pressed": on ? "true" : "false", "data-person": p, onclick: () => setUnits(it, p, on ? 0 : 1) }, markEl(p, !on), p),
            on && h("span", { class: "stepper" },
              h("button", { type: "button", "aria-label": `Fewer shares for ${p}`, onclick: () => setUnits(it, p, a[p] - 1) }, "−"),
              h("b", {}, a[p]),
              h("button", { type: "button", "aria-label": `More shares for ${p}`, onclick: () => setUnits(it, p, a[p] + 1) }, "+")));
        })),
        h("div", { class: "actions", style: "margin:0" },
          h("button", { class: "text", type: "button", onclick: () => { S.assign[it.uid] = Object.fromEntries(S.people.map((p) => [p, 1])); S.result = null; step4(true); } }, "Everyone"),
          h("button", { class: "text", type: "button", onclick: () => { delete S.assign[it.uid]; S.result = null; step4(true); } }, "Clear")));
    }),
    h("h2", {}, "Someone missing?"),
    h("form", { class: "add-person", onsubmit: (e) => { e.preventDefault(); const i = $("#late"); const v = i.value; i.value = ""; if (v.trim()) { const before = S.people.length; addPersonLate(v); if (S.people.length === before) return; } } },
      h("label", { class: "field" }, h("span", {}, "Add a name"), h("input", { id: "late", type: "text", autocomplete: "off", autocapitalize: "words", "aria-label": "Add a name" })),
      h("button", { class: "btn", type: "submit" }, "Add")),
    S.personErr && h("p", { class: "note bad", role: "alert" }, S.personErr),
    h("ul", { class: "people" }, ...S.people.map((n) => h("li", {}, markEl(n), h("span", { class: "nm" }, n),
      h("button", { class: "text danger", type: "button", "aria-label": "Remove " + n, onclick: () => removePerson(n) }, S.removing === n ? "Remove? They have items" : "Remove")))));
  const t = tallyNow();
  setBar(h("div", {}, h("span", { class: "tot" }, rupees(toNum(S.bill.grand_total) ?? 0)),
      h("span", { class: "sub" }, left === 0 ? "Every item is assigned" : left + (left === 1 ? " item" : " items") + " left to assign")),
    h("button", { class: "btn", type: "button", id: "to-split", "aria-disabled": left > 0 ? "true" : "false", onclick: () => { if (left === 0) go(5); } }, "See the split"),
    h("div", { class: "tally", "aria-label": "Items assigned to each person so far" }, ...S.people.map((p) => h("span", { class: "t" }, markEl(p), p, h("b", {}, rupees(t[p]))))));
  renderSteps();
  if (keepScroll) window.scrollTo(0, y);
}
function addPersonLate(raw) {
  const name = raw.trim().replace(/\s+/g, " ").slice(0, 40);
  if (S.people.some((p) => p.toLowerCase() === name.toLowerCase())) { S.personErr = `${name} is already on the list.`; return step4(true); }
  S.people.push(name); S.personErr = ""; S.result = null; step4(true);
}

/* ======================================================================= step 5: the split */
async function runSplit() {
  const assignments = {};
  S.bill.items.forEach((it, i) => { if (S.assign[it.uid]) assignments[i] = { ...S.assign[it.uid] }; });
  try {
    const r = await api("/split", { bill: cleanBill(), people: S.people, assignments, overrides: S.overrides, rebalance: S.rebalance });
    S.result = r.result; S.expl = r.explanation; S.summary = r.summary; S.splitErr = "";
  } catch (e) { S.result = null; S.splitErr = e.message; }
  if (S.step === 5) step5(true);
}
function personCard(name, idx) {
  const p = S.result.people[name];
  const lines = [
    ...p.items.map((l) => h("div", { class: "row" }, h("span", {}, `${l.name} (${l.units === l.total_units ? "full" : l.units + "/" + l.total_units + " share"})`), h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(l.amount)))),
    h("div", { class: "row total" }, h("span", {}, "Your items"), h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(p.pre_tax))),
    ...p.charges.filter((c) => c.amount !== 0).map((c) => h("div", { class: "row" }, h("span", {}, TYPE_LABEL[c.type] || c.type), h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(c.amount)))),
  ];
  const pct = p.pre_tax ? ((p.charge_total / p.pre_tax) * 100).toFixed(1) : null;
  const editing = S.adjusting === name;
  return h("section", { class: "person", "data-person": name },
    h("div", { class: "head" }, markEl(name), h("span", { class: "nm" }, name), h("div", { class: "row big" }, h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(p.final)))),
    p.manual && h("p", { style: "margin:4px 0 0" }, h("span", { class: "stamp" }, "ADJUSTED BY HAND"), h("span", { class: "was" }, "worked out as " + fromPaise(p.computed_final))),
    p.auto_rebalanced && h("p", { class: "note" }, "Adjusted from " + fromPaise(p.computed_final) + " to cover the amount other people changed."),
    h("details", {}, h("summary", {}, "How this was worked out"),
      h("div", { class: "ledger" }, ...lines,
        h("div", { class: "row total" }, h("span", {}, "Final"), h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(p.final)))),
      pct !== null && h("p", { class: "note" }, `Charges and discounts follow your share of the pre-tax bill: ${fromPaise(p.charge_total)} on ${fromPaise(p.pre_tax)} (${pct}%).`),
      h("p", { class: "expl" }, S.expl[idx + 1] || "")),
    editing ? h("form", { class: "override", onsubmit: (e) => { e.preventDefault(); const v = $("#ov").value.trim(); S.adjusting = null; if (toNum(v) != null) S.overrides[name] = String(toNum(v)); runSplit(); } },
        h("label", { class: "field" }, h("span", {}, "New amount for " + name), h("input", { id: "ov", class: "num", inputmode: "decimal", type: "text", value: String(p.final / 100), autocomplete: "off" })),
        h("button", { class: "btn", type: "submit" }, "Set"), h("button", { class: "text", type: "button", onclick: () => { S.adjusting = null; step5(true); } }, "Cancel"))
      : h("div", { class: "actions", style: "margin:0" },
          h("button", { class: "text", type: "button", "data-adjust": name, onclick: () => { S.adjusting = name; step5(true); setTimeout(() => $("#ov")?.select(), 0); } }, "Adjust the amount"),
          p.manual && h("button", { class: "text", type: "button", onclick: () => { delete S.overrides[name]; runSplit(); } }, "Undo the adjustment")));
}
function matrixTable(R) {
  const P = S.people, bill = S.bill, sum = (xs) => xs.reduce((a, b) => a + b, 0);
  const cell = (v) => h("td", { class: v < 0 ? "neg" : "" }, fromPaise(v));
  const rows = [
    h("tr", {}, h("td", {}, "Items"), ...P.map((n) => cell(R.people[n].pre_tax)), h("td", { class: "billcol" }, fromPaise(sum(P.map((n) => R.people[n].pre_tax))))),
    ...bill.charges.map((c, i) => {
      const v = P.map((n) => (R.people[n].charges[i] ? R.people[n].charges[i].amount : 0));
      return h("tr", { class: "sub" }, h("td", {}, TYPE_LABEL[c.type] || c.type), ...v.map(cell), h("td", { class: "billcol" }, fromPaise(sum(v))));
    }),
    h("tr", { class: "fin" }, h("td", {}, "Total"), ...P.map((n) => cell(R.people[n].pre_tax + R.people[n].charge_total)),
      h("td", { class: "billcol" }, fromPaise(sum(P.map((n) => R.people[n].pre_tax + R.people[n].charge_total))))),
  ];
  if (P.some((n) => R.people[n].final !== R.people[n].pre_tax + R.people[n].charge_total))
    rows.push(h("tr", { class: "fin" }, h("td", {}, "After adjustments"), ...P.map((n) => cell(R.people[n].final)), h("td", { class: "billcol" }, fromPaise(sum(P.map((n) => R.people[n].final))))));
  return h("table", { class: "matrix", id: "matrix" }, h("thead", {}, h("tr", {}, h("th", {}, ""), ...P.map((n) => h("th", {}, markEl(n), n)), h("th", { class: "billcol" }, "Bill"))), h("tbody", {}, ...rows));
}
function step5(keep) {
  const y = window.scrollY, R = S.result;
  const main = $("#main");
  if (!R) {
    fill(main, h("h1", {}, "The split"), S.splitErr ? h("p", { class: "note bad", role: "alert" }, S.splitErr) : h("div", { class: "reading", role: "status" }, h("span", { class: "sweep" }), "Working it out"));
    setBar(null, null); if (!S.splitErr && !S.busySplit) { S.busySplit = true; runSplit().finally(() => { S.busySplit = false; }); } return;
  }
  const total = R.printed_total, sum = Object.values(R.people).reduce((a, p) => a + p.final, 0);
  const hasOv = Object.keys(S.overrides).length > 0;
  fill(main,
    h("h1", {}, "The split"),
    h("div", { class: "big-total" }, fromPaise(R.printed_total)),
    h("div", { class: "sharebar", role: "img", "aria-label": S.people.map((n) => `${n} ${fromPaise(R.people[n].final)}`).join(", ") },
      ...S.people.map((n) => h("span", { class: "seg", style: `flex:${Math.max(R.people[n].final, 1)};background:${markColor(n)}` }))),
    h("div", { class: "sharekey" }, ...S.people.map((n) => h("span", {}, h("i", { style: `background:${markColor(n)}` }), `${n} ${fromPaise(R.people[n].final)}`))),
    R.diff_vs_printed ? h("p", { class: "note bad" }, `Items and charges come to ${fromPaise(R.computed_total)}, the bill says ${fromPaise(R.printed_total)}.`) : null,
    h("h2", {}, "The bill"), h("div", { class: "ledger" }, ...Object.entries(S.summary || {}).map(([k, v]) => h("div", { class: "row" + (k === "Total" ? " total" : "") }, h("span", {}, k), h("span", { class: "dots" }), h("span", { class: "amt" }, fromPaise(v))))),
    h("h2", {}, "Who pays what"),
    h("div", { class: "scroll" }, matrixTable(R)),
    h("p", { class: "matrix-note" }, "Every tax, service charge and discount is shared by each person's pre-tax subtotal, not equally, so each line adds up to the Bill column."),
    h("h2", {}, "Each person"),
    ...S.people.map((n, i) => personCard(n, i)),
    hasOv && R.unreconciled ? h("p", { class: "note bad", role: "status" }, `${fromPaise(Math.abs(R.unreconciled))} of the bill is ${R.unreconciled > 0 ? "not covered" : "over-covered"} after your adjustments.`) : null,
    hasOv && h("label", { class: "check" }, h("input", { type: "checkbox", id: "rebalance", checked: S.rebalance, onchange: (e) => { S.rebalance = e.target.checked; runSplit(); } }), "Spread the difference over the people who weren't adjusted"),
    h("div", { class: "actions" }, h("button", { class: "text", type: "button", onclick: () => go(4) }, "Change who had what"), h("button", { class: "text", type: "button", onclick: () => go(3) }, "Change the people"),
      h("button", { class: "text", type: "button", id: "json", onclick: downloadJson }, "Bill as JSON")));
  setBar(h("div", {}, h("span", { class: "tot" }, fromPaise(sum)), h("span", { class: "sub" }, sum === total ? "Adds up to the bill" : "Differs from the bill")),
    h("button", { class: "btn", type: "button", id: "copy", onclick: copySummary }, "Copy summary"));
  renderSteps();
  if (keep) window.scrollTo(0, y);
}

async function downloadJson() {
  try {
    const data = await api("/export", { bill: cleanBill() });
    const a = h("a", { href: URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })), download: "bill.json" });
    document.body.append(a); a.click(); a.remove();
  } catch (e) { S.splitErr = e.message; step5(true); }
}
async function copySummary() {
  const R = S.result; if (!R) return;
  const text = ["SplitSnap", ...S.people.map((n) => `${n}: ${fromPaise(R.people[n].final)}${R.people[n].manual ? " (adjusted by hand)" : ""}`), `Total: ${fromPaise(R.printed_total)}`, "", ...S.expl.slice(1)].join("\n");
  try { await navigator.clipboard.writeText(text); } catch { const t = h("textarea", { style: "position:fixed;opacity:0" }, text); document.body.append(t); t.select(); try { document.execCommand("copy"); } catch { /* ignore */ } t.remove(); }
  const b = $("#copy"); if (b) { b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy summary"; }, 1500); }
}

/* ---------- start ---------- */
api("/health").then((r) => { S.threshold = r.conf_threshold; S.demoServer = r.demo; }).catch(() => {}).finally(() => { render(); loadExamples(); });
