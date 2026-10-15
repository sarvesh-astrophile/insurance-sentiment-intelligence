"use strict";
const $ = (id) => document.getElementById(id);
let rows = [], revision = "", token = "", currentId = null, dirty = false, busy = false;
const radios = [...document.querySelectorAll('input[name="sentiment"]')];
const isApproved = (row) => row.human_approved.toLowerCase() === "true";
function visible() {
  return rows.filter(row => $("filter").value === "all" ||
    ($("filter").value === "approved" ? isApproved(row) : !isApproved(row)));
}
function message(text, error = false) {
  $("message").textContent = text;
  $("message").className = error ? "error" : "success";
}
function confirmDiscard() {
  return !dirty || window.confirm("Discard unsaved edits to this review?");
}
function render() {
  const list = visible();
  let row = list.find(r => r.review_id === currentId) || list[0];
  currentId = row ? row.review_id : null;
  const count = rows.filter(isApproved).length;
  const common = rows.filter(r => isApproved(r) && r.human_sentiment !== "mixed_demo").length;
  $("progress").textContent = `${count} / ${rows.length} approved · ${common} three-class gold labels · aim for at least 40`;
  $("bar").max = rows.length;
  $("bar").value = count;
  $("card").hidden = !row;
  $("empty").hidden = !!row;
  dirty = false;
  if (!row) return;
  const index = list.indexOf(row);
  $("position").textContent = `Review ${index + 1} of ${list.length}`;
  $("state").textContent = isApproved(row) ? `Approved by ${row.reviewer}` : "Awaiting review";
  $("review-id").textContent = row.review_id;
  $("text").textContent = row.text;
  $("draft").textContent = `AI draft: ${row.draft_sentiment}`;
  $("note").textContent = row.annotation_notes;
  document.querySelector("details").open = false;
  // Never preselect an AI draft: the human makes an explicit independent choice.
  radios.forEach(radio => { radio.checked = radio.value === row.human_sentiment; });
  $("approved").checked = isApproved(row);
  $("previous").disabled = index === 0;
  $("next").disabled = index === list.length - 1;
}
async function load() {
  if (busy || !confirmDiscard()) return;
  busy = true;
  try {
    const response = await fetch("/api/reviews", {cache: "no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    rows = data.reviews; revision = data.revision; token = data.token;
    render(); message("Loaded saved annotations.");
  } catch (error) { message(error.message, true); }
  finally { busy = false; }
}
function move(direction) {
  if (busy || !confirmDiscard()) return;
  const list = visible(), index = list.findIndex(row => row.review_id === currentId);
  const next = list[index + direction];
  if (next) { currentId = next.review_id; render(); message(""); }
}
async function save(advance) {
  if (busy || !currentId) return;
  const label = radios.find(radio => radio.checked)?.value;
  const reviewer = $("reviewer").value.trim();
  if (!reviewer || !label) return message("Enter your initials/name and select a label first.", true);
  const list = visible(), index = list.findIndex(row => row.review_id === currentId);
  const nextId = list[index + 1]?.review_id;
  const payload = {revision, review_id: currentId, human_sentiment: label,
    reviewer, human_approved: $("approved").checked};
  busy = true;
  // Prevent edits being lost while the save request is in flight.
  const controls = [...document.querySelectorAll("input, select, button")];
  controls.forEach(control => { control.disabled = true; });
  try {
    const response = await fetch("/api/reviews", {method: "POST",
      headers: {"Content-Type": "application/json", "X-Review-Token": token},
      body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    rows = data.reviews; revision = data.revision;
    if (advance && nextId) currentId = nextId;
    controls.forEach(control => { control.disabled = false; });
    render();
    message(payload.human_approved ? "Approved label saved to CSV (backup created)." : "Saved as unapproved; excluded from gold metrics.");
  } catch (error) {
    controls.forEach(control => { control.disabled = false; });
    message(error.message, true);
  } finally { busy = false; }
}
radios.forEach(radio => radio.addEventListener("change", () => { dirty = true; $("approved").checked = false; }));
$("approved").addEventListener("change", () => { dirty = true; });
$("reviewer").addEventListener("input", () => { dirty = true; });
let oldFilter = "all";
$("filter").addEventListener("change", () => {
  if (!confirmDiscard()) { $("filter").value = oldFilter; return; }
  oldFilter = $("filter").value; render();
});
$("previous").addEventListener("click", () => move(-1));
$("next").addEventListener("click", () => move(1));
$("save").addEventListener("click", () => save(false));
$("save-next").addEventListener("click", () => save(true));
$("reload").addEventListener("click", load);
window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
load();
