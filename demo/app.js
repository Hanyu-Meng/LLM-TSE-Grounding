const tableData = {
  clean: {
    intro: "Held-out frozen TEST · 6,000 trials. Lower is better for WER and switch rates; higher is better for speaker margin and P.808.",
    interpretation: "Pool D direct is the content-fidelity endpoint. On the same repaired evidence, CSG improves Q-Full UD and nearly eliminates acoustic speaker drift, but it does not replace the direct endpoint.",
    rows: [
      ["Primary WeSep", "baseline", "13.68", "7.00", "6.98", ".449", "3.688 P.808", "—"],
      ["Pool D direct", "repair endpoint", "6.54", ".88", ".78", ".506", "3.699 P.808", "—", "best"],
      ["Pool D → Q-Full UD", "ungrounded", "14.08", "1.00", ".43", ".422", "3.705 P.808", "—", "grounded"],
      ["Pool D → fixed CSG", "grounded", "12.74", "1.00", ".23", ".422", "3.699 P.808", "—", "grounded"],
      ["Pool D → selected GNR", "local refinement", "pending", "—", "—", "—", "—", "—", "grounded"],
    ],
  },
  noisyDev: {
    intro: "Frozen Noisy DEV · 8,400 trials (6,000 natural + 2,400 controlled). Used for selection and ablation; never relabelled as TEST.",
    interpretation: "DEV is intentionally preserved. Pool D direct is the fidelity endpoint; fixed CSG is the selected generative reliability endpoint. Adaptive CSG and GNR are negative ablations.",
    rows: [
      ["Primary WeSep", "baseline", "53.50", "20.11", "19.54", ".269", "2.243 OVRL", "1.997"],
      ["Pool D direct", "repair endpoint", "46.75", "11.86", "10.93", ".332", "2.132 OVRL", "1.930", "best"],
      ["Pool D → Q-Full UD", "ungrounded", "66.87", "13.68", "2.99", ".362", "3.063 OVRL", "3.164", "grounded"],
      ["Pool D → fixed CSG", "selected grounded", "62.66", "13.44", "1.27", ".370", "3.066 OVRL", "3.047", "grounded"],
      ["Pool D → adaptive CSG", "negative ablation", "62.92", "13.42", "1.64", ".369", "3.079 OVRL", "3.104", "grounded"],
      ["Pool D → GNR K20/R2", "negative ablation", "65.96", "13.44", "1.67", ".369", "3.081 OVRL", "3.118", "grounded"],
    ],
  },
  noisyTest: {
    intro: "Frozen Noisy TEST · 8,400 trials · one permitted run is in progress. Partial outputs are never inserted here.",
    interpretation: "This block remains blank until the ledger is COMPLETE and the final report passes validation. DEV evidence remains available in its own tab.",
    rows: [
      ["Primary WeSep", "baseline", "—", "—", "—", "—", "—", "—"],
      ["Pool D direct", "repair endpoint", "—", "—", "—", "—", "—", "—", "best"],
      ["Pool D → Q-Full UD", "ungrounded", "—", "—", "—", "—", "—", "—", "grounded"],
      ["Pool D → fixed CSG", "selected grounded", "—", "—", "—", "—", "—", "—", "grounded"],
      ["Pool D → adaptive CSG", "negative ablation", "—", "—", "—", "—", "—", "—", "grounded"],
      ["Pool D → GNR K20/R2", "negative ablation", "—", "—", "—", "—", "—", "—", "grounded"],
    ],
  },
};

if (window.FROZEN_NOISY_TEST) {
  tableData.noisyTest = window.FROZEN_NOISY_TEST.table;
  const statusItems = document.querySelectorAll(".status");
  statusItems[1]?.classList.remove("active");
  statusItems[1]?.classList.add("done");
  statusItems[1]?.querySelector("p")?.replaceChildren("DEV selection frozen");
  statusItems[2]?.classList.remove("locked");
  statusItems[2]?.classList.add("done");
  statusItems[2]?.querySelector("p")?.replaceChildren("One-run replication complete");
  const statusNote = document.querySelector(".status-note");
  if (statusNote) statusNote.textContent = `Frozen TEST · ${window.FROZEN_NOISY_TEST.generatedAt}`;
  const badge = document.querySelector(".dev-badge");
  if (badge) badge.textContent = "FROZEN TEST";
  const snrSubtitle = document.querySelector("#snr-subtitle");
  if (snrSubtitle) snrSubtitle.textContent = "Frozen TEST replication, 480 paired trials at each SNR.";
}

const tbody = document.querySelector("#result-table tbody");
const intro = document.querySelector("#table-intro");
const interpretation = document.querySelector("#table-interpretation");
const tableButtons = document.querySelectorAll("[data-table]");

function renderTable(key) {
  const data = tableData[key];
  tbody.innerHTML = data.rows.map((row) => {
    const marker = row[8] || "";
    const cells = row.slice(0, 8).map((value, index) => {
      if (index === 1) return `<td><span class="role-pill">${value}</span></td>`;
      return `<td>${value}</td>`;
    }).join("");
    return `<tr class="${marker}">${cells}</tr>`;
  }).join("");
  intro.textContent = data.intro;
  interpretation.textContent = data.interpretation;
  tableButtons.forEach((button) => button.classList.toggle("active", button.dataset.table === key));
}

tableButtons.forEach((button) => button.addEventListener("click", () => renderTable(button.dataset.table)));
renderTable("noisyDev");

const snrData = window.FROZEN_NOISY_TEST?.snr || [
  { snr: -5, selected: 0.0, oracle: 4.0 },
  { snr: 0, selected: 6.10, oracle: 19.51 },
  { snr: 5, selected: 34.91, oracle: 48.11 },
  { snr: 10, selected: 71.30, oracle: 76.85 },
  { snr: 15, selected: 87.04, oracle: 91.67 },
];

function drawChart() {
  const canvas = document.querySelector("#snr-chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(620, rect.width);
  const height = width * 0.47;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 54, right: 24, top: 26, bottom: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const x = (index) => pad.left + (innerW * index) / (snrData.length - 1);
  const y = (value) => pad.top + innerH * (1 - value / 100);

  ctx.font = "11px 'DM Mono', monospace";
  ctx.textAlign = "right";
  ctx.fillStyle = "#7b8580";
  ctx.strokeStyle = "#e3e3dc";
  ctx.lineWidth = 1;
  [0, 25, 50, 75, 100].forEach((tick) => {
    ctx.beginPath(); ctx.moveTo(pad.left, y(tick)); ctx.lineTo(width - pad.right, y(tick)); ctx.stroke();
    ctx.fillText(`${tick}%`, pad.left - 10, y(tick) + 4);
  });
  ctx.textAlign = "center";
  snrData.forEach((point, index) => ctx.fillText(`${point.snr} dB`, x(index), height - 14));

  const line = (field, color) => {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 3;
    ctx.beginPath();
    snrData.forEach((point, index) => index ? ctx.lineTo(x(index), y(point[field])) : ctx.moveTo(x(index), y(point[field])));
    ctx.stroke();
    snrData.forEach((point, index) => {
      ctx.beginPath(); ctx.arc(x(index), y(point[field]), 5, 0, Math.PI * 2); ctx.fill();
      ctx.font = "10px 'DM Mono', monospace"; ctx.fillText(`${point[field].toFixed(1)}%`, x(index), y(point[field]) - 12);
    });
  };
  line("oracle", "#c99a35");
  line("selected", "#0d746c");
}

let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawChart, 120); });
drawChart();

document.querySelectorAll("audio").forEach((player) => {
  player.addEventListener("play", () => {
    document.querySelectorAll("audio").forEach((other) => { if (other !== player) other.pause(); });
  });
});
