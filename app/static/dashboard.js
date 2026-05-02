/* EZZAYRA olive anomaly dashboard.
 * Loads /api/parcels, renders colored polygons on a Leaflet map, supports filters,
 * a click-to-detail right panel, and a "Tester une nouvelle oliveraie" Leaflet.draw flow
 * that POSTs the drawn polygon to /api/diagnostic-anomalie.
 */
(function () {
  "use strict";

  const STATUS_COLORS = {
    vert: "#3fb950",
    orange: "#db7c1f",
    rouge: "#f85149",
    null: "#6e7681",
    undefined: "#6e7681",
  };

  let map;
  let parcelsLayer;
  let drawnLayer;
  let drawControl;
  let allParcels = [];
  let parcelLayers = new Map(); // id -> Leaflet layer
  let chart;

  function dom(id) { return document.getElementById(id); }

  function ensureMap() {
    if (map) return;
    map = L.map("map", { zoomControl: true }).setView([35.5, 10.0], 7);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap",
      maxZoom: 19,
    }).addTo(map);
    parcelsLayer = L.featureGroup().addTo(map);
    drawnLayer = L.featureGroup().addTo(map);
  }

  function styleForStatus(statut) {
    const color = STATUS_COLORS[statut] || STATUS_COLORS.null;
    return {
      color,
      weight: 2,
      fillColor: color,
      fillOpacity: 0.35,
    };
  }

  function popupHtml(p) {
    const status = p.statut || "non diagnostiqué";
    const score = p.anomaly_score !== null && p.anomaly_score !== undefined
      ? p.anomaly_score.toFixed(2)
      : "—";
    return `
      <div class="popup-id">${p.name || p.id}</div>
      <div class="popup-meta">
        ${p.system} · ${p.gouvernorat || "?"} · ${(p.area_ha || 0).toFixed(0)} ha
      </div>
      <div>Statut : <strong style="color:${STATUS_COLORS[p.statut] || "#6e7681"}">${status}</strong></div>
      <div>Score : ${score}</div>
    `;
  }

  function renderParcels() {
    parcelsLayer.clearLayers();
    parcelLayers.clear();

    const gouv = dom("filter-gouv").value;
    const sys = dom("filter-system").value;
    const stat = dom("filter-status").value;
    const q = dom("filter-id").value.toLowerCase().trim();

    const counts = { vert: 0, orange: 0, rouge: 0, total: 0 };

    for (const p of allParcels) {
      if (gouv && p.gouvernorat !== gouv) continue;
      if (sys && p.system !== sys) continue;
      if (stat && p.statut !== stat) continue;
      if (q && !(p.id.toLowerCase().includes(q) || (p.name || "").toLowerCase().includes(q))) continue;

      const layer = L.geoJSON(p.polygon_geojson, { style: styleForStatus(p.statut) });
      layer.bindPopup(popupHtml(p));
      layer.on("click", () => showDetail(p));
      layer.addTo(parcelsLayer);
      parcelLayers.set(p.id, layer);

      counts.total++;
      if (p.statut === "vert") counts.vert++;
      else if (p.statut === "orange") counts.orange++;
      else if (p.statut === "rouge") counts.rouge++;
    }

    dom("counts").innerHTML = `
      <span class="badge badge--vert">vert: <strong>${counts.vert}</strong></span>
      <span class="badge badge--orange">orange: <strong>${counts.orange}</strong></span>
      <span class="badge badge--rouge">rouge: <strong>${counts.rouge}</strong></span>
      <span class="badge badge--gray">total: <strong>${counts.total}</strong></span>
    `;

    if (parcelsLayer.getLayers().length) {
      try { map.fitBounds(parcelsLayer.getBounds(), { padding: [20, 20] }); } catch (_) { /* empty */ }
    }
  }

  function renderChart(observe, attendu) {
    if (!observe || !observe.length) return;
    const ctx = dom("ndvi-chart").getContext("2d");
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: observe.map((_, i) => `S-${observe.length - i}`),
        datasets: [
          {
            label: "NDVI observé",
            data: observe,
            borderColor: "#58a6ff",
            backgroundColor: "rgba(88, 166, 255, 0.2)",
            tension: 0.25,
          },
          {
            label: "NDVI attendu",
            data: attendu,
            borderColor: "#db7c1f",
            borderDash: [4, 4],
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: false,
        plugins: {
          legend: { labels: { color: "#e6edf3", boxWidth: 12, font: { size: 11 } } },
        },
        scales: {
          x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
          y: {
            ticks: { color: "#8b949e" },
            grid: { color: "#30363d" },
            suggestedMin: 0.1,
            suggestedMax: 0.7,
          },
        },
      },
    });
  }

  async function showDetail(p) {
    dom("detail-empty").style.display = "none";
    dom("detail-content").style.display = "block";
    dom("detail-title").textContent = p.name || p.id;
    dom("detail-system").textContent = p.system;
    dom("detail-gouv").textContent = p.gouvernorat || "?";
    dom("detail-area").textContent = `${(p.area_ha || 0).toFixed(0)} ha`;

    let payload = null;
    if (p.statut) {
      try {
        const r = await fetch(`/api/diagnostic/cached/${encodeURIComponent(p.id)}`);
        if (r.ok) payload = await r.json();
      } catch (e) { /* ignore */ }
    }

    if (!payload) {
      dom("detail-badge").textContent = "non diagnostiqué";
      dom("detail-badge").className = "badge badge--gray";
      dom("detail-score").textContent = "—";
      dom("detail-explication").textContent = "Aucun diagnostic en cache. Lancez scripts/refresh_all.py.";
      dom("detail-reco").textContent = "—";
      dom("detail-raw").textContent = "—";
      return;
    }

    dom("detail-badge").textContent = payload.statut;
    dom("detail-badge").className = `badge badge--${payload.statut}`;
    dom("detail-score").textContent = payload.anomaly_score?.toFixed?.(2) ?? "—";
    dom("detail-explication").textContent = payload.explication || "—";
    dom("detail-reco").textContent = payload.recommandation || "—";
    dom("detail-raw").textContent = JSON.stringify(payload, null, 2);
    renderChart(payload.ndvi_observe, payload.ndvi_attendu);
  }

  function populateGouvernoratFilter() {
    const sel = dom("filter-gouv");
    const seen = new Set();
    for (const p of allParcels) {
      if (p.gouvernorat && !seen.has(p.gouvernorat)) {
        seen.add(p.gouvernorat);
        const opt = document.createElement("option");
        opt.value = p.gouvernorat;
        opt.textContent = p.gouvernorat;
        sel.appendChild(opt);
      }
    }
  }

  function attachFilters() {
    ["filter-gouv", "filter-system", "filter-status"].forEach((id) =>
      dom(id).addEventListener("change", renderParcels)
    );
    dom("filter-id").addEventListener("input", renderParcels);
  }

  function showLoading(text) {
    dom("loading-text").textContent = text || "Diagnostic en cours...";
    dom("loading").style.display = "flex";
  }
  function hideLoading() { dom("loading").style.display = "none"; }

  function setupDraw() {
    drawControl = new L.Control.Draw({
      draw: {
        polygon: { allowIntersection: false, showArea: true, shapeOptions: { color: "#4cc38a" } },
        marker: false, circle: false, rectangle: false, polyline: false, circlemarker: false,
      },
      edit: { featureGroup: drawnLayer, edit: false, remove: true },
    });

    dom("btn-draw").addEventListener("click", () => {
      drawnLayer.clearLayers();
      map.addControl(drawControl);
      const handler = new L.Draw.Polygon(map, drawControl.options.draw.polygon);
      handler.enable();
    });

    map.on(L.Draw.Event.CREATED, async (e) => {
      drawnLayer.addLayer(e.layer);
      map.removeControl(drawControl);

      const gj = e.layer.toGeoJSON();
      const newId = `JURY_${Date.now()}`;
      const today = new Date().toISOString().slice(0, 10);

      showLoading("Diagnostic de la nouvelle oliveraie...");
      let payload;
      try {
        const r = await fetch("/api/diagnostic-anomalie", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            oliveraie: {
              id: newId,
              polygone: gj.geometry,
              systeme: dom("filter-system").value || "intensif",
            },
            date: today,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        payload = await r.json();
      } catch (err) {
        hideLoading();
        alert("Diagnostic échoué : " + err.message);
        return;
      }
      hideLoading();

      // Style the drawn polygon by status.
      e.layer.setStyle({
        color: STATUS_COLORS[payload.statut] || "#4cc38a",
        fillColor: STATUS_COLORS[payload.statut] || "#4cc38a",
        fillOpacity: 0.35,
      });
      e.layer.bindPopup(`
        <div class="popup-id">${newId}</div>
        <div class="popup-meta">JURY · diagnostic live</div>
        <div>Statut : <strong style="color:${STATUS_COLORS[payload.statut]}">${payload.statut}</strong></div>
        <div>Score : ${payload.anomaly_score}</div>
      `).openPopup();

      // Show the diagnostic in the right panel.
      dom("detail-empty").style.display = "none";
      dom("detail-content").style.display = "block";
      dom("detail-title").textContent = `JURY · ${newId.slice(0, 24)}`;
      dom("detail-system").textContent = dom("filter-system").value || "?";
      dom("detail-gouv").textContent = "—";
      dom("detail-area").textContent = "—";
      dom("detail-badge").textContent = payload.statut;
      dom("detail-badge").className = `badge badge--${payload.statut}`;
      dom("detail-score").textContent = payload.anomaly_score?.toFixed?.(2) ?? "—";
      dom("detail-explication").textContent = payload.explication || "—";
      dom("detail-reco").textContent = payload.recommandation || "—";
      dom("detail-raw").textContent = JSON.stringify(payload, null, 2);
      renderChart(payload.ndvi_observe, payload.ndvi_attendu);
    });
  }

  async function loadParcels() {
    const r = await fetch("/api/parcels");
    if (!r.ok) {
      alert("Erreur de chargement des parcelles : HTTP " + r.status);
      return;
    }
    allParcels = await r.json();
    populateGouvernoratFilter();
    renderParcels();
  }

  document.addEventListener("DOMContentLoaded", async () => {
    ensureMap();
    attachFilters();
    setupDraw();
    await loadParcels();
  });
})();
