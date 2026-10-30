(() => {
  "use strict";

  const form = document.getElementById("run-form");
  const errorBox = document.getElementById("form-error");
  const jobSelect = document.getElementById("job-select");
  const cancelButton = document.getElementById("cancel-button");
  const stateBadge = document.getElementById("run-state");
  const runTitle = document.getElementById("run-title");
  const progressBar = document.getElementById("progress-bar");
  const progressLabel = document.getElementById("progress-label");
  const summaryStrip = document.getElementById("summary-strip");
  const runLog = document.getElementById("run-log");
  const mapImage = document.getElementById("map-image");
  const mapEmpty = document.getElementById("map-empty");
  const mapField = document.getElementById("map-field");
  const frameSlider = document.getElementById("frame-slider");
  const frameLabel = document.getElementById("frame-label");
  const playButton = document.getElementById("play-button");
  const earthImage = document.getElementById("earth-image");
  const earthEmpty = document.getElementById("earth-empty");

  let currentJob = "";
  let frames = [];
  let polling = null;
  let playing = null;
  let earthLoadedJob = "";
  let refreshing = false;

  function formPayload(smoke) {
    const data = Object.fromEntries(new FormData(form).entries());
    data.shock_flattening = form.elements.shock_flattening.checked;
    data.smoke = smoke;
    return data;
  }

  async function requestJson(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new Error(
        "Dashboard server is offline. Restart `python -m swcmep.ui` " +
        "in its own PowerShell window and leave that window open."
      );
    }
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
    return body;
  }

  async function startRun(smoke) {
    errorBox.textContent = "";
    try {
      const job = await requestJson("/api/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(formPayload(smoke))
      });
      selectJob(job.job_id);
      await loadJobs();
      jobSelect.value = currentJob;
      await refresh();
    } catch (error) {
      errorBox.textContent = error.message;
    }
  }

  form.addEventListener("submit", event => {
    event.preventDefault();
    startRun(false);
  });
  document.getElementById("smoke-button").addEventListener("click", () => startRun(true));

  async function loadJobs() {
    const jobs = await requestJson("/api/jobs");
    jobSelect.innerHTML = "";
    if (!jobs.length) {
      jobSelect.add(new Option("No runs yet", ""));
      return;
    }
    for (const job of jobs) {
      const label = `${job.event_name} — ${job.state}`;
      jobSelect.add(new Option(label, job.job_id));
    }
    if (!currentJob) currentJob = jobs[0].job_id;
    jobSelect.value = currentJob;
  }

  function selectJob(jobId) {
    currentJob = jobId;
    frames = [];
    frameSlider.value = "0";
    frameSlider.max = "0";
    frameSlider.disabled = true;
    playButton.disabled = true;
    earthLoadedJob = "";
    earthImage.classList.add("hidden");
    earthEmpty.classList.remove("hidden");
    stopPlayback();
    updateFrame();
  }

  jobSelect.addEventListener("change", async () => {
    selectJob(jobSelect.value);
    await refresh();
  });

  function updateStatus(job) {
    runTitle.textContent = job.event_name;
    stateBadge.textContent = job.state;
    stateBadge.className = `state ${job.state}`;
    progressBar.style.width = `${Math.round(job.progress * 100)}%`;
    progressLabel.textContent = `${job.completed_hours.toFixed(1)} / ${job.target_hours.toFixed(1)} simulated hours`;
    cancelButton.disabled = job.state !== "running";
    summaryStrip.innerHTML = "";
    if (job.summary) {
      const summary = job.summary;
      const items = [
        ["SWCMEP", summary.modeled_arrival_utc ||
          (job.state === "completed" ? "No impact detected" : "Run incomplete")],
        ["Peak speed", `${Math.round(summary.peak_earth.v_kms)} km/s`],
        ["Peak density", `${summary.peak_earth.n_cm3.toFixed(1)} cm⁻³`],
        ["Peak |B|", `${summary.peak_earth.b_nt.toFixed(1)} nT`]
      ];
      for (const [key, value] of items) {
        const chip = document.createElement("span");
        chip.textContent = `${key}: ${value}`;
        summaryStrip.appendChild(chip);
      }
      if (earthLoadedJob !== currentJob) {
        earthImage.src = `/api/jobs/${currentJob}/earth.png?v=${Date.now()}`;
        earthImage.classList.remove("hidden");
        earthEmpty.classList.add("hidden");
        earthLoadedJob = currentJob;
      }
    }
  }

  async function loadFrames(jobId) {
    const fresh = await requestJson(`/api/jobs/${jobId}/frames`);
    if (jobId !== currentJob) return;
    if (fresh.length !== frames.length) {
      const wasLatest = !frames.length || Number(frameSlider.value) >= frames.length - 1;
      frames = fresh;
      frameSlider.max = Math.max(frames.length - 1, 0);
      frameSlider.disabled = !frames.length;
      playButton.disabled = frames.length < 2;
      if (wasLatest && frames.length) frameSlider.value = String(frames.length - 1);
      updateFrame();
    }
  }

  function updateFrame() {
    if (!frames.length || !currentJob) {
      mapImage.classList.add("hidden");
      mapEmpty.classList.remove("hidden");
      frameLabel.textContent = "No frames";
      return;
    }
    const index = Number(frameSlider.value);
    const frame = frames[index];
    mapImage.src = `/api/jobs/${currentJob}/frame/${index}.png?field=${encodeURIComponent(mapField.value)}&v=${Date.now()}`;
    mapImage.classList.remove("hidden");
    mapEmpty.classList.add("hidden");
    frameLabel.textContent = `${frame.utc} · t=${Number(frame.t_hours).toFixed(1)} h`;
  }

  frameSlider.addEventListener("input", updateFrame);
  mapField.addEventListener("change", updateFrame);

  function stopPlayback() {
    if (playing) window.clearInterval(playing);
    playing = null;
    playButton.textContent = "Play";
  }

  playButton.addEventListener("click", () => {
    if (playing) {
      stopPlayback();
      return;
    }
    playButton.textContent = "Pause";
    playing = window.setInterval(() => {
      let next = Number(frameSlider.value) + 1;
      if (next >= frames.length) next = 0;
      frameSlider.value = String(next);
      updateFrame();
    }, 850);
  });

  cancelButton.addEventListener("click", async () => {
    if (!currentJob) return;
    try {
      await requestJson(`/api/jobs/${currentJob}/cancel`, {method: "POST"});
      await refresh();
    } catch (error) {
      errorBox.textContent = error.message;
    }
  });

  async function refresh() {
    if (!currentJob || refreshing) return;
    const jobId = currentJob;
    refreshing = true;
    try {
      const [job, log] = await Promise.all([
        requestJson(`/api/jobs/${jobId}`),
        fetch(`/api/jobs/${jobId}/log`).then(response => response.text())
      ]);
      if (jobId !== currentJob) return;
      updateStatus(job);
      runLog.textContent = log || "Waiting for solver output…";
      runLog.scrollTop = runLog.scrollHeight;
      await loadFrames(jobId);
      if (job.state !== "running") await loadJobs();
    } catch (error) {
      if (jobId === currentJob) runLog.textContent = error.message;
    } finally {
      refreshing = false;
    }
  }

  async function initialize() {
    try {
      await loadJobs();
      if (currentJob) await refresh();
    } catch (error) {
      errorBox.textContent = error.message;
    }
    polling = window.setInterval(refresh, 2500);
  }

  window.addEventListener("beforeunload", () => {
    if (polling) window.clearInterval(polling);
    stopPlayback();
  });
  initialize();
})();
