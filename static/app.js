document.addEventListener("DOMContentLoaded", () => {
    // Application State
    let selectedFile = null;
    let selectedSamplePath = null;
    let sampleList = [];
    let currentEnergyThreshold = 0.50;
    let currentXaiData = null;
    let activeXaiLayer = "fused";
    let currentOpacity = 0.50;

    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzoneEmpty = document.getElementById("dropzoneEmpty");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeImgBtn = document.getElementById("removeImgBtn");
    const scanOverlay = document.getElementById("scanOverlay");
    const analyzeBtn = document.getElementById("analyzeBtn");

    // XAI & Modality Controls
    const xaiControlsCard = document.getElementById("xaiControlsCard");
    const xaiPills = document.getElementById("xaiPills");
    const xaiActiveBadge = document.getElementById("xaiActiveBadge");
    const opacitySlider = document.getElementById("opacitySlider");
    const opacityVal = document.getElementById("opacityVal");
    const cnnWeightText = document.getElementById("cnnWeightText");
    const vitWeightText = document.getElementById("vitWeightText");
    const modalityCnnBar = document.getElementById("modalityCnnBar");
    const modalityVitBar = document.getElementById("modalityVitBar");

    // Alerts & Diagnostics
    const globalAlertBanner = document.getElementById("globalAlertBanner");
    const globalAlertTitle = document.getElementById("globalAlertTitle");
    const globalAlertMessage = document.getElementById("globalAlertMessage");
    const closeAlertBannerBtn = document.getElementById("closeAlertBannerBtn");

    const oodAlertWrapper = document.getElementById("oodAlertWrapper");
    const oodTitle = document.getElementById("oodTitle");
    const diagStage = document.getElementById("diagStage");
    const diagEnergy = document.getElementById("diagEnergy");
    const diagSim = document.getElementById("diagSim");
    const diagReason = document.getElementById("diagReason");
    const oodTryAgainBtn = document.getElementById("oodTryAgainBtn");

    // Dataset Browser
    const sampleGrid = document.getElementById("sampleGrid");
    const sampleCount = document.getElementById("sampleCount");
    const classChips = document.getElementById("classChips");

    // Result Containers
    const emptyState = document.getElementById("emptyState");
    const analyzingState = document.getElementById("analyzingState");
    const diagnosisWrapper = document.getElementById("diagnosisWrapper");

    // Diagnostics Fields
    const deviceText = document.getElementById("deviceText");
    const resSeverity = document.getElementById("resSeverity");
    const resTitle = document.getElementById("resTitle");
    const resPathogen = document.getElementById("resPathogen");
    const resConfidence = document.getElementById("resConfidence");
    const resConfidenceCircle = document.getElementById("resConfidenceCircle");
    const resSummary = document.getElementById("resSummary");
    const resEnergyScore = document.getElementById("resEnergyScore");
    const resSimScore = document.getElementById("resSimScore");
    const disambiguationBox = document.getElementById("disambiguationBox");
    const disambiguationText = document.getElementById("disambiguationText");
    const probList = document.getElementById("probList");

    const symptomsList = document.getElementById("symptomsList");
    const organicList = document.getElementById("organicList");
    const chemicalList = document.getElementById("chemicalList");
    const preventionList = document.getElementById("preventionList");

    const archModal = document.getElementById("archModal");
    const openArchModalBtn = document.getElementById("openArchModalBtn");
    const closeArchModalBtn = document.getElementById("closeArchModalBtn");

    // Initialize System Status
    fetchHealthStatus();
    loadDatasetSamples();
    setupAlertBanner();
    setupXaiControls();

    // 0. Notification Banner Controls
    function setupAlertBanner() {
        if (closeAlertBannerBtn) {
            closeAlertBannerBtn.addEventListener("click", hideNotificationBanner);
        }
        if (oodTryAgainBtn) {
            oodTryAgainBtn.addEventListener("click", () => {
                resetInput();
                fileInput.click();
            });
        }
    }

    function showNotificationBanner(message, type = "danger") {
        if (!globalAlertBanner) return;
        if (globalAlertTitle) globalAlertTitle.textContent = "Invalid Input: ";
        if (globalAlertMessage) globalAlertMessage.textContent = message;
        globalAlertBanner.className = `alert-banner alert-${type}`;
        globalAlertBanner.classList.remove("hidden");
    }

    function hideNotificationBanner() {
        if (!globalAlertBanner) return;
        globalAlertBanner.classList.add("hidden");
    }

    // 1. Health Status
    async function fetchHealthStatus() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            if (data.cuda_available) {
                deviceText.innerHTML = `<span style="color: #10b981;">GPU (CUDA)</span>`;
            } else {
                deviceText.innerHTML = `<span>CPU Inference</span>`;
            }
            if (data.energy_threshold !== undefined) {
                currentEnergyThreshold = data.energy_threshold;
            }
        } catch (err) {
            console.error("Health check error:", err);
            deviceText.textContent = "Offline";
        }
    }

    // 2. Load Dataset Samples
    async function loadDatasetSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            sampleList = data.samples || [];
            sampleCount.textContent = `${sampleList.length} specimens available`;
            renderSamples("all");
        } catch (err) {
            console.error("Samples loading error:", err);
            sampleCount.textContent = "No dataset specimens found";
        }
    }

    function renderSamples(filterClass) {
        sampleGrid.innerHTML = "";
        const filtered = filterClass === "all"
            ? sampleList
            : sampleList.filter(s => s.class_name.toLowerCase() === filterClass.toLowerCase());

        filtered.forEach(sample => {
            const isOod = sample.is_ood || sample.class_name.includes("Non-Potato");
            const item = document.createElement("div");
            item.className = "sample-item" + (isOod ? " sample-item-ood" : "");
            item.dataset.path = sample.relative_path;
            item.dataset.class = sample.class_name;
            const displayTitle = sample.label || sample.class_name;
            item.title = `Click to test: ${displayTitle}`;

            item.innerHTML = `
                <img src="${sample.url}" alt="${displayTitle}" loading="lazy">
                <div class="sample-tag ${isOod ? 'ood-sample-tag' : ''}">${isOod ? '<i class="fa-solid fa-shield-halved"></i> ' : ''}${displayTitle}</div>
            `;

            item.addEventListener("click", () => {
                selectSampleImage(sample);
            });

            sampleGrid.appendChild(item);
        });
    }

    function selectSampleImage(sample) {
        selectedFile = null;
        selectedSamplePath = sample.relative_path;
        fileInput.value = "";
        currentXaiData = null;

        imagePreview.src = sample.url;
        dropzoneEmpty.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        if (xaiControlsCard) xaiControlsCard.classList.add("hidden");
        analyzeBtn.disabled = false;

        // Auto trigger diagnostic run for snappy demonstration
        triggerDiagnosis();
    }

    // 3. Filter Chips
    classChips.addEventListener("click", (e) => {
        const chip = e.target.closest(".chip");
        if (chip) {
            classChips.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            renderSamples(chip.dataset.class);
        }
    });

    // 4. File Drag & Drop
    dropzone.addEventListener("click", (e) => {
        if (!previewContainer.classList.contains("hidden") && e.target === removeImgBtn) return;
        if (previewContainer.classList.contains("hidden")) {
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, WEBP)");
            return;
        }

        selectedFile = file;
        selectedSamplePath = null;
        currentXaiData = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneEmpty.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            if (xaiControlsCard) xaiControlsCard.classList.add("hidden");
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeImgBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetInput();
    });

    function resetInput() {
        selectedFile = null;
        selectedSamplePath = null;
        currentXaiData = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzoneEmpty.classList.remove("hidden");
        if (xaiControlsCard) xaiControlsCard.classList.add("hidden");
        analyzeBtn.disabled = true;
        scanOverlay.classList.add("hidden");
        hideNotificationBanner();
        showState("empty");
    }

    // 5. Trigger Diagnosis
    analyzeBtn.addEventListener("click", triggerDiagnosis);

    async function triggerDiagnosis() {
        if (!selectedFile && !selectedSamplePath) return;

        showState("analyzing");
        scanOverlay.classList.remove("hidden");
        analyzeBtn.disabled = true;
        hideNotificationBanner();

        const formData = new FormData();
        if (selectedFile) {
            formData.append("file", selectedFile);
        } else if (selectedSamplePath) {
            formData.append("sample_path", selectedSamplePath);
        }
        formData.append("energy_threshold", currentEnergyThreshold);
        formData.append("alpha", currentOpacity);

        try {
            const startTime = performance.now();
            const res = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Inference failed");
            }

            const data = await res.json();
            const elapsed = performance.now() - startTime;
            const remainingDelay = Math.max(350 - elapsed, 80);

            setTimeout(() => {
                scanOverlay.classList.add("hidden");
                analyzeBtn.disabled = false;

                // Out-of-Distribution (OOD) / Invalid Input Check
                if (data.is_ood || data.is_valid === false || data.status === "INVALID_INPUT") {
                    if (xaiControlsCard) xaiControlsCard.classList.add("hidden");
                    renderOodAlert(data);
                    showState("ood");
                    const alertMsg = data.error_message || "Invalid Input: The uploaded image is not recognized as a valid potato leaf specimen. Please upload a clear image of a potato leaf.";
                    showNotificationBanner(alertMsg, "danger");
                } else {
                    hideNotificationBanner();
                    renderResults(data);
                    renderXai(data);
                    showState("results");
                }
            }, remainingDelay);

        } catch (err) {
            console.error("Diagnosis error:", err);
            scanOverlay.classList.add("hidden");
            analyzeBtn.disabled = false;
            showNotificationBanner("Error analyzing specimen: " + err.message, "danger");
            showState("empty");
        }
    }

    // 6. Render Results for Valid Specimen
    function renderResults(data) {
        const details = data.details || {};
        const conf = data.confidence || 0;

        resTitle.textContent = details.title || data.predicted_class;
        resPathogen.textContent = `Pathogen: ${details.pathogen || 'N/A'}`;
        resSeverity.textContent = details.severity || 'Unknown';
        resSeverity.style.backgroundColor = details.severity_color || '#dc2626';

        if (resEnergyScore && data.energy_score !== undefined) {
            resEnergyScore.textContent = `${data.energy_score >= 0 ? "+" : ""}${Number(data.energy_score).toFixed(2)}`;
        }
        if (resSimScore && data.potato_similarity !== undefined) {
            resSimScore.textContent = `${(data.potato_similarity * 100).toFixed(1)}%`;
        }

        // Pest vs Fungi Disambiguation
        if (disambiguationBox && disambiguationText) {
            if (data.disambiguation_note) {
                disambiguationText.textContent = data.disambiguation_note;
                disambiguationBox.classList.remove("hidden");
            } else {
                disambiguationBox.classList.add("hidden");
            }
        }

        resConfidence.textContent = `${conf.toFixed(1)}%`;
        resConfidenceCircle.style.background = `conic-gradient(var(--primary) ${conf}%, rgba(255, 255, 255, 0.1) 0)`;
        resSummary.textContent = details.summary || "";

        // Probability distribution
        probList.innerHTML = "";
        (data.probabilities || []).forEach(p => {
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <div class="prob-name">${p.class}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: 0%;"></div>
                </div>
                <div class="prob-pct">${p.probability.toFixed(1)}%</div>
            `;
            probList.appendChild(row);

            setTimeout(() => {
                const bar = row.querySelector(".prob-bar");
                bar.style.width = `${Math.min(p.probability, 100)}%`;
                if (p.class === data.predicted_class) {
                    bar.style.background = "linear-gradient(90deg, #059669, #10b981)";
                } else {
                    bar.style.background = "linear-gradient(90deg, #475569, #64748b)";
                }
            }, 40);
        });

        populateList(symptomsList, details.symptoms);
        populateList(organicList, details.organic_remedies);
        populateList(chemicalList, details.chemical_remedies);
        populateList(preventionList, details.prevention);
    }

    // 6.1 Render Explainable AI (XAI) Suite
    function renderXai(data) {
        if (!data.xai) return;
        currentXaiData = data.xai;

        if (xaiControlsCard) {
            xaiControlsCard.classList.remove("hidden");
        }

        // Modality weights bar
        const weights = data.modality_weights || (data.xai && data.xai.modality_weights) || { cnn_pct: 50, vit_pct: 50 };
        if (cnnWeightText) cnnWeightText.textContent = `${weights.cnn_pct.toFixed(1)}%`;
        if (vitWeightText) vitWeightText.textContent = `${weights.vit_pct.toFixed(1)}%`;
        if (modalityCnnBar) modalityCnnBar.style.width = `${weights.cnn_pct}%`;
        if (modalityVitBar) modalityVitBar.style.width = `${weights.vit_pct}%`;

        // Default active view to Fused XAI
        setXaiLayer("fused");
    }

    function setupXaiControls() {
        if (xaiPills) {
            xaiPills.addEventListener("click", (e) => {
                const pill = e.target.closest(".xai-pill");
                if (pill && pill.dataset.layer) {
                    setXaiLayer(pill.dataset.layer);
                }
            });
        }

        if (opacitySlider) {
            opacitySlider.addEventListener("input", (e) => {
                const val = parseInt(e.target.value, 10);
                currentOpacity = val / 100.0;
                if (opacityVal) opacityVal.textContent = `${val}%`;
                // Debounce opacity update if needed
                updateXaiOpacityDebounced();
            });
        }
    }

    let opacityTimer = null;
    function updateXaiOpacityDebounced() {
        if (!selectedSamplePath || !currentXaiData) return;
        clearTimeout(opacityTimer);
        opacityTimer = setTimeout(async () => {
            try {
                const res = await fetch("/api/xai", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        sample_path: selectedSamplePath,
                        alpha: currentOpacity,
                    })
                });
                if (res.ok) {
                    const xai = await res.json();
                    currentXaiData = xai;
                    setXaiLayer(activeXaiLayer);
                }
            } catch (err) {
                console.error("XAI Opacity update error:", err);
            }
        }, 200);
    }

    function setXaiLayer(layer) {
        activeXaiLayer = layer;
        if (xaiPills) {
            xaiPills.querySelectorAll(".xai-pill").forEach(p => {
                p.classList.toggle("active", p.dataset.layer === layer);
            });
        }

        const layerLabels = {
            "original": "Original RGB",
            "gradcam": "CNN Grad-CAM",
            "rollout": "ViT Rollout",
            "fused": "Fused XAI"
        };
        if (xaiActiveBadge) {
            xaiActiveBadge.textContent = layerLabels[layer] || layer;
        }

        if (currentXaiData) {
            let targetSrc = null;
            if (layer === "original") targetSrc = currentXaiData.original_b64;
            else if (layer === "gradcam") targetSrc = currentXaiData.gradcam_b64;
            else if (layer === "rollout") targetSrc = currentXaiData.rollout_b64;
            else if (layer === "fused") targetSrc = currentXaiData.fused_b64;

            if (targetSrc && imagePreview) {
                imagePreview.src = targetSrc;
            }
        }
    }

    // 6.2 Render Invalid Input Alert State
    function renderOodAlert(data) {
        if (oodTitle) {
            oodTitle.textContent = data.error_message || "Invalid Input: The uploaded image is not recognized as a valid potato leaf specimen. Please upload a clear image of a potato leaf.";
        }
        if (diagStage) {
            diagStage.textContent = data.rejection_stage || "Stage 1 Gatekeeper";
        }
        if (diagEnergy) {
            const e = data.energy_score !== undefined ? data.energy_score : (data.diagnostic && data.diagnostic.energy_score);
            const thresh = data.energy_threshold || (data.diagnostic && data.diagnostic.energy_threshold) || 0.50;
            diagEnergy.textContent = (e !== null && e !== undefined) ? `${e >= 0 ? "+" : ""}${Number(e).toFixed(2)} (Max: +${Number(thresh).toFixed(2)})` : "Failed Gate";
        }
        if (diagSim) {
            const sim = data.potato_similarity !== undefined ? data.potato_similarity : (data.diagnostic && data.diagnostic.potato_similarity);
            diagSim.textContent = (sim !== null && sim !== undefined) ? `${Number(sim).toFixed(2)} (Min: 0.48)` : "N/A";
        }
        if (diagReason) {
            diagReason.textContent = data.rejection_reason || "Non-Potato / OOD Specimen Rejected";
        }
    }

    function populateList(ulElement, items) {
        ulElement.innerHTML = "";
        if (!items || items.length === 0) {
            ulElement.innerHTML = "<li>No specific recommendations recorded.</li>";
            return;
        }
        items.forEach(text => {
            const li = document.createElement("li");
            li.textContent = text;
            ulElement.appendChild(li);
        });
    }

    function showState(state) {
        emptyState.classList.add("hidden");
        analyzingState.classList.add("hidden");
        if (oodAlertWrapper) oodAlertWrapper.classList.add("hidden");
        diagnosisWrapper.classList.add("hidden");

        if (state === "empty") emptyState.classList.remove("hidden");
        else if (state === "analyzing") analyzingState.classList.remove("hidden");
        else if (state === "ood" && oodAlertWrapper) oodAlertWrapper.classList.remove("hidden");
        else if (state === "results") diagnosisWrapper.classList.remove("hidden");
    }

    // 7. Advisory Tabs
    document.querySelectorAll(".adv-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll(".adv-tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetId = `pane-${tab.dataset.tab}`;
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add("active");
        });
    });

    // 8. Architecture Modal
    openArchModalBtn.addEventListener("click", () => {
        archModal.classList.remove("hidden");
    });

    closeArchModalBtn.addEventListener("click", () => {
        archModal.classList.add("hidden");
    });

    archModal.addEventListener("click", (e) => {
        if (e.target === archModal) {
            archModal.classList.add("hidden");
        }
    });

    // 9. Evaluation Metrics Modal
    const openMetricsModalBtn = document.getElementById("openMetricsModalBtn");
    const closeMetricsModalBtn = document.getElementById("closeMetricsModalBtn");
    const metricsModal = document.getElementById("metricsModal");

    if (openMetricsModalBtn && metricsModal) {
        openMetricsModalBtn.addEventListener("click", () => {
            metricsModal.classList.remove("hidden");
            loadEvaluationMetrics();
        });

        if (closeMetricsModalBtn) {
            closeMetricsModalBtn.addEventListener("click", () => {
                metricsModal.classList.add("hidden");
            });
        }

        metricsModal.addEventListener("click", (e) => {
            if (e.target === metricsModal) {
                metricsModal.classList.add("hidden");
            }
        });

        // Visual Tabs
        document.querySelectorAll(".v-tab").forEach(tab => {
            tab.addEventListener("click", () => {
                document.querySelectorAll(".v-tab").forEach(t => t.classList.remove("active"));
                document.querySelectorAll(".v-pane").forEach(p => p.classList.remove("active"));

                tab.classList.add("active");
                const targetPane = document.getElementById(`vpane-${tab.dataset.vtab}`);
                if (targetPane) targetPane.classList.add("active");
            });
        });
    }

    async function loadEvaluationMetrics() {
        try {
            const res = await fetch("/api/metrics");
            if (res.ok) {
                const data = await res.json();
                const m = data.overall_metrics;
                if (m) {
                    if (document.getElementById("mAcc")) document.getElementById("mAcc").textContent = (m.accuracy * 100).toFixed(2) + "%";
                    if (document.getElementById("mF1")) document.getElementById("mF1").textContent = (m.macro_f1 * 100).toFixed(2) + "%";
                    if (document.getElementById("mPrec")) document.getElementById("mPrec").textContent = (m.macro_precision * 100).toFixed(2) + "%";
                    if (document.getElementById("mMcc")) document.getElementById("mMcc").textContent = Number(m.mcc).toFixed(4);
                }
            }
        } catch (err) {
            console.error("Error fetching metrics:", err);
        }
    }
});
