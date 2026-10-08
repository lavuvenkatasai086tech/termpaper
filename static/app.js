document.addEventListener("DOMContentLoaded", () => {
    // State
    let selectedFile = null;
    let selectedSamplePath = null;
    let sampleList = [];

    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzoneEmpty = document.getElementById("dropzoneEmpty");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeImgBtn = document.getElementById("removeImgBtn");
    const scanOverlay = document.getElementById("scanOverlay");
    const analyzeBtn = document.getElementById("analyzeBtn");

    const sampleGrid = document.getElementById("sampleGrid");
    const sampleCount = document.getElementById("sampleCount");
    const classChips = document.getElementById("classChips");

    const emptyState = document.getElementById("emptyState");
    const analyzingState = document.getElementById("analyzingState");
    const diagnosisWrapper = document.getElementById("diagnosisWrapper");

    const deviceText = document.getElementById("deviceText");
    const resSeverity = document.getElementById("resSeverity");
    const resTitle = document.getElementById("resTitle");
    const resPathogen = document.getElementById("resPathogen");
    const resConfidence = document.getElementById("resConfidence");
    const resConfidenceCircle = document.getElementById("resConfidenceCircle");
    const resSummary = document.getElementById("resSummary");
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

    // 1. Health Status
    async function fetchHealthStatus() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            deviceText.textContent = data.device.toUpperCase();
            if (data.cuda_available) {
                deviceText.innerHTML = `<span style="color: #10b981;">GPU (CUDA)</span>`;
            } else {
                deviceText.innerHTML = `<span>CPU Inference</span>`;
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
            const item = document.createElement("div");
            item.className = "sample-item";
            item.dataset.path = sample.relative_path;
            item.dataset.class = sample.class_name;
            item.title = `Click to test: ${sample.class_name} (${sample.filename})`;

            item.innerHTML = `
                <img src="${sample.url}" alt="${sample.class_name}" loading="lazy">
                <div class="sample-tag">${sample.class_name}</div>
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

        imagePreview.src = sample.url;
        dropzoneEmpty.classList.add("hidden");
        previewContainer.classList.remove("hidden");
        analyzeBtn.disabled = false;

        // Auto trigger diagnostic run for snappy demonstration
        triggerDiagnosis();
    }

    // 3. Filter Chips
    classChips.addEventListener("click", (e) => {
        if (e.target.classList.contains("chip")) {
            classChips.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
            e.target.classList.add("active");
            renderSamples(e.target.dataset.class);
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

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzoneEmpty.classList.add("hidden");
            previewContainer.classList.remove("hidden");
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
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        dropzoneEmpty.classList.remove("hidden");
        analyzeBtn.disabled = true;
        scanOverlay.classList.add("hidden");
        showState("empty");
    }

    // 5. Run Diagnosis
    analyzeBtn.addEventListener("click", triggerDiagnosis);

    async function triggerDiagnosis() {
        if (!selectedFile && !selectedSamplePath) return;

        showState("analyzing");
        scanOverlay.classList.remove("hidden");
        analyzeBtn.disabled = true;

        const formData = new FormData();
        if (selectedFile) {
            formData.append("file", selectedFile);
        } else if (selectedSamplePath) {
            formData.append("sample_path", selectedSamplePath);
        }

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
            // Provide smooth brief scan animation feel
            const elapsed = performance.now() - startTime;
            const remainingDelay = Math.max(400 - elapsed, 100);

            setTimeout(() => {
                scanOverlay.classList.add("hidden");
                analyzeBtn.disabled = false;
                renderResults(data);
                showState("results");
            }, remainingDelay);

        } catch (err) {
            console.error("Diagnosis error:", err);
            scanOverlay.classList.add("hidden");
            analyzeBtn.disabled = false;
            alert("Error analyzing image: " + err.message);
            showState("empty");
        }
    }

    // 6. Render Results
    function renderResults(data) {
        const details = data.details || {};
        const conf = data.confidence || 0;

        // Title and Severity
        resTitle.textContent = details.title || data.predicted_class;
        resPathogen.textContent = `Pathogen: ${details.pathogen || 'N/A'}`;
        resSeverity.textContent = details.severity || 'Unknown';
        resSeverity.style.backgroundColor = details.severity_color || '#dc2626';

        // Confidence circle & value
        resConfidence.textContent = `${conf.toFixed(1)}%`;
        resConfidenceCircle.style.background = `conic-gradient(var(--primary) ${conf}%, rgba(255, 255, 255, 0.1) 0)`;

        // Summary text
        resSummary.textContent = details.summary || "";

        // Probability bars
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

            // Animate bar width after appending
            setTimeout(() => {
                const bar = row.querySelector(".prob-bar");
                bar.style.width = `${Math.min(p.probability, 100)}%`;
                if (p.class === data.predicted_class) {
                    bar.style.background = "linear-gradient(90deg, #059669, #10b981)";
                } else {
                    bar.style.background = "linear-gradient(90deg, #475569, #64748b)";
                }
            }, 50);
        });

        // Populate lists
        populateList(symptomsList, details.symptoms);
        populateList(organicList, details.organic_remedies);
        populateList(chemicalList, details.chemical_remedies);
        populateList(preventionList, details.prevention);
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
        diagnosisWrapper.classList.add("hidden");

        if (state === "empty") emptyState.classList.remove("hidden");
        else if (state === "analyzing") analyzingState.classList.remove("hidden");
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
