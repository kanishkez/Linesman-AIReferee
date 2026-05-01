/**
 * AI VAR — Modern Frontend Application Logic
 */

// ─── DOM Elements ────────────────────────────────────────────────────────────

const uploadSection = document.getElementById('uploadSection');
const uploadZone = document.getElementById('uploadZone');
const videoInput = document.getElementById('videoInput');
const browseBtn = document.getElementById('browseBtn');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const removeFileBtn = document.getElementById('removeFile');
const videoPreview = document.getElementById('videoPreview');
const analyzeBtn = document.getElementById('analyzeBtn');

const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// ─── State ───────────────────────────────────────────────────────────────────

let selectedFile = null;
let currentJobId = null;
let pollInterval = null;

// ─── Initialization ──────────────────────────────────────────────────────────

// Pre-initialize Feather icons
document.addEventListener("DOMContentLoaded", () => {
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});

// ─── Upload Handling ─────────────────────────────────────────────────────────

browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    videoInput.click();
});

uploadZone.addEventListener('click', () => {
    videoInput.click();
});

videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a valid video file format.');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    const url = URL.createObjectURL(file);
    videoPreview.src = url;

    uploadZone.classList.add('hidden');
    filePreview.classList.remove('hidden');
}

removeFileBtn.addEventListener('click', resetUpload);

function resetUpload() {
    selectedFile = null;
    videoInput.value = '';
    videoPreview.src = '';
    filePreview.classList.add('hidden');
    uploadZone.classList.remove('hidden');
}

// ─── Analysis Workflow ───────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', startAnalysis);

async function startAnalysis() {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<div class="step-loader" style="position:relative;width:20px;height:20px;border-color:transparent;border-top-color:#000;display:inline-block;vertical-align:middle;margin-right:8px;"></div> Processing...';

    try {
        const formData = new FormData();
        formData.append('video', selectedFile);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await response.json();
        currentJobId = data.job_id;

        showSection('processing');
        startPolling();

    } catch (error) {
        showError(error.message);
    }
}

// ─── Status Polling ──────────────────────────────────────────────────────────

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${currentJobId}`);
            const data = await response.json();

            updatePipelineUI(data.status);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                pollInterval = null;
                await loadResults();
            } else if (data.status === 'error') {
                clearInterval(pollInterval);
                pollInterval = null;
                showError(data.error_message || 'System Fault in AI Pipeline');
            }
        } catch (error) {
            console.error('Polling connection error:', error);
        }
    }, 2000);
}

function updatePipelineUI(status) {
    const stages = ['yolo_processing', 'gemini_analyzing', 'rules_engine'];
    const stageElements = [
        document.getElementById('stage1'),
        document.getElementById('stage2'),
        document.getElementById('stage3'),
    ];

    const currentIndex = stages.indexOf(status);

    stageElements.forEach((el, i) => {
        const loader = el.querySelector('.step-loader');
        
        if (i < currentIndex || status === 'completed') {
            // Completed
            el.classList.remove('active');
            el.classList.add('completed');
            loader.classList.add('hidden');
        } else if (i === currentIndex) {
            // Active
            el.classList.add('active');
            el.classList.remove('completed');
            loader.classList.remove('hidden');
        } else {
            // Pending
            el.classList.remove('active', 'completed');
            loader.classList.add('hidden');
        }
    });
}

// ─── Results Loading & Rendering ─────────────────────────────────────────────

async function loadResults() {
    try {
        const response = await fetch(`/api/results/${currentJobId}`);
        const data = await response.json();
        renderResults(data);
        showSection('results');
        
        // Trigger stagger animations slightly after render
        setTimeout(() => {
            const items = document.querySelectorAll('.stagger-item, .stagger-1, .stagger-2, .stagger-3, .stagger-4, .stagger-5');
            items.forEach(el => el.classList.add('stagger-item')); // Ensure class exists
            
            // Re-trigger reflow
            void document.body.offsetWidth;
            
            items.forEach(el => el.classList.add('visible'));
        }, 100);

    } catch (error) {
        showError('Failed to parse final results: ' + error.message);
    }
}

function renderResults(data) {
    const decision = data.decision;
    const gemini = data.gemini;
    const yolo = data.yolo;

    // 1. Verdict Hero
    const card = document.getElementById('decisionCard');
    const badge = document.getElementById('decisionBadge');

    if (decision.is_foul) {
        card.className = 'verdict-hero glass-panel foul';
        badge.innerHTML = `
            <div class="verdict-icon"><i data-feather="alert-octagon"></i></div>
            <div class="verdict-text">
                <div class="verdict-label">Official Verdict</div>
                <div class="verdict-value">FOUL</div>
            </div>
        `;
    } else {
        card.className = 'verdict-hero glass-panel no-foul';
        badge.innerHTML = `
            <div class="verdict-icon"><i data-feather="check-circle"></i></div>
            <div class="verdict-text">
                <div class="verdict-label">Official Verdict</div>
                <div class="verdict-value">NO FOUL</div>
            </div>
        `;
    }

    // Confidence Animation
    const confPercent = Math.round(decision.confidence * 100);
    setTimeout(() => {
        document.getElementById('confidenceFill').style.width = confPercent + '%';
    }, 500);
    document.getElementById('confidenceValue').textContent = confPercent + '%';
    document.getElementById('processingTime').textContent = data.processing_time_sec + 's';

    // 2. Incident Details
    document.getElementById('foulType').textContent = formatLabel(decision.foul_type);
    document.getElementById('severity').textContent = formatLabel(decision.severity);

    const cardRec = document.getElementById('cardRec');
    cardRec.textContent = formatLabel(decision.card_recommendation);
    cardRec.className = 'detail-val';
    if (decision.card_recommendation === 'yellow') {
        cardRec.classList.add('card-yellow');
    } else if (decision.card_recommendation === 'red') {
        cardRec.classList.add('card-red');
    }

    document.getElementById('freeKick').textContent = formatLabel(decision.free_kick_recommendation);

    // 3. Reasoning Panel
    document.getElementById('reasoning').textContent = decision.reasoning;
    document.getElementById('fifaRef').textContent = decision.fifa_law_reference;

    const factorsList = document.getElementById('factorsList');
    factorsList.innerHTML = '';
    (decision.key_factors || []).forEach(f => {
        const li = document.createElement('li');
        li.textContent = f;
        factorsList.appendChild(li);
    });

    const altBox = document.getElementById('altInterpretationBox');
    if (decision.alternative_interpretation) {
        altBox.classList.remove('hidden');
        document.getElementById('altInterpretation').textContent = decision.alternative_interpretation;
    } else {
        altBox.classList.add('hidden');
    }

    // 4. Semantics Matrix (Gemini)
    if (gemini) {
        const grid = document.getElementById('geminiGrid');
        grid.innerHTML = '';
        
        const matrixFields = [
            ['Possession', gemini.ball_possession],
            ['Challenge', gemini.challenge_type],
            ['Init. Contact', gemini.initial_contact_point],
            ['Body Area', gemini.contact_body_area],
            ['Direction', gemini.challenge_direction],
            ['Force Applied', gemini.force_assessment],
            ['Studs Up', gemini.studs_showing ? 'Yes' : 'No'],
            ['Two-Footed', gemini.two_footed ? 'Yes' : 'No'],
            ['Simulation', gemini.simulation_suspected ? 'Suspected' : 'None'],
            ['Ball Playable', gemini.ball_playing_distance ? 'Yes' : 'No'],
        ];

        matrixFields.forEach(([key, val]) => {
            if (!val || val === 'N/A') return;
            const div = document.createElement('div');
            div.className = 'matrix-item';
            div.innerHTML = `
                <div class="matrix-key">${key}</div>
                <div class="matrix-val">${val}</div>
            `;
            grid.appendChild(div);
        });
    }

    // 5. Telemetry Data (YOLO)
    if (yolo) {
        const statsGrid = document.getElementById('yoloStats');
        statsGrid.innerHTML = '';

        const stats = [
            [yolo.fps?.toFixed(0), 'FPS Processed'],
            [yolo.duration_sec?.toFixed(1) + 's', 'Clip Length'],
            [yolo.max_players_detected, 'Entities Tracked'],
            [yolo.contact_frames_count, 'Contact Frames'],
        ];

        stats.forEach(([val, label]) => {
            const div = document.createElement('div');
            div.className = 'stat-box';
            div.innerHTML = `
                <div class="stat-num">${val}</div>
                <div class="stat-label">${label}</div>
            `;
            statsGrid.appendChild(div);
        });
    }

    // Re-render icons if new ones were added dynamically
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
}

// ─── UI Management ───────────────────────────────────────────────────────────

function showSection(name) {
    uploadSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    // Reset staggers when hiding results
    if (name !== 'results') {
        const items = document.querySelectorAll('.stagger-item');
        items.forEach(el => el.classList.remove('visible', 'stagger-item'));
        document.getElementById('confidenceFill').style.width = '0%';
    }

    switch (name) {
        case 'upload':
            uploadSection.classList.remove('hidden');
            break;
        case 'processing':
            processingSection.classList.remove('hidden');
            break;
        case 'results':
            resultsSection.classList.remove('hidden');
            break;
        case 'error':
            errorSection.classList.remove('hidden');
            break;
    }
}

function showError(message) {
    errorMessage.textContent = message;
    showSection('error');
}

// ─── Reset Actions ───────────────────────────────────────────────────────────

retryBtn.addEventListener('click', resetAll);
newAnalysisBtn.addEventListener('click', resetAll);

function resetAll() {
    currentJobId = null;
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
    
    resetUpload();
    resetPipelineUI();
    
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i data-feather="cpu"></i> Initiate VAR Review';
    if (typeof feather !== 'undefined') feather.replace();

    showSection('upload');
}

function resetPipelineUI() {
    ['stage1', 'stage2', 'stage3'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'completed');
        el.querySelector('.step-loader').classList.add('hidden');
    });
}

// ─── Utilities ───────────────────────────────────────────────────────────────

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatLabel(str) {
    if (!str) return 'N/A';
    return str
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}
