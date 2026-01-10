/**
 * Voxtral Transcription - Frontend Application
 * Handles file upload, WebSocket communication, and real-time transcription updates
 */

// State management
const state = {
    uploadedFile: null,
    currentJobId: null,
    socket: null,
    languages: [],
    modelLoaded: false,
    availableModels: [],
    selectedModel: null
};

// DOM Elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    fileInfo: document.getElementById('fileInfo'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    fileType: document.getElementById('fileType'),
    removeFile: document.getElementById('removeFile'),
    languageSelect: document.getElementById('languageSelect'),
    transcribeBtn: document.getElementById('transcribeBtn'),
    progressSection: document.getElementById('progressSection'),
    progressStatus: document.getElementById('progressStatus'),
    progressPercentage: document.getElementById('progressPercentage'),
    progressFill: document.getElementById('progressFill'),
    progressDetails: document.getElementById('progressDetails'),
    transcriptSection: document.getElementById('transcriptSection'),
    transcriptContent: document.getElementById('transcriptContent'),
    wordCount: document.getElementById('wordCount'),
    charCount: document.getElementById('charCount'),
    audioDuration: document.getElementById('audioDuration'),
    copyBtn: document.getElementById('copyBtn'),
    downloadBtn: document.getElementById('downloadBtn'),
    clearBtn: document.getElementById('clearBtn'),
    toast: document.getElementById('toast'),
    // System banners
    memoryBanner: document.getElementById('memoryBanner'),
    memoryBannerTitle: document.getElementById('memoryBannerTitle'),
    memoryBannerMessage: document.getElementById('memoryBannerMessage'),
    closeMemoryBanner: document.getElementById('closeMemoryBanner'),
    updateBanner: document.getElementById('updateBanner'),
    updateBannerMessage: document.getElementById('updateBannerMessage'),
    installUpdateBtn: document.getElementById('installUpdateBtn'),
    viewReleaseBtn: document.getElementById('viewReleaseBtn'),
    closeUpdateBanner: document.getElementById('closeUpdateBanner'),
    checkUpdateBtn: document.getElementById('checkUpdateBtn'),
    // Model selection
    modelModal: document.getElementById('modelModal'),
    modelGrid: document.getElementById('modelGrid'),
    initializeModelBtn: document.getElementById('initializeModelBtn'),
    modelLoadingOverlay: document.getElementById('modelLoadingOverlay'),
    modelLoadingTitle: document.getElementById('modelLoadingTitle'),
    modelLoadingMessage: document.getElementById('modelLoadingMessage')
};

// Initialize application
async function init() {
    setupEventListeners();
    connectWebSocket();

    // Check if model is loaded, show modal if not
    const modelStatus = await checkModelStatus();
    if (!modelStatus.loaded) {
        await showModelSelectionModal();
    } else {
        state.modelLoaded = true;
        loadLanguages();
        startMemoryMonitoring();
        checkForUpdates();
    }
}

// Setup all event listeners
function setupEventListeners() {
    // File upload
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    elements.removeFile.addEventListener('click', clearFile);

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Transcription
    elements.transcribeBtn.addEventListener('click', startTranscription);

    // Transcript actions
    elements.copyBtn.addEventListener('click', copyTranscript);
    elements.downloadBtn.addEventListener('click', downloadTranscript);
    elements.clearBtn.addEventListener('click', clearTranscript);

    // System banners
    if (elements.closeMemoryBanner) {
        elements.closeMemoryBanner.addEventListener('click', () => {
            elements.memoryBanner.style.display = 'none';
        });
    }
    if (elements.closeUpdateBanner) {
        elements.closeUpdateBanner.addEventListener('click', () => {
            elements.updateBanner.style.display = 'none';
        });
    }

    // Install update button
    if (elements.installUpdateBtn) {
        elements.installUpdateBtn.addEventListener('click', installUpdate);
    }

    // Manual update check
    if (elements.checkUpdateBtn) {
        elements.checkUpdateBtn.addEventListener('click', manualUpdateCheck);
    }
}

// Load supported languages
async function loadLanguages() {
    try {
        const response = await fetch('/api/languages');
        state.languages = await response.json();

        // Populate language select
        elements.languageSelect.innerHTML = state.languages.map(lang =>
            `<option value="${lang.code}">${lang.name}</option>`
        ).join('');

    } catch (error) {
        console.error('Failed to load languages:', error);
        showToast('Failed to load languages', 'error');
    }
}

// WebSocket connection
function connectWebSocket() {
    state.socket = io();

    state.socket.on('connect', () => {
        console.log('WebSocket connected');
    });

    state.socket.on('connected', (data) => {
        console.log('Server message:', data.message);
    });

    state.socket.on('transcription_progress', handleProgressUpdate);
    state.socket.on('transcription_complete', handleTranscriptionComplete);
    state.socket.on('transcription_error', handleTranscriptionError);
    state.socket.on('memory_warning', handleMemoryWarning);
    state.socket.on('model_loading', handleModelLoading);
    state.socket.on('update_progress', handleUpdateProgress);

    state.socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
    });
}

// Model Selection Functions

/**
 * Check if model is loaded
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model/status');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Failed to check model status:', error);
        return { loaded: false, loading: false };
    }
}

/**
 * Load available models from API
 */
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        state.availableModels = data.models;
        return data.models;
    } catch (error) {
        console.error('Failed to load available models:', error);
        showToast('Failed to load models', 'error');
        return [];
    }
}

/**
 * Show model selection modal
 */
async function showModelSelectionModal() {
    // Load available models
    const models = await loadAvailableModels();

    if (models.length === 0) {
        showToast('No models available', 'error');
        return;
    }

    // Default to 'full' model
    state.selectedModel = models.find(m => m.id === 'full') || models[0];

    // Render model cards
    elements.modelGrid.innerHTML = models.map((model, index) => {
        const isDefault = model.id === 'full';
        const isMac = model.platform === 'mac';
        const isSelected = state.selectedModel.id === model.id;

        // Build badge HTML
        let badgeHtml = '';
        if (isDefault) {
            badgeHtml = '<div class="model-card-badge">Default</div>';
        } else if (isMac) {
            badgeHtml = '<div class="model-card-badge model-card-badge-mac">Mac</div>';
        }

        return `
            <div class="model-card ${isSelected ? 'selected' : ''} ${isMac ? 'model-card-mac' : ''}" data-model-id="${model.id}">
                ${badgeHtml}
                <h3 class="model-card-title">${model.name}</h3>
                <div class="model-card-size">${model.size_gb} GB</div>
                <p class="model-card-description">${model.description}</p>
                <div class="model-card-specs">
                    <h4>Memory Requirements:</h4>
                    <ul>
                        <li><strong>Disk:</strong> <span>${model.memory_requirements.disk}</span></li>
                        <li><strong>RAM (Loading):</strong> <span>${model.memory_requirements.ram_loading}</span></li>
                        <li><strong>RAM (Running):</strong> <span>${model.memory_requirements.ram_inference}</span></li>
                    </ul>
                </div>
            </div>
        `;
    }).join('');

    // Add click handlers to model cards
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', () => {
            const modelId = card.dataset.modelId;
            state.selectedModel = models.find(m => m.id === modelId);

            // Update selection UI
            document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');

            // Enable initialize button
            elements.initializeModelBtn.disabled = false;
        });
    });

    // Enable initialize button (default model is pre-selected)
    elements.initializeModelBtn.disabled = false;

    // Add initialize button handler
    elements.initializeModelBtn.onclick = initializeSelectedModel;

    // Show modal
    elements.modelModal.style.display = 'flex';
}

/**
 * Initialize the selected model
 */
async function initializeSelectedModel() {
    if (!state.selectedModel) {
        showToast('Please select a model', 'error');
        return;
    }

    try {
        // Disable button and show loading overlay
        elements.initializeModelBtn.disabled = true;
        elements.modelLoadingOverlay.style.display = 'flex';
        elements.modelLoadingTitle.textContent = `Loading ${state.selectedModel.name}...`;
        elements.modelLoadingMessage.textContent = 'This may take a few minutes. Please wait...';

        // Send initialization request
        const response = await fetch('/api/model/initialize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Voxtral-Request': 'voxtral-web-ui'
            },
            body: JSON.stringify({
                version: state.selectedModel.id
            })
        });

        const data = await response.json();

        if (response.ok) {
            console.log('Model initialization started:', data);
            // WebSocket will handle progress updates via handleModelLoading
        } else {
            throw new Error(data.message || 'Failed to initialize model');
        }
    } catch (error) {
        console.error('Failed to initialize model:', error);
        showToast(`Failed to initialize model: ${error.message}`, 'error');

        // Re-enable button and hide overlay
        elements.initializeModelBtn.disabled = false;
        elements.modelLoadingOverlay.style.display = 'none';
    }
}

/**
 * Handle model loading progress from WebSocket
 */
function handleModelLoading(data) {
    console.log('Model loading update:', data);

    if (data.status === 'loading') {
        elements.modelLoadingTitle.textContent = data.message || 'Loading model...';
    } else if (data.status === 'loaded') {
        // Model loaded successfully
        state.modelLoaded = true;

        // Hide modal
        elements.modelModal.style.display = 'none';
        elements.modelLoadingOverlay.style.display = 'none';

        showToast(`${data.model} loaded successfully!`, 'success');

        // Initialize the rest of the app
        loadLanguages();
        startMemoryMonitoring();
        checkForUpdates();
    } else if (data.status === 'error') {
        // Error loading model
        showToast(`Error loading model: ${data.message}`, 'error');

        // Re-enable button and hide overlay
        elements.initializeModelBtn.disabled = false;
        elements.modelLoadingOverlay.style.display = 'none';
    }
}

// Memory Monitoring Functions

/**
 * Start periodic memory monitoring
 */
function startMemoryMonitoring() {
    // Check memory immediately
    checkMemoryStatus();

    // Check every 15 seconds
    setInterval(checkMemoryStatus, 15000);
}

/**
 * Check current memory status via API
 */
async function checkMemoryStatus() {
    try {
        const response = await fetch('/api/system/memory');
        const data = await response.json();

        if (response.ok) {
            updateMemoryBanner(data);
        }
    } catch (error) {
        console.error('Failed to check memory status:', error);
    }
}

/**
 * Handle real-time memory warning from WebSocket
 */
function handleMemoryWarning(data) {
    console.log('Memory warning received:', data);

    const isCritical = data.level === 'critical';
    const title = isCritical ? 'Critical Memory Usage' : 'High Memory Usage';
    const message = data.message || `Memory usage is at ${data.percent}%`;

    // Update banner
    elements.memoryBannerTitle.textContent = title;
    elements.memoryBannerMessage.textContent = message;

    // Add/remove critical class
    if (isCritical) {
        elements.memoryBanner.classList.add('critical');
    } else {
        elements.memoryBanner.classList.remove('critical');
    }

    // Show banner
    elements.memoryBanner.style.display = 'block';
}

/**
 * Update memory banner based on current status
 */
function updateMemoryBanner(memoryData) {
    const status = memoryData.system.status;
    const percent = memoryData.system.percent;
    const availableGB = memoryData.system.available_gb;

    if (status === 'normal') {
        // Hide banner if memory is normal
        elements.memoryBanner.style.display = 'none';
    } else if (status === 'warning' || status === 'critical') {
        const isCritical = status === 'critical';
        const title = isCritical ? 'Critical Memory Usage' : 'High Memory Usage';
        const message = isCritical
            ? `Memory usage is very high (${percent}%). Consider stopping transcription to prevent system slowdown.`
            : `Memory usage is high (${percent}%). Available: ${availableGB} GB`;

        elements.memoryBannerTitle.textContent = title;
        elements.memoryBannerMessage.textContent = message;

        if (isCritical) {
            elements.memoryBanner.classList.add('critical');
        } else {
            elements.memoryBanner.classList.remove('critical');
        }

        elements.memoryBanner.style.display = 'block';
    }
}

// Update Checking Functions

/**
 * Check for application updates
 */
async function checkForUpdates() {
    try {
        const response = await fetch('/api/updates/check');
        const data = await response.json();

        if (response.ok && data.update_available) {
            showUpdateBanner(data);
        }
    } catch (error) {
        console.error('Failed to check for updates:', error);
    }
}

/**
 * Show update available banner
 */
function showUpdateBanner(updateData) {
    const message = `Version ${updateData.latest_version} is available (current: ${updateData.current_version})`;

    elements.updateBannerMessage.textContent = message;
    elements.viewReleaseBtn.href = updateData.release_url || '#';
    elements.updateBanner.style.display = 'block';

    console.log('Update available:', updateData);
}

/**
 * Manual update check triggered by user
 * Shows feedback whether update is available or not
 */
async function manualUpdateCheck() {
    try {
        showToast('Checking for updates...', 'info');

        const response = await fetch('/api/updates/check');
        const data = await response.json();

        if (response.ok) {
            if (data.update_available) {
                // Show banner and notify user
                showUpdateBanner(data);
                showToast(
                    `Update available: ${data.latest_version} (current: ${data.current_version})`,
                    'success'
                );
            } else {
                // No update available - show positive feedback
                showToast(
                    `You're up to date! Running version ${data.current_version}`,
                    'success'
                );
            }
        } else {
            throw new Error(data.error || 'Failed to check for updates');
        }
    } catch (error) {
        console.error('Update check failed:', error);
        showToast('Failed to check for updates. Please try again later.', 'error');
    }
}

/**
 * Install available update
 * Downloads and installs the latest version from GitHub
 */
async function installUpdate() {
    try {
        // Disable the button to prevent double-clicks
        const btn = elements.installUpdateBtn;
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Installing...';

        showToast('Starting update installation...', 'info');

        const response = await fetch('/api/updates/install', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Voxtral-Request': 'voxtral-web-ui'
            }
        });

        const data = await response.json();

        if (response.ok) {
            if (data.updated) {
                showToast('Update installed! Application restarting...', 'success');
                // Hide the update banner
                elements.updateBanner.style.display = 'none';

                // Show a modal or overlay to inform user about restart
                setTimeout(() => {
                    showToast('Restarting application...', 'info');
                    // The server will restart, which will disconnect the socket
                    // The page should automatically reconnect when server is back
                }, 1000);
            } else {
                showToast(data.message || 'Already up to date', 'info');
                btn.disabled = false;
                btn.textContent = originalText;
            }
        } else {
            throw new Error(data.message || 'Failed to install update');
        }
    } catch (error) {
        console.error('Update installation failed:', error);
        showToast(`Update failed: ${error.message}`, 'error');

        // Re-enable button
        if (elements.installUpdateBtn) {
            elements.installUpdateBtn.disabled = false;
            elements.installUpdateBtn.textContent = 'Update Now';
        }
    }
}

/**
 * Handle update progress from WebSocket
 * Shows real-time progress of ZIP-based updates with timestamps
 */
let lastUpdateStage = null;  // Track stage to avoid duplicate toasts
let updateProgressLog = [];  // Store all progress events for debugging

function handleUpdateProgress(data) {
    const { stage, message, progress, timestamp } = data;

    // Log with timestamp for debugging
    const logEntry = `[${timestamp || 'N/A'}] ${stage}: ${message} (${progress}%)`;
    console.log('Update progress:', logEntry);
    updateProgressLog.push(logEntry);

    // Update button text with progress and timestamp
    const btn = elements.installUpdateBtn;
    if (btn) {
        const displayText = timestamp ? `${message} @ ${timestamp}` : message;
        btn.textContent = `${displayText} (${progress}%)`;
    }

    // Only show toast when stage CHANGES (not on every progress update)
    if (stage !== lastUpdateStage) {
        const stageMessages = {
            checking: 'Checking for updates...',
            downloading: 'Downloading update files...',
            extracting: 'Extracting files...',
            configuring: 'Merging configuration...',
            migrating: 'Migrating user data...',
            backing_up: 'Backing up your data...',
            preparing: 'Preparing update script...',
            installing: 'Installing update...',
            restoring: 'Restoring your data...',
            dependencies: 'Updating dependencies...',
            launching: 'Launching updater...',
            complete: 'Update ready! Application restarting...'
        };

        if (stageMessages[stage]) {
            const toastMsg = timestamp ? `${stageMessages[stage]} @ ${timestamp}` : stageMessages[stage];
            showToast(toastMsg, stage === 'complete' ? 'success' : 'info');
            lastUpdateStage = stage;
        }
    }

    // Log all progress events to console for support debugging
    if (updateProgressLog.length <= 100) {  // Prevent memory leak
        console.log('Update Progress Log:', updateProgressLog);
    }
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadArea.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadArea.classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadArea.classList.remove('dragover');

    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

// Process selected file
function processFile(file) {
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/x-m4a', 'audio/m4a',
                          'video/mp4', 'video/x-msvideo', 'video/quicktime'];
    const allowedExtensions = ['.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov'];

    const extension = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedExtensions.includes(extension)) {
        showToast('File type not supported. Please upload audio or video files.', 'error');
        return;
    }

    // Validate file size (500MB)
    if (file.size > 500 * 1024 * 1024) {
        showToast('File too large. Maximum size is 500MB.', 'error');
        return;
    }

    // Upload file
    uploadFile(file);
}

// Upload file to server
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        showToast('Uploading file...', 'info');

        const response = await fetch('/api/upload', {
            method: 'POST',
            headers: {
                'X-Voxtral-Request': 'voxtral-web-ui'
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            state.uploadedFile = data;
            displayFileInfo(file, data);
            elements.transcribeBtn.disabled = false;
            showToast('File uploaded successfully!', 'success');
        } else {
            throw new Error(data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showToast('Upload failed: ' + error.message, 'error');
    }
}

// Display file information
function displayFileInfo(file, uploadData) {
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = uploadData.size_mb + ' MB';
    elements.fileType.textContent = uploadData.is_video ? '🎬 Video' : '🎵 Audio';

    elements.uploadArea.style.display = 'none';
    elements.fileInfo.style.display = 'block';
}

// Clear selected file
function clearFile() {
    state.uploadedFile = null;
    elements.fileInput.value = '';
    elements.uploadArea.style.display = 'block';
    elements.fileInfo.style.display = 'none';
    elements.transcribeBtn.disabled = true;
}

// Start transcription
async function startTranscription() {
    if (!state.uploadedFile) {
        showToast('Please upload a file first', 'error');
        return;
    }

    const language = elements.languageSelect.value;

    try {
        elements.transcribeBtn.disabled = true;
        elements.transcribeBtn.textContent = 'Transcribing...';

        const response = await fetch('/api/transcribe', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Voxtral-Request': 'voxtral-web-ui'
            },
            body: JSON.stringify({
                file_id: state.uploadedFile.file_id,
                language: language
            })
        });

        const data = await response.json();

        if (response.ok) {
            state.currentJobId = data.job_id;
            showProgressSection();
            showToast('Transcription started!', 'success');
        } else {
            throw new Error(data.error || 'Failed to start transcription');
        }
    } catch (error) {
        console.error('Transcription error:', error);
        showToast('Failed to start transcription: ' + error.message, 'error');
        elements.transcribeBtn.disabled = false;
        elements.transcribeBtn.textContent = 'Start Transcription';
    }
}

// Show progress section
function showProgressSection() {
    elements.progressSection.style.display = 'block';
    elements.transcriptSection.style.display = 'none';
}

// Handle progress update from WebSocket
function handleProgressUpdate(data) {
    if (data.job_id !== state.currentJobId) return;

    const progress = data.progress || 0;
    const message = data.message || 'Processing...';

    elements.progressPercentage.textContent = progress + '%';
    elements.progressFill.style.width = progress + '%';
    elements.progressStatus.textContent = data.status || 'Processing';

    if (data.current_chunk && data.total_chunks) {
        elements.progressDetails.textContent =
            `${message} (Chunk ${data.current_chunk}/${data.total_chunks})`;
    } else {
        elements.progressDetails.textContent = message;
    }

    console.log('Progress:', progress + '%', message);
}

// Handle transcription complete
function handleTranscriptionComplete(data) {
    if (data.job_id !== state.currentJobId) return;

    console.log('Transcription complete!');

    // Update UI
    elements.progressPercentage.textContent = '100%';
    elements.progressFill.style.width = '100%';
    elements.progressStatus.textContent = 'Complete';
    elements.progressDetails.textContent = 'Transcription completed successfully!';

    // Display transcript
    displayTranscript(data);

    // Reset button
    elements.transcribeBtn.disabled = false;
    elements.transcribeBtn.textContent = 'Start Transcription';

    showToast('Transcription completed!', 'success');
}

// Handle transcription error
function handleTranscriptionError(data) {
    if (data.job_id !== state.currentJobId) return;

    console.error('Transcription error:', data.error);

    elements.progressStatus.textContent = 'Error';
    elements.progressDetails.textContent = 'Error: ' + data.error;
    elements.progressFill.style.width = '0%';

    elements.transcribeBtn.disabled = false;
    elements.transcribeBtn.textContent = 'Start Transcription';

    showToast('Transcription failed: ' + data.error, 'error');
}

// Display transcript
function displayTranscript(data) {
    const transcript = data.transcript || '';
    const words = transcript.split(/\s+/).filter(w => w.length > 0);
    const chars = transcript.length;
    const duration = data.duration || 0;

    // Update transcript content
    elements.transcriptContent.textContent = transcript;

    // Update stats
    elements.wordCount.textContent = words.length.toLocaleString();
    elements.charCount.textContent = chars.toLocaleString();
    elements.audioDuration.textContent = formatDuration(duration);

    // Show transcript section
    elements.transcriptSection.style.display = 'block';

    // Scroll to transcript
    elements.transcriptSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format duration (minutes to MM:SS)
function formatDuration(minutes) {
    const totalSeconds = Math.round(minutes * 60);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Copy transcript to clipboard
async function copyTranscript() {
    const text = elements.transcriptContent.textContent;

    if (!text || text.trim() === '') {
        showToast('No transcript to copy', 'warning');
        return;
    }

    try {
        await navigator.clipboard.writeText(text);
        showToast('Transcript copied to clipboard!', 'success');
    } catch (error) {
        console.error('Copy failed:', error);

        // Fallback method
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            document.execCommand('copy');
            showToast('Transcript copied to clipboard!', 'success');
        } catch (err) {
            showToast('Failed to copy transcript', 'error');
        }

        document.body.removeChild(textarea);
    }
}

// Download transcript
async function downloadTranscript() {
    if (!state.currentJobId) {
        showToast('No transcript to download', 'warning');
        return;
    }

    try {
        window.location.href = `/api/transcript/${state.currentJobId}/download`;
        showToast('Downloading transcript...', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showToast('Download failed', 'error');
    }
}

// Clear transcript
function clearTranscript() {
    elements.transcriptContent.textContent = '';
    elements.wordCount.textContent = '0';
    elements.charCount.textContent = '0';
    elements.audioDuration.textContent = '0:00';
    elements.transcriptSection.style.display = 'none';
    elements.progressSection.style.display = 'none';
    state.currentJobId = null;
}

// Show toast notification
function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = 'toast show ' + type;

    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// History Management Functions

const historyElements = {
    historyOverlay: document.getElementById('historyOverlay'),
    historyBackdrop: document.getElementById('historyBackdrop'),
    historySection: document.getElementById('historySection'),
    toggleHistoryBtn: document.getElementById('toggleHistory'),
    closeHistory: document.getElementById('closeHistory'),
    tabTranscriptions: document.getElementById('tabTranscriptions'),
    tabUploads: document.getElementById('tabUploads'),
    transcriptionsTab: document.getElementById('transcriptionsTab'),
    uploadsTab: document.getElementById('uploadsTab'),
    transcriptionsList: document.getElementById('transcriptionsList'),
    uploadsList: document.getElementById('uploadsList'),
    refreshTranscriptions: document.getElementById('refreshTranscriptions'),
    refreshUploads: document.getElementById('refreshUploads'),
    deleteAllTranscriptions: document.getElementById('deleteAllTranscriptions'),
    deleteAllUploads: document.getElementById('deleteAllUploads')
};

function toggleHistoryOverlay(show) {
    if (!historyElements.historyOverlay) return;
    const shouldShow = typeof show === 'boolean' ? show : !historyElements.historyOverlay.classList.contains('open');
    historyElements.historyOverlay.classList.toggle('open', shouldShow);
    if (historyElements.toggleHistoryBtn) {
        historyElements.toggleHistoryBtn.innerHTML = shouldShow
            ? '<span class="btn-icon">📚</span> Hide History'
            : '<span class="btn-icon">📚</span> View History';
    }
    if (shouldShow) {
        loadTranscriptions();
    }
}

if (historyElements.toggleHistoryBtn) {
    historyElements.toggleHistoryBtn.addEventListener('click', () => toggleHistoryOverlay());
}

if (historyElements.closeHistory) {
    historyElements.closeHistory.addEventListener('click', () => toggleHistoryOverlay(false));
}

if (historyElements.historyBackdrop) {
    historyElements.historyBackdrop.addEventListener('click', () => toggleHistoryOverlay(false));
}

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && historyElements.historyOverlay?.classList.contains('open')) {
        toggleHistoryOverlay(false);
    }
});

// Tab switching
if (historyElements.tabTranscriptions) {
    historyElements.tabTranscriptions.addEventListener('click', () => {
        historyElements.tabTranscriptions.classList.add('active');
        historyElements.tabUploads.classList.remove('active');
        historyElements.transcriptionsTab.style.display = 'block';
        historyElements.uploadsTab.style.display = 'none';
        loadTranscriptions();
    });
}

if (historyElements.tabUploads) {
    historyElements.tabUploads.addEventListener('click', () => {
        historyElements.tabUploads.classList.add('active');
        historyElements.tabTranscriptions.classList.remove('active');
        historyElements.uploadsTab.style.display = 'block';
        historyElements.transcriptionsTab.style.display = 'none';
        loadUploads();
    });
}

// Load transcriptions
async function loadTranscriptions() {
    try {
        const response = await fetch('/api/history/transcriptions');
        const transcriptions = await response.json();

        if (transcriptions.length === 0) {
            historyElements.transcriptionsList.innerHTML = '<div class="history-placeholder">No transcriptions yet</div>';
            return;
        }

        historyElements.transcriptionsList.innerHTML = transcriptions.map(item => `
            <div class="history-item">
                <div class="history-item-info">
                    <div class="history-item-name">${item.filename}</div>
                    <div class="history-item-meta">
                        <span>${item.size_kb} KB</span>
                        <span>${new Date(item.modified).toLocaleString()}</span>
                    </div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon-only" onclick="viewTranscription('${item.filename}')" title="View">
                        👁️
                    </button>
                    <button class="btn-icon-only" onclick="downloadTranscriptionFile('${item.filename}')" title="Download">
                        💾
                    </button>
                    <button class="btn-icon-only" onclick="deleteTranscription('${item.filename}')" title="Delete">
                        🗑️
                    </button>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading transcriptions:', error);
        showToast('Failed to load transcriptions', 'error');
    }
}

// Load uploads
async function loadUploads() {
    try {
        const response = await fetch('/api/history/uploads');
        const uploads = await response.json();

        if (uploads.length === 0) {
            historyElements.uploadsList.innerHTML = '<div class="history-placeholder">No uploads yet</div>';
            return;
        }

        historyElements.uploadsList.innerHTML = uploads.map(item => `
            <div class="history-item">
                <div class="history-item-info">
                    <div class="history-item-name">${item.filename}</div>
                    <div class="history-item-meta">
                        <span>${item.size_mb} MB</span>
                        <span>${new Date(item.modified).toLocaleString()}</span>
                    </div>
                </div>
                <div class="history-item-actions">
                    <button class="btn-icon-only" onclick="deleteUpload('${item.filename}')" title="Delete">
                        🗑️
                    </button>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading uploads:', error);
        showToast('Failed to load uploads', 'error');
    }
}

// View transcription
async function viewTranscription(filename) {
    try {
        const response = await fetch(`/api/history/transcriptions/${encodeURIComponent(filename)}`);
        const data = await response.json();

        if (response.ok) {
            elements.transcriptContent.textContent = data.content;
            elements.wordCount.textContent = data.word_count.toLocaleString();
            elements.charCount.textContent = data.char_count.toLocaleString();
            elements.transcriptSection.style.display = 'block';
            elements.transcriptSection.scrollIntoView({ behavior: 'smooth' });
            showToast('Transcription loaded', 'success');
        } else {
            throw new Error(data.error || 'Failed to load transcription');
        }
    } catch (error) {
        console.error('Error viewing transcription:', error);
        showToast('Failed to load transcription', 'error');
    }
}

// Download transcription file
function downloadTranscriptionFile(filename) {
    window.location.href = `/api/history/transcriptions/${encodeURIComponent(filename)}/download`;
}

// Delete transcription
async function deleteTranscription(filename) {
    if (!confirm(`Delete "${filename}"?`)) return;

    try {
        const response = await fetch(`/api/history/transcriptions/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
            headers: {
                'X-Voxtral-Request': 'voxtral-web-ui'
            }
        });

        if (response.ok) {
            showToast('Transcription deleted', 'success');
            loadTranscriptions();
        } else {
            throw new Error('Failed to delete');
        }
    } catch (error) {
        console.error('Error deleting transcription:', error);
        showToast('Failed to delete transcription', 'error');
    }
}

// Delete upload
async function deleteUpload(filename) {
    if (!confirm(`Delete "${filename}"?`)) return;

    try {
        const response = await fetch(`/api/history/uploads/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
            headers: {
                'X-Voxtral-Request': 'voxtral-web-ui'
            }
        });

        if (response.ok) {
            showToast('Upload deleted', 'success');
            loadUploads();
        } else {
            throw new Error('Failed to delete');
        }
    } catch (error) {
        console.error('Error deleting upload:', error);
        showToast('Failed to delete upload', 'error');
    }
}

// Delete all transcriptions
if (historyElements.deleteAllTranscriptions) {
    historyElements.deleteAllTranscriptions.addEventListener('click', async () => {
        if (!confirm('Delete ALL transcriptions? This cannot be undone!')) return;

        try {
            const response = await fetch('/api/history/transcriptions/all', {
                method: 'DELETE',
                headers: {
                    'X-Voxtral-Request': 'voxtral-web-ui'
                }
            });

            const data = await response.json();

            if (response.ok) {
                showToast(`Deleted ${data.count} transcriptions`, 'success');
                loadTranscriptions();
            } else {
                throw new Error('Failed to delete all');
            }
        } catch (error) {
            console.error('Error deleting all transcriptions:', error);
            showToast('Failed to delete all transcriptions', 'error');
        }
    });
}

// Delete all uploads
if (historyElements.deleteAllUploads) {
    historyElements.deleteAllUploads.addEventListener('click', async () => {
        if (!confirm('Delete ALL uploads? This cannot be undone!')) return;

        try {
            const response = await fetch('/api/history/uploads/all', {
                method: 'DELETE',
                headers: {
                    'X-Voxtral-Request': 'voxtral-web-ui'
                }
            });

            const data = await response.json();

            if (response.ok) {
                showToast(`Deleted ${data.count} uploads`, 'success');
                loadUploads();
            } else {
                throw new Error('Failed to delete all');
            }
        } catch (error) {
            console.error('Error deleting all uploads:', error);
            showToast('Failed to delete all uploads', 'error');
        }
    });
}

// Refresh buttons
if (historyElements.refreshTranscriptions) {
    historyElements.refreshTranscriptions.addEventListener('click', loadTranscriptions);
}

if (historyElements.refreshUploads) {
    historyElements.refreshUploads.addEventListener('click', loadUploads);
}

// REMOVED: Auto-shutdown on page unload
// This feature was removed for security reasons. The /api/shutdown endpoint
// was vulnerable to CSRF attacks - any malicious website could kill the server.
// Users should stop the server manually via Ctrl+C in the terminal.
//
// Previous behavior: window.addEventListener('beforeunload', ...) would call /api/shutdown

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
