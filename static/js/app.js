/**
 * Voxtral Transcription - Frontend Application
 * Handles file upload, WebSocket communication, and real-time transcription updates
 */

// State management
const state = {
    uploadedFile: null,
    currentJobId: null,
    socket: null,
    languages: []
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
    toast: document.getElementById('toast')
};

// Initialize application
function init() {
    setupEventListeners();
    loadLanguages();
    connectWebSocket();
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

    state.socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
    });
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
                'Content-Type': 'application/json'
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

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
