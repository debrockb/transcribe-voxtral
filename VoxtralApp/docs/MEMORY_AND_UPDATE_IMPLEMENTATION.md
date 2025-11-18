# Memory Monitoring & Auto-Update System - Implementation Plan

## Overview

This document outlines the implementation plan for two critical features:
1. **RAM Monitoring & Warning System** - Prevent system swap usage by warning users when memory is high
2. **Auto-Update System** - Keep application up-to-date via GitHub releases

---

## Feature 1: RAM Monitoring & Warning System

### Requirements

- Monitor system-wide RAM usage in real-time
- Monitor transcription process-specific memory usage
- Display warning banner at 80% RAM usage
- Display critical banner at 90% RAM usage
- Non-intrusive: doesn't stop transcription, just warns user
- Suggest actions to reduce memory pressure

### Implementation Steps

#### 1.1 Backend - Memory Monitoring (app.py)

**Add Dependencies:**
```python
import psutil  # Add to requirements.txt
```

**New API Endpoint:**
```python
@app.route("/api/system/memory", methods=["GET"])
def get_memory_status():
    """Get current system and process memory usage."""
    # System memory
    system_memory = psutil.virtual_memory()

    # Process memory (this Flask app)
    process = psutil.Process()
    process_memory = process.memory_info()

    return jsonify({
        "system": {
            "total_gb": round(system_memory.total / (1024**3), 2),
            "available_gb": round(system_memory.available / (1024**3), 2),
            "used_gb": round(system_memory.used / (1024**3), 2),
            "percent": system_memory.percent,
            "status": get_memory_status_level(system_memory.percent)
        },
        "process": {
            "rss_mb": round(process_memory.rss / (1024**2), 2),
            "vms_mb": round(process_memory.vms / (1024**2), 2),
            "percent": process.memory_percent()
        }
    })

def get_memory_status_level(percent):
    """Determine memory warning level."""
    if percent >= 90:
        return "critical"
    elif percent >= 80:
        return "warning"
    else:
        return "normal"
```

**Background Memory Monitor:**
```python
def start_memory_monitor():
    """Start background thread to monitor memory and emit warnings."""
    def monitor():
        while True:
            memory = psutil.virtual_memory()
            percent = memory.percent

            if percent >= 80:
                socketio.emit("memory_warning", {
                    "level": "critical" if percent >= 90 else "warning",
                    "percent": percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "message": get_memory_warning_message(percent)
                })

            time.sleep(5)  # Check every 5 seconds

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

def get_memory_warning_message(percent):
    """Get appropriate warning message based on memory usage."""
    if percent >= 90:
        return "CRITICAL: Memory usage is very high. Consider stopping transcription."
    elif percent >= 80:
        return "WARNING: Memory usage is high. Transcription may slow down."
    return ""
```

#### 1.2 Frontend - Memory Warning Banner (app.js & index.html)

**HTML Banner (index.html):**
```html
<!-- Memory Warning Banner (add after header) -->
<div id="memoryWarningBanner" class="memory-warning-banner hidden">
    <div class="warning-content">
        <span class="warning-icon">⚠️</span>
        <div class="warning-text">
            <strong id="memoryWarningTitle">Memory Warning</strong>
            <p id="memoryWarningMessage"></p>
        </div>
        <button id="dismissMemoryWarning" class="btn-dismiss">Dismiss</button>
    </div>
</div>
```

**CSS Styling (style.css):**
```css
.memory-warning-banner {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: slideDown 0.3s ease;
}

.memory-warning-banner.warning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-bottom: 2px solid #ffc107;
    color: #856404;
}

.memory-warning-banner.critical {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-bottom: 2px solid #dc3545;
    color: #721c24;
}

.memory-warning-banner.hidden {
    display: none;
}

@keyframes slideDown {
    from {
        transform: translateY(-100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}
```

**JavaScript Monitoring (app.js):**
```javascript
// Memory monitoring
let memoryCheckInterval = null;

function startMemoryMonitoring() {
    // Check memory every 10 seconds via API
    memoryCheckInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/system/memory');
            const data = await response.json();

            if (data.system.status !== 'normal') {
                showMemoryWarning(data.system.status, data.system.percent, data.system.available_gb);
            } else {
                hideMemoryWarning();
            }
        } catch (error) {
            console.error('Error checking memory:', error);
        }
    }, 10000);
}

// Listen for real-time memory warnings via WebSocket
socket.on('memory_warning', (data) => {
    showMemoryWarning(data.level, data.percent, data.available_gb);
});

function showMemoryWarning(level, percent, availableGB) {
    const banner = document.getElementById('memoryWarningBanner');
    const title = document.getElementById('memoryWarningTitle');
    const message = document.getElementById('memoryWarningMessage');

    banner.className = `memory-warning-banner ${level}`;

    if (level === 'critical') {
        title.textContent = 'CRITICAL: Memory Usage Very High!';
        message.textContent = `RAM usage at ${percent}%. Only ${availableGB}GB available. Consider stopping transcription to prevent system slowdown.`;
    } else {
        title.textContent = 'Warning: High Memory Usage';
        message.textContent = `RAM usage at ${percent}%. ${availableGB}GB available. Transcription may slow down.`;
    }
}

function hideMemoryWarning() {
    const banner = document.getElementById('memoryWarningBanner');
    banner.classList.add('hidden');
}

// Dismiss button
document.getElementById('dismissMemoryWarning')?.addEventListener('click', hideMemoryWarning);

// Start monitoring on app init
startMemoryMonitoring();
```

#### 1.3 Memory Optimization Strategies

**Dynamic Chunk Size Adjustment:**
```python
def get_optimal_chunk_size():
    """Determine optimal chunk size based on available memory."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)

    # Adjust chunk size based on available memory
    if available_gb < 4:
        return 60  # 1 minute chunks for low memory
    elif available_gb < 8:
        return 90  # 1.5 minute chunks for medium memory
    else:
        return 120  # 2 minute chunks for good memory
```

**Memory Cleanup During Transcription:**
```python
# In transcription_engine.py
def _transcribe_chunk(self, chunk_path):
    """Transcribe a single chunk with memory cleanup."""
    try:
        # Process chunk
        result = self._process_audio_chunk(chunk_path)

        # Force garbage collection after each chunk
        import gc
        gc.collect()

        # Clear MPS cache if on Apple Silicon
        if self.device == "mps":
            import torch
            torch.mps.empty_cache()

        return result
    finally:
        # Ensure temporary files are cleaned up
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
```

---

## Feature 2: Auto-Update System

### Version Management Strategy

**Semantic Versioning:**
- Format: `vMAJOR.MINOR.PATCH` (e.g., v1.0.0, v1.2.3)
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

**Version Storage:**
```
VoxtralApp/VERSION
```
Content: `1.0.0` (just the version number)

**GitHub Releases:**
- Create releases on GitHub with version tags
- Include release notes describing changes
- Attach assets if needed (though not required for this project)

### Implementation Steps

#### 2.1 Version File

**Create VERSION file:**
```bash
# VoxtralApp/VERSION
1.0.0
```

**Add to .gitignore exclusion:**
```gitignore
# Ensure VERSION file is tracked
!VERSION
```

#### 2.2 Backend - Update Checker (app.py)

**New Module: update_checker.py**
```python
"""
Update Checker Module
Checks GitHub for new releases and manages updates
"""

import os
import requests
import logging
from pathlib import Path
from packaging import version

logger = logging.getLogger(__name__)

GITHUB_REPO = "debrockb/transcribe-voxtral"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
VERSION_FILE = Path(__file__).parent / "VERSION"

def get_current_version():
    """Read current version from VERSION file."""
    try:
        if VERSION_FILE.exists():
            return VERSION_FILE.read_text().strip()
        return "0.0.0"
    except Exception as e:
        logger.error(f"Error reading version file: {e}")
        return "0.0.0"

def get_latest_release():
    """Fetch latest release info from GitHub."""
    try:
        response = requests.get(GITHUB_API_URL, timeout=5)
        response.raise_for_status()

        data = response.json()
        return {
            "version": data["tag_name"].lstrip('v'),
            "name": data["name"],
            "body": data["body"],
            "published_at": data["published_at"],
            "html_url": data["html_url"],
            "download_url": data["zipball_url"]
        }
    except Exception as e:
        logger.error(f"Error fetching latest release: {e}")
        return None

def check_for_updates():
    """Check if a new version is available."""
    current = get_current_version()
    latest = get_latest_release()

    if not latest:
        return {
            "update_available": False,
            "current_version": current,
            "error": "Could not check for updates"
        }

    latest_version = latest["version"]

    try:
        is_newer = version.parse(latest_version) > version.parse(current)
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        is_newer = False

    return {
        "update_available": is_newer,
        "current_version": current,
        "latest_version": latest_version,
        "release_name": latest.get("name"),
        "release_notes": latest.get("body"),
        "release_url": latest.get("html_url"),
        "download_url": latest.get("download_url"),
        "published_at": latest.get("published_at")
    }
```

**API Endpoints (app.py):**
```python
from update_checker import check_for_updates, get_current_version

@app.route("/api/version", methods=["GET"])
def get_version():
    """Get current application version."""
    return jsonify({
        "version": get_current_version(),
        "app_name": "Voxtral Transcription"
    })

@app.route("/api/updates/check", methods=["GET"])
def check_updates():
    """Check for available updates."""
    update_info = check_for_updates()
    return jsonify(update_info)

@app.route("/api/updates/download", methods=["POST"])
def download_update():
    """Download and prepare update for installation."""
    # This will download the update and stage it for next restart
    # Implementation in Phase 2
    return jsonify({
        "status": "success",
        "message": "Update downloaded. Restart application to apply."
    })
```

#### 2.3 Frontend - Update Notification (app.js & index.html)

**HTML Update Banner (index.html):**
```html
<!-- Update Available Banner (add after header) -->
<div id="updateBanner" class="update-banner hidden">
    <div class="update-content">
        <span class="update-icon">🎉</span>
        <div class="update-text">
            <strong>Update Available!</strong>
            <p id="updateMessage"></p>
        </div>
        <div class="update-actions">
            <button id="viewUpdateBtn" class="btn-secondary">View Details</button>
            <button id="dismissUpdateBtn" class="btn-dismiss">Later</button>
        </div>
    </div>
</div>
```

**CSS Styling (style.css):**
```css
.update-banner {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-bottom: 2px solid #28a745;
    color: #155724;
    padding: 1rem;
    z-index: 9998;
    animation: slideDown 0.3s ease;
}

.update-content {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.update-icon {
    font-size: 2rem;
}

.update-actions {
    margin-left: auto;
    display: flex;
    gap: 0.5rem;
}
```

**JavaScript Update Check (app.js):**
```javascript
// Check for updates on app startup
async function checkForUpdates() {
    try {
        const response = await fetch('/api/updates/check');
        const data = await response.json();

        if (data.update_available) {
            showUpdateBanner(data);
        }
    } catch (error) {
        console.error('Error checking for updates:', error);
    }
}

function showUpdateBanner(updateInfo) {
    const banner = document.getElementById('updateBanner');
    const message = document.getElementById('updateMessage');

    message.textContent = `Version ${updateInfo.latest_version} is available! (You're on ${updateInfo.current_version})`;

    banner.classList.remove('hidden');

    // Store update info for "View Details" button
    window.updateInfo = updateInfo;
}

function hideUpdateBanner() {
    const banner = document.getElementById('updateBanner');
    banner.classList.add('hidden');
}

// View update details
document.getElementById('viewUpdateBtn')?.addEventListener('click', () => {
    if (window.updateInfo) {
        // Open GitHub release page
        window.open(window.updateInfo.release_url, '_blank');
    }
});

// Dismiss update notification
document.getElementById('dismissUpdateBtn')?.addEventListener('click', () => {
    hideUpdateBanner();
    // Store in localStorage to not show again this session
    localStorage.setItem('dismissedUpdate', window.updateInfo?.latest_version);
});

// Check for updates on app init
checkForUpdates();
```

#### 2.4 Update Application Process (Future Enhancement)

**Phase 2: Automatic Update Download & Application**

This will be implemented in a future update and will include:

1. **Download Update:**
   - Download ZIP from GitHub
   - Extract to temporary directory
   - Verify integrity

2. **Apply Update:**
   - On next app restart, detect pending update
   - Backup current version
   - Replace files with new version
   - Update VERSION file
   - Restart application

**Script: update_applier.py** (future implementation)
```python
# To be implemented - handles actual update process
# Will run on app startup to check for pending updates
```

---

## Dependencies to Add

**requirements.txt additions:**
```
psutil>=5.9.0        # System and process monitoring
packaging>=23.0      # Version comparison
requests>=2.31.0     # GitHub API calls (already included)
```

**requirements-dev.txt additions:**
```
# All already included
```

---

## Implementation Phases

### Phase 1: Memory Monitoring (Immediate)
1. Add psutil to requirements.txt
2. Implement backend memory monitoring endpoints
3. Add background memory monitor thread
4. Create frontend memory warning banner
5. Add WebSocket memory warnings
6. Test on low-memory systems

**Estimated Time:** 3-4 hours

### Phase 2: Update Checking (Immediate)
1. Create VERSION file with 1.0.0
2. Implement update_checker.py module
3. Add update check API endpoints
4. Create frontend update notification banner
5. Add update check on app startup
6. Document version strategy in README

**Estimated Time:** 2-3 hours

### Phase 3: Update Application (Future)
1. Implement update download functionality
2. Create update staging system
3. Add update application on restart
4. Add rollback capability
5. Test update process thoroughly

**Estimated Time:** 4-6 hours (future enhancement)

---

## Testing Plan

### Memory Monitoring Tests
- [ ] Test warning at 80% memory usage
- [ ] Test critical warning at 90% memory usage
- [ ] Test banner display and dismissal
- [ ] Test WebSocket real-time warnings
- [ ] Test on systems with 8GB, 16GB, 32GB RAM
- [ ] Test during active transcription

### Update System Tests
- [ ] Test version comparison logic
- [ ] Test GitHub API connectivity
- [ ] Test update detection with mock versions
- [ ] Test update banner display
- [ ] Test "View Details" opens GitHub release
- [ ] Test dismissed update doesn't show again

---

## Documentation Updates

### README.md additions:

#### Versioning Strategy Section:
```markdown
## Versioning

Voxtral uses [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes or major rewrites
- **MINOR** version: New features, backward compatible
- **PATCH** version: Bug fixes, backward compatible

Current version is stored in `VoxtralApp/VERSION`.

### Creating a New Release

1. Update `VoxtralApp/VERSION` file with new version
2. Commit changes: `git commit -m "Bump version to X.Y.Z"`
3. Create and push tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. Create GitHub Release from tag with release notes

### Checking for Updates

The application automatically checks for updates on startup. If an update
is available, a notification banner will appear with release information.
```

#### System Requirements Addition:
```markdown
## System Requirements

### Memory Requirements

- **Minimum RAM:** 8GB (with warnings enabled above 80% usage)
- **Recommended RAM:** 16GB+ for optimal performance
- **Critical:** 90%+ RAM usage will trigger critical warnings

The application monitors memory usage and warns you before system
swap/SSD usage occurs.
```

---

## Success Criteria

### Memory Monitoring
- ✅ Warnings appear at correct thresholds (80%, 90%)
- ✅ Warnings don't interrupt active transcription
- ✅ User can dismiss warnings
- ✅ System doesn't use swap when warned early
- ✅ Works across platforms (macOS, Windows, Linux)

### Update System
- ✅ Correctly detects when updates are available
- ✅ Shows user-friendly update notification
- ✅ Links to GitHub release page
- ✅ Respects user's "dismiss" choice
- ✅ Version comparison works correctly
- ✅ Handles network errors gracefully

---

## Files to Create/Modify

### New Files:
- `VoxtralApp/VERSION` - Version tracking
- `VoxtralApp/update_checker.py` - Update checking logic
- `VoxtralApp/docs/MEMORY_AND_UPDATE_IMPLEMENTATION.md` - This document

### Modified Files:
- `VoxtralApp/requirements.txt` - Add psutil, packaging
- `VoxtralApp/app.py` - Add memory & update endpoints
- `VoxtralApp/static/js/app.js` - Add monitoring & update UI
- `VoxtralApp/static/css/style.css` - Add banner styles
- `VoxtralApp/templates/index.html` - Add warning banners
- `VoxtralApp/transcription_engine.py` - Add memory optimization
- `README.md` - Add versioning & memory info

---

## Notes

- Memory monitoring uses minimal resources (check every 5-10 seconds)
- Update checks happen once on startup, no continuous polling
- Version file is simple text for easy manual editing
- GitHub API rate limits: 60 requests/hour (unauthenticated)
- Consider GitHub personal access token for higher limits if needed

