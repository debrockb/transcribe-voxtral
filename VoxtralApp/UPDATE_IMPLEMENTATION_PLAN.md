# Comprehensive Update Fix - Implementation Plan

## Issues to Address

### 1. Windows Directory Lock
- **Problem**: `move` fails even after process exits due to directory locks
- **Solution**:
  - Use `robocopy /MIR` instead of `move` on Windows
  - Use `rsync -a` instead of `mv` on Mac/Linux
  - Infinite loop waiting for successful copy (not just 30s timeout)
  - Delete source after successful copy

### 2. Fragile Restart Path
- **Problem**: Hardcoded `python.exe app.py` breaks if venv changes
- **Solution**:
  - Windows: Restart via `Start Voxtral Web - Windows.bat`
  - Mac/Linux: Restart via `VoxtralApp/start_web.sh`
  - These scripts handle venv activation properly

### 3. Lossy Config Merge
- **Problem**: Only preserving `model.version`, losing other user settings
- **Solution**:
  - Use config_manager to get all user preferences
  - Merge recursively, preserving all user-modified keys
  - Document what gets preserved

### 4. No Error Feedback
- **Problem**: User sees app vanish with no explanation if update fails
- **Solution**:
  - Create `.UPDATE_FAILED` file in install root on error
  - Include log path and error message in file
  - Check for this file on app startup
  - Display banner with instructions if found

### 5. No Tests
- **Problem**: ZIP updater path completely untested
- **Solution**:
  - Add unit tests mocking requests.get and zipfile
  - Test successful update flow
  - Test error scenarios
  - Verify script creation and os._exit scheduling

## Implementation Steps

1. Rewrite Windows batch script with robocopy + infinite loop
2. Rewrite Mac/Linux shell script with rsync + infinite loop
3. Add config preservation logic using config_manager
4. Add .UPDATE_FAILED file creation and startup check
5. Write comprehensive tests
6. Test on both Windows and Mac

## Cross-Platform Considerations

- Windows uses `robocopy /MIR`
- Mac/Linux uses `rsync -a --delete`
- Windows launcher: `Start Voxtral Web - Windows.bat`
- Mac/Linux launcher: `VoxtralApp/start_web.sh`
- Different path separators handled via Path
- Different process checking (tasklist vs kill -0)
