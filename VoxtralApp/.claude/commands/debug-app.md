---
description: Debug common issues with the Voxtral application
---

Help diagnose and fix common issues with the Voxtral application:

1. Check system status:
   - Python version: `python --version`
   - Virtual environment status
   - Available memory: `python -c "import psutil; print(f'{psutil.virtual_memory().available / (1024**3):.2f} GB available')"`
   - Disk space

2. Check dependencies:
   - Verify all requirements are installed
   - Check for version conflicts

3. Check Flask app status:
   - Is the app running on port 5000?
   - Check for any error logs
   - Verify upload/output folders exist

4. Check model status:
   - What model is configured in config.json?
   - Is the model downloaded?
   - Check model cache location

5. Common issues to check:
   - Port 5000 already in use
   - Out of memory errors
   - Model loading failures
   - Permission issues with folders

Provide a diagnostic report with recommendations for fixing any issues found.
