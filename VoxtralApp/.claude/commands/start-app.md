---
description: Start the Voxtral Flask application
---

Start the Flask web application using the virtual environment:

1. Check if the app is already running on port 5000
2. If running, ask if user wants to restart it
3. Start the application using:
```bash
voxtral_env/bin/python app.py
```

4. Inform the user:
   - The app is running at http://localhost:5000
   - How to stop the app (Ctrl+C)
   - Where to find logs
   - Current configuration (model version, etc.)
