# Smart Plant Intelligence Dashboard

A single-page web interface for real-time crop recommendation and computer vision plant pathology diagnosis.

## Architecture

- **Layout Structure**: Dual-panel design with a fixed left control panel (sensor inputs and analysis controls) and an independently scrollable right intelligence panel (telemetry graphs, disease detection cards, and spatial bounding boxes).
- **Core Technologies**: Vanilla HTML5, CSS3 with responsive glassmorphism styling, and ECMAScript 2022 JavaScript. Charts rendered via Chart.js.
- **Backend Communication**: Directly connects to the FastAPI backend on port 8000 via `/predict`, `/latest`, `/health`, and `/predict/vision`. Falls back gracefully when the server is offline or restarting.

## Running Locally

Serve statically on port 3000 using Python or any static web server:

```bash
python3 -m http.server 3000
```

The dashboard is accessible at `http://localhost:3000`. It is also served directly by the FastAPI backend at `http://localhost:8000/`.
