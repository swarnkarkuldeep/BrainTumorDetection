# Scanline (frontend)

React + Vite frontend for the brain tumor detector. Uploads an MRI scan to the
backend API and displays the prediction.

## Setup

```bash
npm install
cp .env.example .env   # points at the backend; edit VITE_API_BASE_URL if needed
npm run dev
```

Requires the backend (`../backend`) running and reachable at
`VITE_API_BASE_URL` (defaults to `http://localhost:8000`).

## Build

```bash
npm run build      # outputs to dist/
npm run preview    # serve the production build locally
```

See the repo root `README.md` for the full project (training, backend, Docker).
