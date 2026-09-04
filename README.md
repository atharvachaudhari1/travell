# ARKA — AI Travel Planner ✈️

Solo travel planning powered by Google Gemini AI + Groq fallback, with a deterministic travel-disruption recovery engine.

## Features
- 🗺 AI Trip Planner
- 💰 Travel Wallet
- ⭐ Discover Recommendations
- 🌐 Translate
- 🚀 TravelMe Booking
- 📅 Timeline Builder
- 🛡️ ARKA Rescue — deterministic disruption recovery

## Rescue Engine

ARKA Rescue models a connected itinerary as a dependency graph. Its timing,
downstream-risk, scoring, and recovery calculations are deterministic, so the
critical path still works if an AI provider is unavailable.

- Simulates flight cancellation/delay, train cancellation, hotel cancellation,
  and severe weather.
- Tags graph nodes as `SAFE`, `RISK`, `DELAYED`, or `CANCELLED`.
- Calculates lowest-cost, fastest-arrival, and maximum-continuity plans.
- Returns an approved recovery plan to the existing Timeline UI.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/rescue/scenarios` | Available simulations |
| `POST /api/rescue/analyze` | Graph, impact, scores, and recovery plans |
| `POST /api/rescue/apply` | Validates a chosen plan and restores trip health |

Example: `{ "scenario": "flight_delayed", "preferences": { "priority": "continuity" } }`.

## Tech Stack
- Frontend: HTML5 / CSS3 / Vanilla JS (SPA)
- Backend: Node.js / Express
- Rescue: deterministic dependency graph + multi-objective optimizer
- Primary AI: Google Gemini Flash
- Fallback AI: Groq (GPT-OSS, Qwen, Compound)

## Folder Layout
- `frontend/` web app
- `backend/` API proxy and same-origin web hosting
- `mobile-app/` separate mobile launcher project for the same ARKA experience
