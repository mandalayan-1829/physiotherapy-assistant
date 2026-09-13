# AI Physio — Move Better. Recover Smarter.

An AI-powered physiotherapy assistant that provides real-time pose analysis, exercise guidance, form correction, and progress tracking.

## Architecture

```
React + TypeScript Frontend
         ↓
    FastAPI Backend
         ↓
  Python AI / MediaPipe / OpenCV Core
         ↓
     SQLite Database
```

## Features

- **Real-Time Pose Analysis** — MediaPipe-powered body tracking with frame-by-frame analysis
- **Smart Rep Counting** — Automatic repetition counting with stage detection (up/down)
- **Form Correction** — Instant feedback on posture, joint angles, and movement patterns
- **10 Supported Exercises** — 7 physiotherapy + 3 yoga exercises
- **Progress Tracking** — Session history, form scores, and analytics
- **Exercise Library** — Browse, search, and filter exercises with video tutorials
- **Diet Tracker** — Log meals and track daily nutrition
- **Doctor Appointments** — Find specialists and book appointments
- **Guardian Alerts** — WhatsApp alerts for wrong posture or unsafe exercise times

## Tech Stack

### Frontend
- React 18 + TypeScript
- Vite
- Tailwind CSS
- React Router
- Recharts (charts)
- Lucide React (icons)

### Backend
- FastAPI
- Pydantic / Pydantic Settings
- Uvicorn

### AI Core (preserved)
- MediaPipe Pose
- OpenCV
- NumPy

### Database
- SQLite (via Python sqlite3 module)

## Project Structure

```
physiotherapy-assistant/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── api/
│   │   ├── routes/              # API route handlers
│   │   │   ├── auth.py          # Login/signup
│   │   │   ├── users.py         # User profile
│   │   │   ├── exercises.py     # Exercise library
│   │   │   ├── sessions.py      # Exercise sessions
│   │   │   ├── progress.py      # Progress analytics
│   │   │   ├── diet.py          # Diet tracking
│   │   │   ├── notes.py         # User notes
│   │   │   ├── doctors.py       # Doctors, appointments, messages
│   │   │   └── websocket.py     # Real-time session WebSocket
│   │   └── dependencies.py      # Auth middleware
│   ├── services/
│   │   ├── exercise_service.py  # Exercise metadata + safety checks
│   │   ├── session_service.py   # Session CRUD
│   │   ├── user_service.py      # User CRUD
│   │   ├── progress_service.py  # Analytics
│   │   └── pose_service.py      # Real-time pose processing
│   ├── schemas/                 # Pydantic request/response models
│   └── config/
│       └── settings.py          # Environment-based configuration
│
├── core/                        # Preserved AI core (do not modify unless necessary)
│   ├── angle_calculator.py      # 3-point angle math
│   ├── pose_detector.py         # MediaPipe Pose wrapper
│   ├── exercise_detector.py     # 10 exercises with rep counting + form detection
│   ├── database.py              # SQLite schema + CRUD functions
│   └── __init__.py              # Package exports
│
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── layout/          # Sidebar layout
│   │   │   └── common/          # Card, StatusBadge, StatCard, etc.
│   │   ├── pages/               # Application pages
│   │   │   ├── Landing.tsx      # Product landing page
│   │   │   ├── Login.tsx        # Login/signup
│   │   │   ├── Dashboard.tsx    # Patient dashboard
│   │   │   ├── Exercises.tsx    # Exercise library
│   │   │   ├── ExerciseDetails.tsx  # Individual exercise page
│   │   │   ├── ExerciseSetup.tsx    # Session setup
│   │   │   ├── LiveSession.tsx      # Real-time AI session
│   │   │   ├── SessionResults.tsx   # Post-session summary
│   │   │   ├── Progress.tsx         # Analytics & charts
│   │   │   └── Profile.tsx          # User profile
│   │   ├── services/            # API client + mock data
│   │   ├── hooks/               # React hooks
│   │   ├── types/               # TypeScript interfaces
│   │   ├── App.tsx              # Router setup
│   │   └── main.tsx             # Entry point
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── tests/
│   ├── test_angle_calculator.py
│   ├── test_exercise_detector.py
│   ├── test_database.py
│   ├── test_services.py
│   └── camerapose_test.py       # Camera test (requires display)
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── aiphysio.db                  # SQLite database
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/mandalayan-1829/physiotherapy-assistant.git
cd physiotherapy-assistant
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

### Backend
```bash
# From the project root
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Environment Variables

### Backend (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Debug mode |
| `DATABASE_PATH` | `./aiphysio.db` | SQLite database path |
| `CORS_ORIGINS` | `["http://localhost:5173"]` | Allowed origins |

### Frontend (`frontend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend API URL |
| `VITE_WS_BASE_URL` | `ws://localhost:8000` | WebSocket URL |
| `VITE_USE_MOCK_API` | `true` | Enable mock mode |

## Mock Mode

When `VITE_USE_MOCK_API=true`, the frontend uses mock data for all API calls. This allows development without a running backend.

Set to `false` to connect to the real FastAPI backend.

## API Endpoints

### Auth
- `POST /api/auth/signup` — Register new account
- `POST /api/auth/login` — Login

### Users
- `GET /api/users/me` — Get profile
- `PUT /api/users/me` — Update profile

### Exercises
- `GET /api/exercises` — List all exercises
- `GET /api/exercises/{id}` — Exercise details
- `GET /api/exercises/{id}/safety` — Safety check

### Sessions
- `POST /api/sessions` — Start session
- `PUT /api/sessions/{id}/end` — End session
- `GET /api/sessions` — List sessions
- `GET /api/sessions/recent` — Recent sessions

### Progress
- `GET /api/progress` — Full progress data
- `GET /api/progress/summary` — Quick summary

### Diet
- `POST /api/diet` — Add meal
- `GET /api/diet/today` — Today's meals
- `GET /api/diet` — All meals
- `DELETE /api/diet/{id}` — Delete entry

### Notes
- `POST /api/notes` — Add note
- `GET /api/notes` — List notes
- `DELETE /api/notes/{id}` — Delete note

### Doctors & Appointments
- `GET /api/doctors` — List doctors
- `GET /api/doctors/{id}` — Doctor details
- `POST /api/appointments` — Book appointment
- `GET /api/appointments` — List appointments

### WebSocket
- `WS /ws/session/{session_id}` — Real-time pose analysis

## WebSocket Protocol

### Client → Server
```json
{"type": "start", "exercise": "squat", "target_reps": 10}
{"type": "frame", "data": "<base64 JPEG>"}
{"type": "reset"}
{"type": "end"}
```

### Server → Client
```json
{"type": "analysis", "rep_count": 8, "angle": 92, "stage": "down", "form_status": "good", "feedback": "Keep going!"}
{"type": "completed", "rep_count": 10, "form_accuracy": 85}
{"type": "ended", "form_accuracy": 85}
```

## Supported Exercises

| Exercise | Type | Target | Difficulty |
|----------|------|--------|------------|
| Squat | Physio | Knee & Hip Rehab | Moderate |
| Shoulder Raises | Physio | Shoulder Rehab | Easy |
| Crossover Arm Stretch | Physio | Shoulder Mobility | Easy |
| Lateral Walks | Physio | Hip & Knee Rehab | Moderate |
| Lunges | Physio | Leg Strength | Moderate |
| Calf Raises | Physio | Ankle & Calf Rehab | Easy |
| Knee Raises | Physio | Hip Flexor & Core | Easy |
| Tree Pose | Yoga | Balance & Stability | Moderate |
| Warrior Pose | Yoga | Leg & Core Strength | Moderate |
| Cat-Cow Stretch | Yoga | Spine Flexibility | Easy |

## Testing
```bash
python -m pytest tests/ -v
```

## License

This project was built for educational and demonstration purposes.
