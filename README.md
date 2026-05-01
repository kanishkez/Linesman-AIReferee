# ⚽ Football AI VAR — Video Assistant Referee

An AI-powered Video Assistant Referee that analyzes football match clips and makes foul/no-foul decisions using computer vision and large language models.

## 🏗️ Architecture

```
Video Upload → Stage 1: YOLOv8 Pose Detection → Stage 2: Gemini Video Analysis → Stage 3: LLM Rules Engine → VAR Decision
```

| Stage | Tool | What It Does |
|-------|------|-------------|
| **1** | YOLOv8x-pose + ByteTrack | Detects players, extracts 17-point pose skeletons, tracks IDs, computes velocities, finds contact zones |
| **2** | Gemini 2.5 Pro (video) | Watches the video natively — analyzes ball possession, contact point, challenge type, force, intent |
| **3** | Gemini 2.5 Pro (text) | Receives all evidence, applies FIFA Law 12, outputs structured foul decision with reasoning |

## 🚀 Quick Start

### 1. Get a Gemini API Key

Visit [Google AI Studio](https://aistudio.google.com/) and create a free API key.

### 2. Set up environment

```bash
# Create .env file
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 5. Open the UI

Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

### 6. Upload a clip

Upload a 5-30 second video clip of a football incident and wait for the analysis (~30-90 seconds).

## 📁 Project Structure

```
AI var/
├── app/
│   ├── main.py                  # FastAPI server
│   ├── pipeline.py              # 3-stage pipeline orchestrator
│   ├── yolo_analyzer.py         # Stage 1: YOLOv8 pose + tracking
│   ├── gemini_video_analyzer.py # Stage 2: Gemini video analysis
│   ├── rules_engine.py          # Stage 3: FIFA Law 12 rules engine
│   ├── models.py                # Pydantic data models
│   └── prompts.py               # All LLM prompts
├── static/
│   ├── index.html               # Web UI
│   ├── styles.css               # Dark VAR theme
│   └── app.js                   # Frontend logic
├── uploads/                     # Uploaded videos
├── outputs/                     # Annotated results
├── requirements.txt
├── .env.example
└── README.md
```

## 🎯 Tips for Best Results

- Use **5-30 second clips** focused on the incident
- **Broadcast quality** footage works better than phone recordings
- Include a few seconds **before and after** the incident for context
- Clear view of the challenge (not too far, not too close)

## 🔧 Configuration

You can adjust these in the source code:

- **YOLO model size**: In `pipeline.py`, change `yolov8x-pose.pt` to a smaller model (`yolov8m-pose.pt`) for faster processing
- **Contact distance threshold**: In `yolo_analyzer.py`, adjust `CONTACT_DISTANCE_THRESHOLD`
- **Sample rate**: In `pipeline.py`, adjust `sample_rate` (1 = every frame, higher = faster but less detailed)
- **Gemini model**: In `gemini_video_analyzer.py` and `rules_engine.py`, change the model name
