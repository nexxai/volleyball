# Volleyball AI Analysis System | 排球 AI 分析系統

A modern web application for volleyball video analysis using AI-powered ball tracking, player detection, and action recognition.

一個使用 AI 技術進行排球影片分析的現代化網頁應用程式，包含球追蹤、球員偵測和動作識別功能。

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue)
![React](https://img.shields.io/badge/React-18.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[![Website](https://img.shields.io/badge/Website-VolleyVision%20AI-0033A0)](https://dl-volleyball-analysis.github.io/volleyvision-website/)

---

## Features | 功能特色

- **Ball Tracking** - VballNet ONNX model for trajectory detection | 使用 VballNet 進行球軌跡追蹤
- **Action Recognition** - YOLOv11 for action classification (spike, set, receive, serve, block) | 動作識別（扣球、舉球、接球、發球、攔網）
- **Player Detection** - YOLOv8 + Norfair for multi-object tracking | 球員偵測與追蹤
- **Interactive Timeline** - Drag-to-seek with event markers | 互動式時間軸
- **Player Statistics** - Individual action lists and statistics | 球員統計資料
- **Modern UI** - Responsive design with Tailwind CSS | 響應式設計

## Tech Stack | 技術棧

### Backend | 後端
- FastAPI + Uvicorn
- SQLite / PostgreSQL
- Celery + Redis (optional)

### Frontend | 前端
- React 18 + TypeScript
- Tailwind CSS
- React Router
- Axios

### AI/ML
- PyTorch + Ultralytics YOLO
- OpenCV
- ONNX Runtime
- Norfair (multi-object tracking)

## Quick Start | 快速開始

```bash
# Clone | 複製
git clone https://github.com/DL-Volleyball-Analysis/volleyball-analysis-webapp.git
cd volleyball-analysis-webapp

# Backend setup | 後端設定
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.lock

# Frontend setup | 前端設定
cd frontend && npm install && cd ..

# Create directories | 建立目錄
mkdir -p data/uploads data/results

# Start backend | 啟動後端
cd backend && uvicorn main:app --reload

# Start frontend (new terminal) | 啟動前端（新終端）
cd frontend && npm start
```

## Access | 存取

- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Project Structure | 專案結構

```
volleyball_analysis_webapp/
├── backend/           # FastAPI API
├── frontend/          # React + TypeScript
├── ai_core/           # AI processing (processor.py, worker.py)
├── models/            # Pre-trained models (gitignored)
├── tests/             # Test suite
└── docs/              # Documentation
```

## API Endpoints | API 端點

| Method | Endpoint | Description | 說明 |
|--------|----------|-------------|------|
| GET | `/videos` | List videos | 影片列表 |
| POST | `/upload` | Upload video | 上傳影片 |
| POST | `/analyze/{id}` | Start analysis | 開始分析 |
| GET | `/results/{id}` | Get results | 取得結果 |

## Models Required | 所需模型

Place in `models/` directory:
- `VballNetV1_seq9_grayscale_148_h288_w512.onnx`
- `action_recognition_yv11m.pt`
- `player_detection_yv8.pt`

## Testing | 測試

```bash
# Run tests | 執行測試
pytest tests/

# With coverage | 含覆蓋率
pytest tests/ --cov=backend --cov=ai_core --cov-report=html
```

Test Coverage | 測試覆蓋率: 59%

## License | 授權

MIT License

---

## Related Projects | 相關專案

| Project | Description |
|---------|-------------|
| [volleyvision-website](https://github.com/DL-Volleyball-Analysis/volleyvision-website) | Landing page website |
| [volleyball-court-detection](https://github.com/DL-Volleyball-Analysis/volleyball-court-detection) | Court detection and ball landing |

---

*Part of [DL-Volleyball-Analysis](https://github.com/DL-Volleyball-Analysis) - Senior Capstone Project*

*National Taiwan Ocean University - Department of Computer Science*
