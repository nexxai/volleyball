#!/bin/bash

# 排球分析系統 - 快速啟動腳本

echo "🏐 排球分析系統啟動中..."

# 檢查Python環境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安裝，請先安裝Python3"
    exit 1
fi

# 創建虛擬環境
if [ ! -d "venv" ]; then
    echo "📦 創建虛擬環境..."
    python3 -m venv venv
fi

# 激活虛擬環境
echo "🔧 激活虛擬環境..."
source venv/bin/activate

# 安裝依賴
echo "📚 安裝Python依賴..."
pip install -r requirements.lock

# 創建必要的目錄
echo "📁 創建目錄結構..."
mkdir -p data/uploads data/results backend/data/uploads backend/data/results models static

# 複製現有模型 (如果存在)
echo "🤖 檢查並複製模型文件..."
if [ -f "../Volley_Vision/volleyball_capstone/models/VballNetV1_seq9_grayscale_148_h288_w512.onnx" ]; then
    cp "../Volley_Vision/volleyball_capstone/models/VballNetV1_seq9_grayscale_148_h288_w512.onnx" "models/"
    echo "✅ 球追蹤模型已複製"
fi

if [ -f "../Volley_Vision/volleyball_capstone/models/action_recognition_yv11m.pt" ]; then
    cp "../Volley_Vision/volleyball_capstone/models/action_recognition_yv11m.pt" "models/"
    echo "✅ 動作識別模型已複製"
fi

if [ -f "../Volley_Vision/volleyball_capstone/models/player_detection_yv8.pt" ]; then
    cp "../Volley_Vision/volleyball_capstone/models/player_detection_yv8.pt" "models/"
    echo "✅ 球員偵測模型已複製"
fi

# 檢查Redis是否運行
if ! pgrep -x "redis-server" > /dev/null; then
    echo "⚠️  Redis未運行，請先啟動Redis服務"
    echo "   在macOS上: brew services start redis"
    echo "   在Ubuntu上: sudo systemctl start redis"
    echo "   或使用Docker: docker run -d -p 6379:6379 redis:alpine"
fi

# 啟動後端服務
echo "🚀 啟動後端API服務..."
cd backend
python main.py &
BACKEND_PID=$!

# 等待後端啟動
sleep 3

# 啟動AI工作器
echo "🤖 啟動AI處理工作器..."
cd ../ai_core
python worker.py &
WORKER_PID=$!

# 等待工作器啟動
sleep 2

# 啟動前端 (如果Node.js可用)
if command -v npm &> /dev/null; then
    echo "🎨 啟動前端服務..."
    cd ../frontend
    
    # 安裝前端依賴
    if [ ! -d "node_modules" ]; then
        echo "📦 安裝前端依賴..."
        npm install
    fi
    
    # 啟動前端
    npm start &
    FRONTEND_PID=$!
    
    echo ""
    echo "🎉 所有服務已啟動!"
    echo "📱 前端: http://localhost:3000"
    echo "🔧 後端API: http://localhost:8000"
    echo "📚 API文檔: http://localhost:8000/docs"
    echo ""
    echo "按 Ctrl+C 停止所有服務"
    
    # 等待用戶中斷
    trap "echo '🛑 停止服務...'; kill $BACKEND_PID $WORKER_PID $FRONTEND_PID 2>/dev/null; exit" INT
    wait
    
else
    echo "⚠️  Node.js未安裝，跳過前端啟動"
    echo "   請手動安裝Node.js後運行: cd frontend && npm install && npm start"
    echo ""
    echo "🎉 後端服務已啟動!"
    echo "🔧 後端API: http://localhost:8000"
    echo "📚 API文檔: http://localhost:8000/docs"
    echo ""
    echo "按 Ctrl+C 停止服務"
    
    # 等待用戶中斷
    trap "echo '🛑 停止服務...'; kill $BACKEND_PID $WORKER_PID 2>/dev/null; exit" INT
    wait
fi
