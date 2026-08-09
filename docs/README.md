# 文檔目錄

本目錄包含排球分析系統的所有技術文檔。

## 📁 文件夾結構

```
docs/
├── README.md                    # 本文件（文檔索引）
├── architecture/                # 架構和設計文檔
│   └── ARCHITECTURE.md         # 系統架構文檔
├── testing/                    # 測試相關文檔
│   ├── BACKEND_TEST_COVERAGE_IMPROVEMENTS.md
│   ├── COVERAGE_REPORT.md
│   ├── PYTHON_MODERNIZATION.md
│   ├── TEST_COVERAGE_IMPROVEMENTS.md
│   └── VIDEO_PROCESSING_BENCHMARK.md
├── features/                   # 功能實現文檔
│   ├── JERSEY_NUMBER_DETECTION_COMPARISON.md
│   └── OCR_WORKFLOW.md
├── changelog/                  # 開發日誌
│   └── SESSION_CHANGELOG_2025-12-09.md
└── GEMINI_CANVAS_SLIDES_PROMPT.md  # Gemini 簡報生成提示詞
```

## 📚 文檔說明

### Architecture（架構文檔）
- **ARCHITECTURE.md**: 系統架構設計，包括前端、後端、AI 核心的技術棧和組件關係

### Testing（測試文檔）
- **BACKEND_TEST_COVERAGE_IMPROVEMENTS.md**: 後端測試覆蓋率改進記錄
- **COVERAGE_REPORT.md**: 當前測試覆蓋率報告和改進建議
- **PYTHON_MODERNIZATION.md**: Python 3.14、現代依賴版本、鎖檔與驗證結果
- **TEST_COVERAGE_IMPROVEMENTS.md**: 測試覆蓋率改進總結
- **VIDEO_PROCESSING_BENCHMARK.md**: 影片分析 CPU 效能基準、分析結果與優化實驗記錄

### Features（功能文檔）
- **JERSEY_NUMBER_DETECTION_COMPARISON.md**: 球衣號碼檢測方法比較（YOLO vs EasyOCR）
- **OCR_WORKFLOW.md**: OCR 工作流程詳細說明

### Changelog（開發日誌）
- **SESSION_CHANGELOG_2025-12-09.md**: 開發會話變更日誌，記錄每次開發的改進和修復

### Presentation（簡報生成）
- **GEMINI_CANVAS_SLIDES_PROMPT.md**: 用於生成技術簡報的 Gemini Canvas 提示詞，包含完整的 15 張幻燈片結構和技術細節

## 🔍 快速查找

### 想了解系統架構？
→ 查看 `architecture/ARCHITECTURE.md`

### 想了解測試覆蓋率？
→ 查看 `testing/COVERAGE_REPORT.md`

### 想了解球衣號碼檢測實現？
→ 查看 `features/JERSEY_NUMBER_DETECTION_COMPARISON.md` 和 `features/OCR_WORKFLOW.md`

### 想查看開發歷史？
→ 查看 `changelog/SESSION_CHANGELOG_2025-12-09.md`

### 想生成技術簡報？
→ 查看 `GEMINI_CANVAS_SLIDES_PROMPT.md`，複製提示詞到 Gemini Canvas 生成簡報

## 📝 文檔維護指南

### 添加新文檔
1. **架構文檔** → 放入 `architecture/`
2. **測試文檔** → 放入 `testing/`
3. **功能文檔** → 放入 `features/`
4. **開發日誌** → 放入 `changelog/`
5. **簡報/提示詞** → 放入 `docs/` 根目錄

### 命名規範
- 使用大寫字母開頭：`FEATURE_NAME.md`
- 使用下劃線分隔單詞：`TEST_COVERAGE.md`
- 日期格式：`YYYY-MM-DD`（如 `SESSION_CHANGELOG_2025-12-09.md`）

### 更新 README
當添加新文檔時，請更新本 README.md 文件，添加文檔說明。
