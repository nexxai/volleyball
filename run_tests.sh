#!/bin/bash

# 排球分析系統 - 測試運行腳本

echo "🧪 排球分析系統測試套件"
echo "=========================="
echo ""

# 檢查是否在虛擬環境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告: 未檢測到虛擬環境"
    echo "   建議先激活虛擬環境: source venv/bin/activate"
    echo ""
fi

# 檢查 pytest 是否安裝
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest 未安裝"
    echo "   請運行: pip install -r requirements-dev.lock"
    exit 1
fi

# 解析參數
COVERAGE=false
HTML_REPORT=false
VERBOSE=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --html|-h)
            HTML_REPORT=true
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --test|-t)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        *)
            echo "未知參數: $1"
            echo "用法: $0 [--coverage|-c] [--html|-h] [--verbose|-v] [--test|-t <test_path>]"
            exit 1
            ;;
    esac
done

# 構建 pytest 命令
PYTEST_CMD="pytest tests/"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=backend --cov=ai_core --cov-report=term-missing"
    
    if [ "$HTML_REPORT" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov-report=html"
        echo "📊 將生成 HTML 覆蓋率報告: htmlcov/index.html"
    fi
fi

if [ -n "$SPECIFIC_TEST" ]; then
    PYTEST_CMD="$PYTEST_CMD $SPECIFIC_TEST"
fi

echo "🚀 運行命令: $PYTEST_CMD"
echo ""

# 運行測試
eval $PYTEST_CMD

EXIT_CODE=$?

echo ""
echo "=========================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 測試完成！"
else
    echo "❌ 測試失敗（退出碼: $EXIT_CODE）"
fi

if [ "$HTML_REPORT" = true ] && [ -d "htmlcov" ]; then
    echo ""
    echo "📊 查看 HTML 覆蓋率報告:"
    echo "   open htmlcov/index.html"
fi

exit $EXIT_CODE

