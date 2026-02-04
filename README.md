# Poster2Data AI

活動海報資料擷取工具 - 使用 AI 自動從海報圖片/PDF 中提取活動資訊。

## 功能特點

- 🖼️ **Vision AI** - 直接讀取圖片，準確識別藝術字體
- 📄 **PDF 支援** - 自動處理 PDF 檔案
- 🇹🇼 **繁體中文優化** - 專為繁體中文活動海報設計
- 📊 **結構化輸出** - 自動提取：活動名稱、日期、時間、地點
- 💾 **CSV 匯出** - 一鍵匯出所有資料

## 系統需求

### 獨立執行檔版本（推薦）
- macOS 10.13 或更高版本
- 無需安裝 Python

### 開發版本
- Python 3.8+
- 依賴套件見 `requirements.txt`

## 安裝與使用

### 方法 1：獨立執行檔（給非技術用戶）

1. 從 [Releases](https://github.com/Tony0824X/Poster2Data/releases) 下載 `Poster2Data_Final.zip`
2. 解壓縮後雙擊 `Poster2Data` 執行
3. 瀏覽器會自動開啟

## API Key 設定

應用程式需要設定 API Key 才能使用：

1. **OpenRouter API Key**（推薦，用於 Vision AI）
   - 免費註冊：https://openrouter.ai/keys
   
2. **DeepSeek API Key**（備用，用於 OCR 模式）
   - 註冊：https://platform.deepseek.com/api_keys

## 技術架構

- **前端**：Streamlit
- **Vision AI**：OpenRouter (Gemini 2.0 Flash)
- **OCR 備用**：Tesseract + DeepSeek
- **PDF 處理**：PyMuPDF, pdfplumber

## License

MIT License
