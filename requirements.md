# Project Name: Poster2Data AI Dashboard
# Type: One-Day Sprint Deliverable
# Target Output: Single-file Python Script (app.py) compatible with PyInstaller

## 1. Tech Stack & Framework
- **Frontend/Backend:** Streamlit (Python)
- **Data Handling:** Pandas, JSON, OS
- **UI Styling:** Custom CSS (to match `UI.png`)
- **Deployment:** Must run locally as a standalone script (.exe ready).

## 2. Visual Reference
- Please refer to `UI.png` in the current directory.
- **Key Visual Elements:**
  - **Header:** "Poster2Data AI" with a "System Operational" badge.
    - *Logic:* Badge turns Green if API Key is found, Red/Yellow if missing.
  - **Layout:** 30% Left Sidebar / 70% Right Panel.
  - **Color:** "Futu Blue" (#2A55E5) as primary accent.

## 3. Core Features & Logic

### A. Security & Configuration (NEW! Critical)
- **API Key Management:**
  - The app must NOT include hardcoded API keys.
  - On startup, check for a local file: `config.json`.
  - **Scenario 1 (Key Found):** Load the key. Show "✅ AI System Online" in Sidebar.
  - **Scenario 2 (Key Missing):** Show a warning in Sidebar. Provide a `st.text_input` (type="password") for the user to enter their DeepSeek API Key.
  - **Action:** When user clicks "Save Key", write it to `config.json` and refresh the app.

### B. Left Panel: Upload Zone
- **Component:** Streamlit File Uploader (JPG, PNG, PDF).
- **Visual Feedback:**
  - Show status badges: "OCR Scanning..." (Yellow) -> "AI Processing" -> "Done" (Green).

### C. Right Panel: Data Preview
- **Component:** `st.data_editor` (Editable).
- **Data Columns:** `Event Name`, `Date`, `Venue`, `Description`, `Confidence Score` (Visual bar), `Actions`.
- **Processing Logic (The "Demo Mode"):**
  - Even if the API Key is present, for this specific deliverable, **use Mock Data** to guarantee a smooth demo.
  - *Code Structure:* Create a placeholder function `process_with_deepseek(api_key, file)` but verify it returns the Mock Data for now to avoid runtime errors during the pitch.

### D. Top Right: Export
- **Button:** "Export to CSV".
- **Action:** Download the edited dataframe.

## 4. Specific Implementation Instructions for Cursor
- **Config Handling:** Use `os.path.exists` to check for `config.json`. Handle errors gracefully.
- **CSS Injection:** Use `st.markdown` to hide Streamlit's default hamburger menu and footer. Make it look like a native Windows app.
- **Single File:** Keep all logic (including the config loading function) in `app.py`.

## 5. Definition of Done
- App launches without crashing even if `config.json` is missing.
- User can input an API Key in the sidebar, and it persists after restart.
- UI matches `UI.png` pixel-perfect.
- Dragging a file shows the "Loading" spinner and then populates the Mock Data.