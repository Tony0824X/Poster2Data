#!/usr/bin/env python3
"""
Poster2Data AI - Standalone Launcher
This is the entry point for PyInstaller.
Directly runs Streamlit without subprocess.
"""

import os
import sys
import socket
import webbrowser
import threading
import time

def get_app_dir():
    """Get the directory where the app files are located."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

def get_exe_dir():
    """Get the directory where the executable is located (for config.json)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def find_free_port(start=8501, end=8600):
    """Find a free port."""
    for port in range(start, end):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except OSError:
            continue
    return 8501

def open_browser(port, delay=4):
    """Open browser after a delay."""
    time.sleep(delay)
    url = f'http://localhost:{port}'
    print(f"🌐 Opening browser: {url}")
    webbrowser.open(url)

def main():
    app_dir = get_app_dir()
    exe_dir = get_exe_dir()
    app_path = os.path.join(app_dir, 'app.py')
    port = find_free_port()
    
    print("=" * 50)
    print("🚀 Poster2Data AI")
    print("=" * 50)
    print(f"📁 App directory: {app_dir}")
    print(f"📁 Config directory: {exe_dir}")
    print(f"🌐 URL: http://localhost:{port}")
    print("=" * 50)
    print("正在啟動，請稍候...")
    print("(關閉此視窗可停止應用程式)")
    print()
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = str(port)
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    os.environ['STREAMLIT_BROWSER_SERVER_ADDRESS'] = 'localhost'
    
    # Set config directory for app.py to find config.json
    os.environ['POSTER2DATA_CONFIG_DIR'] = exe_dir
    
    # Change to app directory so relative imports work
    os.chdir(app_dir)
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser, args=(port,))
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run streamlit directly
    try:
        from streamlit.web import cli as stcli
        sys.argv = [
            "streamlit", "run", app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
            "--global.developmentMode", "false",
        ]
        stcli.main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("按 Enter 鍵退出...")
        sys.exit(1)

if __name__ == '__main__':
    main()
