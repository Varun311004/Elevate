import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app

if __name__ == '__main__':
    try:
        app = create_app('development')
        print("App created successfully")
        # Run with threading to avoid blocking
        from werkzeug.serving import make_server
        import threading
        
        server = make_server('127.0.0.1', 5000, app, threaded=True)
        print("Server starting on http://127.0.0.1:5000")
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        print("Server is running. Press Ctrl+C to stop.")
        server_thread.join()
        
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()