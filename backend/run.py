import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app

if __name__ == '__main__':
    try:
        env_name = os.environ.get('FLASK_ENV', 'development')
        port = int(os.environ.get('PORT', '5000'))

        app = create_app(env_name)
        print(f"App created successfully (env={env_name})")

        from werkzeug.serving import make_server
        import threading

        server = make_server('127.0.0.1', port, app, threaded=True)
        print(f"Server starting on http://127.0.0.1:{port}")

        # daemon=False: the main thread must stay alive as long as the server
        # thread is running. A daemon thread is killed the moment the main
        # thread exits, which caused the server to die immediately.
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = False
        server_thread.start()

        print("Server is running. Press Ctrl+C to stop.")
        server_thread.join()

    except KeyboardInterrupt:
        print("Server stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)