import http.server
import socketserver
import webbrowser
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        
    def end_headers(self):
        # Disable caching for CSV and JSON files so the dashboard updates live
        if self.path.endswith('.csv') or self.path.endswith('.json'):
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def start_server():
    print(f"Starting BrainQuant Dashboard Server...")
    print(f"Serving files from {DIRECTORY}")
    print(f"Local URL: http://localhost:{PORT}")
    
    # Try to open the browser automatically
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
        
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard Server stopped.")

if __name__ == "__main__":
    start_server()
