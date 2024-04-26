import http.server
import socketserver

# Set the port number you want to use
PORT = 4000

# This needed so outside requests can access data
# Without this there are CORS (Cross-Origin Resource Sharing) access errors
class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def send_my_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        http.server.SimpleHTTPRequestHandler.end_headers(self)
    def end_headers(self):
        self.send_my_headers()

# create a local server with the specified port and handler
with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    # Start the server
    httpd.serve_forever()
