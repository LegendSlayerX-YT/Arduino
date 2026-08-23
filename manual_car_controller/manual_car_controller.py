"""Simple keyboard-controlled web UI for the motor_car Arduino.

Serves a page that listens for arrow keys / 's' and forwards them to the
motor_car's HTTP endpoints (/up, /down, /shift_left, /shift_right, /stop).
Requests are proxied through this server (rather than fetched directly from
the browser) since the Arduino's web server doesn't send CORS headers.

Pure standard library, no dependencies to install.

Usage:
    python manual_car_controller.py --car-host 192.168.1.60
    python manual_car_controller.py --car-host 192.168.1.60 --car-port 80 --port 5000
"""

import argparse
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAR_HOST = "192.168.0.26"
CAR_PORT = "80"

KEY_TO_ENDPOINT = {
    "up": "up",
    "down": "down",
    "left": "shift_left",
    "right": "shift_right",
    "stop": "stop",
}

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Manual Car Controller</title>
<style>
  body {
    font-family: sans-serif;
    text-align: center;
    margin-top: 4rem;
  }
  #status {
    font-size: 1.5rem;
    margin-top: 2rem;
  }
  #keys {
    font-size: 1rem;
    color: #666;
  }
</style>
</head>
<body>
  <h1>Manual Car Controller</h1>
  <p id="keys">Use Arrow Up / Down / Left / Right, or S to stop.</p>
  <div id="status">Ready</div>

  <script>
    const KEY_MAP = {
      "ArrowUp": "up",
      "ArrowDown": "down",
      "ArrowLeft": "left",
      "ArrowRight": "right",
      "s": "stop",
      "S": "stop",
    };

    const statusEl = document.getElementById("status");

    document.addEventListener("keydown", (event) => {
      const action = KEY_MAP[event.key];
      if (!action) return;
      event.preventDefault();

      fetch(`/trigger/${action}`)
        .then((res) => res.text())
        .then((text) => { statusEl.textContent = text; })
        .catch((err) => { statusEl.textContent = `Error: ${err}`; });
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="text/plain"):
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, content_type="text/html")
            return

        if self.path.startswith("/trigger/"):
            action = self.path[len("/trigger/"):]
            endpoint = KEY_TO_ENDPOINT.get(action)
            if endpoint is None:
                self._send(400, f"Unknown action: {action}")
                return

            url = f"http://{CAR_HOST}:{CAR_PORT}/{endpoint}"
            try:
                urllib.request.urlopen(url, timeout=2)
                self._send(200, f"Sent /{endpoint}")
            except urllib.error.URLError as exc:
                self._send(502, f"Failed to reach car: {exc}")
            return

        self._send(404, "Not found")

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyboard web controller for motor_car")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve this web page on (default 5000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving on http://localhost:{args.port} -> car at {CAR_HOST}:{CAR_PORT}")
    server.serve_forever()
