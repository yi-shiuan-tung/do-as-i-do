"""Serve the fitting-process viewer and read-only reconstruction artifacts locally."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8083);a=p.parse_args()
root=Path(__file__).resolve().parents[1]/'known_scene';server=ThreadingHTTPServer(('127.0.0.1',a.port),partial(SimpleHTTPRequestHandler,directory=str(root)))
print(f'Open http://localhost:{a.port}/fitting_process/ (forward this port over SSH)',flush=True)
try:server.serve_forever()
except KeyboardInterrupt:pass
finally:server.server_close()
