"""Local web UI: type a prompt, get sprites.

  spakie-web --run runs/base            # then open http://127.0.0.1:7860

Standard library only. The server is single-threaded on purpose: the model
lives on the main thread and requests are handled one at a time.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlx.core as mx
import numpy as np

from spakie.checkpoint import load_for_sampling
from spakie.diffusion.sampler import sample
from spakie.render import sprite_image

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>spakie</title>
<style>
  :root { --bg:#16171d; --panel:#1f2129; --line:#30333f; --text:#e8e9ee; --dim:#9296a6; --accent:#f2b84b; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:15px/1.5 ui-monospace, "SF Mono", Menlo, monospace; }
  main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: .04em; }
  h1 span { color: var(--accent); }
  .sub { color: var(--dim); margin: 0 0 24px; font-size: 13px; }
  form { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }
  .row { display:flex; gap:10px; }
  input, select, button { font: inherit; color: var(--text); background: var(--bg);
         border: 1px solid var(--line); border-radius: 7px; padding: 9px 11px; min-width: 0; }
  input:focus, select:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  #prompt { flex: 1; font-size: 16px; }
  button { background: var(--accent); color: #1a1405; border-color: var(--accent); font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  .opts { display:grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap:10px; margin-top:12px; }
  label { display:flex; flex-direction:column; gap:4px; font-size:12px; color: var(--dim); }
  #status { color: var(--dim); font-size: 13px; margin: 16px 2px 10px; min-height: 20px; }
  #status.err { color: #ff7a7a; }
  #grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(112px, 1fr)); gap:10px; }
  .cell { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px;
          display:block; line-height:0; }
  .cell:hover { border-color: var(--accent); }
  .cell img { width:100%; aspect-ratio:1; image-rendering: pixelated; }
  .hint { color: var(--dim); font-size: 12px; margin-top: 14px; }
  .light #grid .cell { background:#e9e9ee; }
</style>
</head>
<body>
<main>
  <h1><span>spakie</span> pixel-art generator</h1>
  <p class="sub" id="info"></p>
  <form id="form">
    <div class="row">
      <input id="prompt" placeholder="red dragon" autofocus autocomplete="off">
      <button id="go">Generate</button>
    </div>
    <div class="opts">
      <label>Sprites <input id="n" type="number" min="1" max="64" value="16"></label>
      <label>Guidance <input id="cfg" type="number" min="1" max="10" step="0.5" value="3"></label>
      <label>Steps <input id="steps" type="number" min="4" max="64" value="24"></label>
      <label>Order <select id="order"><option value="confidence">clean</option><option value="random">varied</option></select></label>
      <label>Seed <input id="seed" type="number" placeholder="random"></label>
      <label>Background <select id="bg"><option value="dark">dark</option><option value="light">light</option></select></label>
    </div>
  </form>
  <div id="status"></div>
  <div id="grid"></div>
  <p class="hint">Click a sprite to download it as a 16×16 transparent PNG.</p>
</main>
<script>
const $ = id => document.getElementById(id);
fetch('/info').then(r => r.json()).then(i => {
  $('info').textContent = `${i.params}M-parameter model · checkpoint step ${i.step} · short captions work best ("gold key with a gem")`;
});
$('bg').onchange = () => document.body.classList.toggle('light', $('bg').value === 'light');
$('form').onsubmit = async e => {
  e.preventDefault();
  const prompt = $('prompt').value.trim() || $('prompt').placeholder;
  const body = { prompt, n: +$('n').value, cfg: +$('cfg').value, steps: +$('steps').value,
                 order: $('order').value, seed: $('seed').value === '' ? null : +$('seed').value };
  $('go').disabled = true; $('status').className = ''; $('status').textContent = 'Generating…';
  try {
    const r = await fetch('/generate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    const slug = prompt.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'sprite';
    $('grid').replaceChildren(...d.sprites.map((src, i) => {
      const a = document.createElement('a');
      a.className = 'cell'; a.href = src; a.download = `${slug}_${d.seed}_${String(i).padStart(2, '0')}.png`;
      a.title = 'Download 16×16 PNG';
      const img = document.createElement('img'); img.src = src; img.alt = `${prompt} ${i + 1}`;
      a.append(img); return a;
    }));
    $('status').textContent = `"${prompt}" · ${d.sprites.length} sprites in ${d.seconds.toFixed(1)}s · seed ${d.seed}`;
  } catch (err) {
    $('status').className = 'err'; $('status').textContent = 'Error: ' + err.message;
  } finally { $('go').disabled = false; }
};
</script>
</body>
</html>
"""


def make_handler(model, tok, palette, state):
    side = model.cfg.side
    info = {"params": round(model.num_params() / 1e6, 1), "step": state["step"]}

    def generate(req: dict) -> dict:
        prompt = str(req.get("prompt", "")).strip()[:200]
        n = max(1, min(int(req.get("n", 16)), 64))
        steps = max(4, min(int(req.get("steps", 24)), 64))
        cfg = max(1.0, min(float(req.get("cfg", 3.0)), 10.0))
        order = req.get("order") if req.get("order") in ("confidence", "random") else "confidence"
        seed = req.get("seed")
        seed = random.randrange(2**31) if seed is None else int(seed) % 2**31

        mx.random.seed(seed)
        t0 = time.perf_counter()
        text = mx.array(tok.encode([prompt] * n))
        out = np.array(sample(model, text, mx.array(tok.empty(n)), steps=steps, cfg=cfg, order=order))
        sprites = []
        for i in range(n):
            buf = io.BytesIO()
            sprite_image(out[i : i + 1], palette, side).save(buf, "PNG")
            sprites.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        return {"sprites": sprites, "seed": seed, "seconds": time.perf_counter() - t0}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/info":
                self._send(200, json.dumps(info).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/generate":
                return self._send(404, b"not found", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 1 << 16)
                result = generate(json.loads(self.rfile.read(length) or b"{}"))
                self._send(200, json.dumps(result).encode(), "application/json")
            except Exception as e:  # report bad input or sampling errors to the page
                self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, fmt: str, *args) -> None:
            if self.command == "POST":
                super().log_message(fmt, *args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/base")
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    model, tok, palette, state = load_for_sampling(args.run, args.checkpoint)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(model, tok, palette, state))
    print(f"spakie web UI: http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
