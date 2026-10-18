"""Build the explanatory demo clip local_private/demo/forte_demo.mp4 (local only).

Content: captioned synthetic primitives (tools/make_synthetic.py), AGGREGATE outputs of a real
local API smoke (fresh uvicorn on a free port, throwaway key), and honest NO-GO result titles.
Every synthetic / drawn screen is labelled; nothing here is an app recording or dataset footage.
The API key is held only in this process and the server env: never drawn, printed or written.
usage: .venv-research/bin/python -m tools.demo_video   (needs cv2, numpy, Pillow)"""
import base64
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "local_private", "demo")
OUT = os.path.join(OUT_DIR, "forte_demo.mp4")
PORT = 8777
W, H, FPS = 1280, 720, 30
SYN = "СИНТЕТИКА / объясняющая визуализация"
FONT_CANDS = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Helvetica.ttc"]
MONO = "/System/Library/Fonts/Menlo.ttc"
FONT = next((p for p in FONT_CANDS if os.path.exists(p)), None)
BG, FG, DIM = (24, 22, 20), (240, 240, 240), (170, 170, 170)  # BGR
ACC, WARN, BAD, OK = (230, 170, 60), (60, 190, 240), (80, 80, 230), (110, 200, 110)
_fc = {}


def font(size, mono=False):
    k = (size, mono)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(MONO if mono else FONT, size)
    return _fc[k]


def canvas():
    return Image.new("RGB", (W, H), BG[::-1])


def rgb(c):
    return c[::-1]


def text(d, xy, s, size=30, col=FG, mono=False, anchor="la"):
    d.text(xy, s, font=font(size, mono), fill=rgb(col), anchor=anchor)


def wrap(d, s, size, maxw, mono=False):
    lines = []
    for para in s.split("\n"):
        cur = ""
        for wd in para.split(" "):
            t = (cur + " " + wd).strip()
            if d.textlength(t, font=font(size, mono)) <= maxw:
                cur = t
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
    return lines


def para(d, x, y, s, size=30, col=FG, maxw=1100, gap=1.35, mono=False):
    for ln in wrap(d, s, size, maxw, mono):
        text(d, (x, y), ln, size, col, mono)
        y += int(size * gap)
    return y


def header(d, title, tag=None):
    text(d, (50, 30), title, 34, ACC)
    d.line([(50, 88), (W - 50, 88)], fill=rgb(DIM), width=2)
    if tag:
        d.rectangle([W - 50 - 400, 24, W - 50, 70], outline=rgb(WARN), width=3)
        text(d, (W - 50 - 200, 47), tag, 17, WARN, anchor="mm")


def footer(d, s):
    d.rectangle([0, H - 70, W, H], fill=(10, 10, 10))
    para(d, 40, H - 62, s, 24, FG, W - 80, 1.25)


def to_bgr(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class Out:
    def __init__(self, path):
        self.w = None
        self.codec = None
        for cc in ("avc1", "mp4v"):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), FPS, (W, H))
            if w.isOpened():
                self.w, self.codec = w, cc
                break
        if not self.w:
            raise SystemExit("no usable mp4 codec in this OpenCV build")
        self.n = 0

    def hold(self, img, sec):
        fr = to_bgr(img) if isinstance(img, Image.Image) else img
        for _ in range(int(sec * FPS)):
            self.w.write(fr)
            self.n += 1

    def put(self, fr):
        self.w.write(fr)
        self.n += 1

    def fade(self, img, sec=0.5):  # simple fade-in from background
        fr = to_bgr(img) if isinstance(img, Image.Image) else img
        base = np.full_like(fr, BG)
        k = int(sec * FPS)
        for i in range(k):
            self.put(cv2.addWeighted(fr, (i + 1) / k, base, 1 - (i + 1) / k, 0))


# ------------------------------------------------------------------ local API smoke
def _req(method, path, key, body=None, ctype=None, timeout=30):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=body, method=method)
    r.add_header("X-API-Key", key)
    if ctype:
        r.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _multipart(fname, data):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def mask(obj):
    o = dict(obj)
    if "job_id" in o:
        o["job_id"] = o["job_id"][:8] + "…"
    return o


def run_smoke(key, clip, extra_env, tag):
    env = dict(os.environ, API_KEY=key, DATA_DIR=os.path.join(OUT_DIR, "jobs"), **extra_env)
    log = open(os.path.join(OUT_DIR, f"uvicorn_{tag}.log"), "wb")
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.asgi:app", "--host", "127.0.0.1",
                            "--port", str(PORT), "--log-level", "warning"], cwd=ROOT, env=env,
                           stdout=log, stderr=log)
    steps = []
    try:
        for _ in range(60):
            try:
                if _req("GET", "/api/v1/model", key, timeout=3)[0] == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("demo uvicorn did not start")
        body, ct = _multipart("synthetic_clip.mp4", open(clip, "rb").read())
        t0 = time.time()
        code, resp = _req("POST", "/api/v1/analyze", key, body, ct)
        steps.append((round(time.time() - t0, 1), code, mask(resp)))
        jid = resp.get("job_id")
        final = resp
        while jid and time.time() - t0 < 200:
            time.sleep(1.0)
            code, final = _req("GET", f"/api/v1/jobs/{jid}", key)
            if final.get("status") != "processing":
                steps.append((round(time.time() - t0, 1), code, mask(final)))
                break
        # a wrong-key probe (static code only)
        pc = _req("GET", f"/api/v1/jobs/{jid}", "x" * 12)[0] if jid else None
        return steps, final, pc
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except Exception:
            srv.kill()
        log.close()


# ------------------------------------------------------------------ scenes
def scene_title(o):
    im = canvas()
    d = ImageDraw.Draw(im)
    text(d, (W // 2, 170), "ForteBank Datathon26", 64, ACC, anchor="mm")
    text(d, (W // 2, 250), "Анализ видео с камеры кассира: демонстрационный ролик", 34, FG, anchor="mm")
    d.rectangle([90, 330, W - 90, 560], outline=rgb(WARN), width=3)
    y = para(d, 120, 350, "ЧЕСТНЫЙ СТАТУС", 30, WARN)
    para(d, 120, y + 6, "• Продуктовый режим fail-closed: без провалидированного правила задача завершается "
         "ошибкой no_validated_rule, а не «нарушений нет».\n"
         "• Провалидированного правила нет. Работающий инференс — не работающая детекция.\n"
         "• Ролик объясняющий: схемы и синтетика размечены, это не запись приложения.",
         26, FG, W - 240)
    footer(d, "Ролик собран локально (Python + OpenCV). Кадры датасета в ролик не входят.")
    o.fade(im)
    o.hold(im, 9.5)


def scene_pipeline(o):
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "Конвейер обработки", SYN)
    boxes = [("1. Загрузка", "POST\n/api/v1/analyze"), ("2. ASGI-guard", "ключ, лимиты,\nтаймауты"),
             ("3. Процесс", "жёсткий дедлайн,\nубийство по таймеру"),
             ("4. YOLOX", "детекция людей\n(Apache-2.0)"), ("5. Наблюдения", "присутствие, зона,\nактивность"),
             ("6. Шлюз правил", "только провалидир.\nправила"), ("7. Вердикт", "done / failed")]
    bw, bh, gap, x0, y0 = 140, 140, 22, 64, 170
    for i, (a, b) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        col = BAD if i == 5 else ACC
        d.rectangle([x, y0, x + bw, y0 + bh], outline=rgb(col), width=3)
        text(d, (x + bw // 2, y0 + 28), a, 16, col, anchor="mm")
        for j, ln in enumerate(b.split("\n")):
            text(d, (x + bw // 2, y0 + 68 + j * 24), ln, 14, FG, anchor="mm")
        if i < len(boxes) - 1:
            d.line([(x + bw, y0 + bh // 2), (x + bw + gap, y0 + bh // 2)], fill=rgb(FG), width=3)
    para(d, 60, 360, "Бюджеты времени:\n"
         "• 60 с — потолок на приём загрузки (upload timeout);\n"
         "• 230 с — бюджет задачи от момента отправки, включая очередь;\n"
         "• 60 + 230 = 290 с < 300 с: жёсткий лимит продукта «менее 5 минут».",
         28, FG, 1150)
    para(d, 60, 530, "Шлюз правил (красный) — ключевая защита: наблюдения всегда считаются, но вердикт "
         "могут дать только провалидированные на размеченном holdout правила. Иначе — failed / "
         "no_validated_rule.", 26, WARN, 1150)
    footer(d, SYN + ": блок-схема, не снимок экрана приложения.")
    o.fade(im)
    o.hold(im, 17)


def gt_box(t, absent):
    if absent[0] <= t < absent[1]:
        return None
    cx = 320 + int(10 * np.sin(t))
    return (cx - 35, 92, cx + 35, 260)  # head top (120-28) .. body bottom, as drawn by the generator


def scene_synthetic(o, clip, absent):
    cap = cv2.VideoCapture(clip)
    n = 0
    sc = 1.5
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = n / FPS
        n += 1
        fr = cv2.resize(fr, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        b = gt_box(t, absent)
        if b:
            cv2.rectangle(fr, tuple(int(v * sc) for v in b[:2]), tuple(int(v * sc) for v in b[2:]), (0, 255, 255), 2)
        pil = canvas()
        pil.paste(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), (160, 100))
        d = ImageDraw.Draw(pil)
        header(d, "Синтетический клип (make_synthetic)", "СИНТЕТИКА — не датасет")
        away = absent[0] <= t < absent[1]
        text(d, (170, 108), f"t = {t:4.1f} c   " + ("оператор ОТСУТСТВУЕТ (по построению)" if away else
                                                         "оператор в кадре"), 24, BAD if away else OK)
        footer(d, "Жёлтая рамка — эталонная (ground truth) разметка самого генератора, НЕ результат "
                  "детектора. Детектор на нарисованных фигурах не имеет смысла.")
        o.put(to_bgr(pil))
    cap.release()


def scene_json(o, title, tag, steps, notes, extra=None):
    """Aggregate API output as text (real JSON; job_id masked to 8 chars)."""
    for upto in range(1, len(steps) + 1):
        im = canvas()
        d = ImageDraw.Draw(im)
        header(d, title, tag)
        y = 110
        for (t, code, js) in steps[:upto]:
            line = f"t+{t:5.1f}s  HTTP {code}  " + json.dumps(js, ensure_ascii=False)
            y = para(d, 50, y, line, 22, OK if js.get("status") == "done" else (BAD if js.get("status") == "failed" else WARN),
                     W - 100, 1.3, mono=True) + 8
        if upto == len(steps) and extra:
            extra(d, y)
        footer(d, notes)
        o.hold(im, 4.0 if upto < len(steps) else 7.0)


def scene_demo_result(o, steps, final):
    viol = (final.get("violations") or [{}])[0]
    frames = viol.get("evidence_frames") or []
    desc = viol.get("description", "")
    summary = [("status", final.get("status")), ("verdict", final.get("verdict")),
               ("violations[0].type", viol.get("type")), ("timecode_sec", viol.get("timecode_sec")),
               ("marker «НЕ валидировано» в description", "present" if "НЕ валидировано" in desc else "ABSENT")]
    imgs = []
    for f in frames[:3]:
        try:
            a = cv2.imdecode(np.frombuffer(base64.b64decode(f["image_base64"]), np.uint8), cv2.IMREAD_COLOR)
            imgs.append((f.get("timestamp_sec"), a))
        except Exception:
            pass
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "Демо-режим: непроверенное правило", SYN)
    d.rectangle([50, 100, W - 50, 160], fill=rgb(BAD))
    text(d, (W // 2, 130), "ДЕМО-РЕЖИМ, не продуктовый verdict — НЕ валидировано", 30, (255, 255, 255), anchor="mm")
    y = 180
    for k, v in summary:
        text(d, (60, y), f"{k}: {v}", 22, FG, mono=True)
        y += 30
    y = para(d, 60, y + 4, "description: " + desc[:230] + ("…" if len(desc) > 230 else ""), 18, DIM, W - 120, 1.3)
    x = 50
    for ts, a in imgs:
        a = cv2.resize(a, (380, int(380 * a.shape[0] / a.shape[1])))
        im.paste(Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)), (x, 380))
        d.text((x, 380 + a.shape[0] + 4), f"evidence JPEG, t={ts} с (синтетика)", font=font(18), fill=rgb(DIM))
        x += 400
    footer(d, "Это синтетический детектор + явное включение непроверенного правила "
              "(ALLOW_UNVALIDATED_RULES / ALLOW_SYNTHETIC_DETECTOR). Кадры — нарисованные, не из датасета.")
    o.fade(im)
    o.hold(im, 12)


RESULTS = [
    ("Этап 1 — baseline на holdout",
     "24/24 клипа: no_validated_rule\nисследовательский recall 0.0, ложных срабатываний 0\n"
     "Вывод: без валидированного правила продукт честно отказывается выдавать вердикт."),
    ("Этап 2 — train-проба признаков",
     "Лучший AUC 0.750 против перестановочного p50 0.833 / p95 0.950\n"
     "Вложенный F1 0.308 против 0.455\nВывод: сигнал не отличим от шума."),
    ("Этап 3 — VLM: NO-GO",
     "Ложное «да» 0.56–0.81 на заведомо пустых синтетических кадрах\n"
     "Вывод: языковая модель не годится как детектор отсутствия."),
    ("Этап 4 — NO-GO",
     "F1 0.400 / hit-rate 0.200\nВывод: порог приёмки не достигнут."),
    ("Этап 5 — NO-GO",
     "SigLIP2 (замороженный энкодер) + временная модель + логистическая регрессия\n"
     "клип F1 0.30 / event F1 0.0 / hit 0.20\n"
     "против always-positive 0.4545 и duration-only 0.60\n"
     "n_perm 1000, 718.8 с, 17/17 блоков выполнено\n"
     "Вывод: хуже тривиальных базлайнов."),
]


def scene_results(o):
    for i, (t, body) in enumerate(RESULTS):
        im = canvas()
        d = ImageDraw.Draw(im)
        header(d, f"Измеренные результаты ({i + 1}/5)", "Реальные агрегаты, без кадров")
        text(d, (60, 150), t, 44, BAD if "NO-GO" in t else ACC)
        para(d, 60, 250, body, 32, FG, 1150, 1.5)
        footer(d, "Только агрегированные метрики. Ни кадров, ни имён, ни путей датасета.")
        o.fade(im, 0.4)
        o.hold(im, 7.5)


def scene_close(o):
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "Итог: лицензии, приватность, статус")
    para(d, 60, 130, "Лицензии\n• YOLOX — Apache-2.0 (поставляется).\n• Исследовательский стек (SigLIP2 и др.) "
         "не поставляется.\n\nПриватность\n• Датасет обрабатывался только локально; в облако ничего не уходило.\n\n"
         "Статус\n• Holdout закрыт; независимая валидация не заявляется.\n\n"
         "Подробности: docs/SUBMISSION.md, docs/RELEASE_PLAN.md, docs/LICENSES.md", 30, FG, 1150, 1.4)
    footer(d, "Спасибо. Ролик объясняющий: схемы и синтетика размечены; кадры датасета не использовались.")
    o.fade(im)
    o.hold(im, 14)


def main():
    if not FONT:
        raise SystemExit("no Cyrillic-capable font found (install Arial Unicode)")
    os.makedirs(OUT_DIR, exist_ok=True)
    key = os.environ.get("DEMO_API_KEY") or secrets.token_urlsafe(24)  # never printed/written
    sys.path.insert(0, ROOT)
    from tools import make_synthetic
    absent = (3.0, 11.0)
    clip = os.path.join(OUT_DIR, "synthetic_clip.mp4")
    make_synthetic.make(clip, dur=12, absent=absent)

    prod_steps, prod_final, pc_wrong = run_smoke(key, clip, {}, "prod")
    rules = json.load(open(os.path.join(ROOT, "app", "rules_default.json")))
    r = rules["rules"]["operator_zone_unattended_with_counter_activity"]
    r["enabled"] = True
    r["min_sec"] = 5.0
    rp = os.path.join(OUT_DIR, "rules_demo_scratch.json")
    json.dump(rules, open(rp, "w"))
    demo_steps, demo_final, _ = run_smoke(
        key, clip, {"ALLOW_UNVALIDATED_RULES": "1", "ALLOW_SYNTHETIC_DETECTOR": "1",
                    "SYNTHETIC_DETECTOR": "1", "RULESET": rp}, "demo")

    o = Out(OUT)
    scene_title(o)
    scene_pipeline(o)
    scene_synthetic(o, clip, absent)
    scene_json(o, "Smoke API: продуктовый режим", "Реальный вывод API, job_id скрыт",
               prod_steps, "Локальный сервер на 127.0.0.1, одноразовый ключ (не показан). Показан агрегированный "
               f"JSON. Запрос с неверным ключом: HTTP {pc_wrong}.")
    scene_json(o, "Smoke API: ДЕМО-РЕЖИМ", "Реальный вывод, job_id скрыт",
               [(t, c, {k: v for k, v in js.items() if k in ("job_id", "status", "verdict", "error")})
                for t, c, js in demo_steps],
               "ДЕМО-РЕЖИМ: синтетический детектор + непроверенное правило. НЕ валидировано.")
    if demo_final.get("status") == "done":
        scene_demo_result(o, demo_steps, demo_final)
    scene_results(o)
    scene_close(o)
    o.w.release()
    info = {"codec": o.codec, "frames": o.n, "sec": round(o.n / FPS, 1), "font": FONT,
            "prod_status": [s[2].get("status") for s in prod_steps], "prod_error": prod_final.get("error"),
            "demo_status": [s[2].get("status") for s in demo_steps], "demo_verdict": demo_final.get("verdict"),
            "demo_error": demo_final.get("error"), "wrong_key_http": pc_wrong}
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
