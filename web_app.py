"""
Web Dashboard - Cantautore Digitale
====================================
Dashboard web per generare canzoni, vedere il progresso in tempo reale,
gestire album e voci, ascoltare e scaricare — tutto dal browser.

Avvia con:  python web_app.py
Poi apri:   http://localhost:5000
"""
import json
import os
import queue
import shutil
import threading
import traceback
import uuid
import time
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, send_file,
    Response, stream_with_context
)

app = Flask(__name__)

PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
VOICES_DIR = PROJECT_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)


# === JOB SYSTEM ===

jobs = {}
job_queue = queue.Queue()
progress_subscribers = {}


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def create_job(job_type, params):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id, "type": job_type, "params": params,
        "status": JobStatus.QUEUED, "progress": 0,
        "current_step": "In coda...", "steps": [],
        "created": datetime.now().isoformat(),
        "started": None, "completed": None,
        "result": None, "error": None,
    }
    job_queue.put(job_id)
    return job_id


def update_progress(job_id, progress, step, detail=""):
    if job_id in jobs:
        jobs[job_id]["progress"] = progress
        jobs[job_id]["current_step"] = step
        jobs[job_id]["steps"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "step": step, "detail": detail, "progress": progress,
        })
    if job_id in progress_subscribers:
        event_data = json.dumps({
            "job_id": job_id, "progress": progress, "step": step,
            "detail": detail,
            "status": jobs[job_id]["status"] if job_id in jobs else "unknown",
        })
        for q in progress_subscribers[job_id]:
            q.put(event_data)


def run_song_generation(job_id, params):
    try:
        jobs[job_id]["status"] = JobStatus.RUNNING
        jobs[job_id]["started"] = datetime.now().isoformat()

        tema = params["tema"]
        titolo = params.get("titolo")
        voice_name = params.get("voce")
        export_stems = params.get("export_stems", False)

        update_progress(job_id, 5, "Generazione testo", "Gemini sta scrivendo...")

        from config import OUTPUT_DIR as CFG_OUTPUT, TEMP_DIR
        from cantautore import genera_testo, genera_musica, separa_vocals, mix_finale, estendi_con_strumentale
        from artist_brain import ArtistBrain

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        song_temp = TEMP_DIR / timestamp
        song_temp.mkdir(exist_ok=True)

        update_progress(job_id, 10, "Generazione testo", "Bozza in corso...")
        song_data = genera_testo(tema, titolo)
        song_title = song_data["titolo"]
        update_progress(job_id, 20, "Testo completato",
                        f'"{song_title}" - {song_data.get("mood","")}, {song_data.get("bpm","")} BPM')

        testo_file = song_temp / "testo.json"
        with open(testo_file, "w", encoding="utf-8") as f:
            json.dump(song_data, f, ensure_ascii=False, indent=2)

        brain = ArtistBrain("cantautore_digitale", PROJECT_DIR)
        brain.digest_song(song_data, tema)

        update_progress(job_id, 25, "Generazione musica", "Lyria 3 Pro sta componendo...")
        raw_song = song_temp / "raw_song.wav"
        genera_musica(song_data, raw_song)
        update_progress(job_id, 55, "Musica generata", "Canzone grezza pronta")

        update_progress(job_id, 58, "Separazione stems", "Demucs sta separando voce e strumenti...")
        separated_dir = song_temp / "separated"
        vocals_path, instrumental_path = separa_vocals(raw_song, separated_dir)
        update_progress(job_id, 70, "Stems separati", "Vocals e strumentali pronti")

        try:
            from config import get_voice_reference
            voice_ref = get_voice_reference(voice_name)
            update_progress(job_id, 72, "Voice conversion",
                            f"Conversione voce ({voice_ref.parent.name})...")
            from cantautore import converti_voce
            converted_path = song_temp / "vocals_converted.wav"
            converti_voce(vocals_path, converted_path, voice_name)
            vocals_path = converted_path
            update_progress(job_id, 85, "Voce convertita", "Voce applicata")
        except FileNotFoundError:
            update_progress(job_id, 85, "Voice conversion saltata", "Nessuna voce di riferimento")

        update_progress(job_id, 87, "Mix finale", "Mixaggio in corso...")
        ver_final = song_temp / "final.wav"
        mix_finale(instrumental_path, vocals_path, ver_final)
        update_progress(job_id, 92, "Mix completato", "Mastering applicato")

        update_progress(job_id, 93, "Estensione", "Intro e outro strumentale...")
        ver_extended = song_temp / "extended.wav"
        estendi_con_strumentale(ver_final, instrumental_path, ver_extended)
        update_progress(job_id, 96, "Canzone estesa", "Completa")

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in song_title)
        final_path = CFG_OUTPUT / f"{safe_title}_{timestamp}.wav"
        shutil.copy2(str(ver_extended), str(final_path))
        testo_output = CFG_OUTPUT / f"{safe_title}_{timestamp}_testo.json"
        shutil.copy2(testo_file, testo_output)

        kit_dir = None
        if export_stems:
            update_progress(job_id, 97, "Export stems", "Kit per DAW...")
            from export_kit import separa_stems
            kit_dir = CFG_OUTPUT / f"{safe_title}_{timestamp}_kit"
            separa_stems(final_path, kit_dir)
            shutil.copy2(str(vocals_path), str(kit_dir / "vocals_original.wav"))
            shutil.copy2(str(instrumental_path), str(kit_dir / "instrumental_original.wav"))
            shutil.copy2(testo_file, str(kit_dir / "testo.json"))

        import soundfile as sf
        info = sf.info(str(final_path))
        result = {
            "title": song_title, "file_name": final_path.name,
            "duration": f"{int(info.duration // 60)}:{int(info.duration % 60):02d}",
            "duration_sec": info.duration,
            "mood": song_data.get("mood", ""), "bpm": song_data.get("bpm", ""),
            "testo": song_data.get("testo", ""),
            "has_kit": kit_dir is not None,
        }

        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["completed"] = datetime.now().isoformat()
        jobs[job_id]["result"] = result
        update_progress(job_id, 100, "Completato!", f'"{song_title}" - {result["duration"]}')

    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed"] = datetime.now().isoformat()
        update_progress(job_id, -1, "Errore", str(e))
        traceback.print_exc()


def run_album_generation(job_id, params):
    try:
        jobs[job_id]["status"] = JobStatus.RUNNING
        jobs[job_id]["started"] = datetime.now().isoformat()
        temi = params.get("temi", [])
        voice_name = params.get("voce")
        export_stems = params.get("export_stems", False)
        n = len(temi)
        results = []
        for i, tema in enumerate(temi):
            update_progress(job_id, int((i/n)*100), f"Traccia {i+1}/{n}", tema[:60])
            try:
                sub_id = create_job("generate_song", {
                    "tema": tema, "voce": voice_name, "export_stems": export_stems,
                })
                run_song_generation(sub_id, jobs[sub_id]["params"])
                if jobs[sub_id]["status"] == JobStatus.COMPLETED:
                    results.append(jobs[sub_id]["result"])
                else:
                    results.append({"title": f"Errore: {tema[:40]}", "error": jobs[sub_id].get("error","")})
            except Exception as e:
                results.append({"title": f"Errore: {tema[:40]}", "error": str(e)})
            time.sleep(3)
        ok = len([r for r in results if "error" not in r])
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["completed"] = datetime.now().isoformat()
        jobs[job_id]["result"] = {"tracks": results, "total": n, "completed": ok}
        update_progress(job_id, 100, "Album completato!", f"{ok}/{n} tracce")
    except Exception as e:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        update_progress(job_id, -1, "Errore", str(e))


def worker():
    while True:
        job_id = job_queue.get()
        if job_id is None:
            break
        job = jobs.get(job_id)
        if not job:
            continue
        if job["type"] == "generate_song":
            run_song_generation(job_id, job["params"])
        elif job["type"] == "generate_album":
            run_album_generation(job_id, job["params"])
        job_queue.task_done()


worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()


# === ROUTES ===

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    tema = data.get("tema", "").strip()
    if not tema:
        return jsonify({"error": "Tema obbligatorio"}), 400
    job_id = create_job("generate_song", {
        "tema": tema,
        "titolo": data.get("titolo", "").strip() or None,
        "voce": data.get("voce", "").strip() or None,
        "export_stems": data.get("export_stems", False),
    })
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/generate-album", methods=["POST"])
def api_generate_album():
    data = request.json
    temi = data.get("temi", [])
    if not temi:
        return jsonify({"error": "Serve almeno un tema"}), 400
    job_id = create_job("generate_album", {
        "temi": temi,
        "voce": data.get("voce", "").strip() or None,
        "export_stems": data.get("export_stems", False),
    })
    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/jobs")
def api_jobs():
    return jsonify(list(jobs.values()))


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job non trovato"}), 404
    return jsonify(job)


@app.route("/api/jobs/<job_id>/progress")
def api_job_progress(job_id):
    def event_stream():
        q = queue.Queue()
        if job_id not in progress_subscribers:
            progress_subscribers[job_id] = []
        progress_subscribers[job_id].append(q)
        try:
            if job_id in jobs:
                job = jobs[job_id]
                yield f"data: {json.dumps({'job_id': job_id, 'progress': job['progress'], 'step': job['current_step'], 'status': job['status']})}\n\n"
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                    parsed = json.loads(data)
                    if parsed.get("status") in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'keepalive': True})}\n\n"
        finally:
            if job_id in progress_subscribers:
                progress_subscribers[job_id].remove(q)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/songs")
def api_songs():
    songs = []
    for f in sorted(OUTPUT_DIR.glob("*.wav"), key=lambda x: x.stat().st_mtime, reverse=True):
        if "_kit" in f.name:
            continue
        info_dict = {"name": f.name, "size_mb": round(f.stat().st_size / (1024*1024), 1)}
        testo_file = f.parent / f"{f.stem}_testo.json"
        if testo_file.exists():
            with open(testo_file, "r", encoding="utf-8") as tf:
                d = json.load(tf)
                info_dict["title"] = d.get("titolo", f.stem)
                info_dict["mood"] = d.get("mood", "")
                info_dict["bpm"] = d.get("bpm", "")
        else:
            info_dict["title"] = f.stem
        try:
            import soundfile as sf
            ai = sf.info(str(f))
            info_dict["duration"] = f"{int(ai.duration // 60)}:{int(ai.duration % 60):02d}"
        except Exception:
            pass
        info_dict["has_kit"] = (f.parent / f"{f.stem}_kit").exists()
        songs.append(info_dict)
    return jsonify(songs)


@app.route("/api/voices")
def api_voices():
    voices = []
    config_file = VOICES_DIR / "voices_config.json"
    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
    default_voice = config.get("default_voice")
    for d in sorted(VOICES_DIR.iterdir()):
        if not d.is_dir():
            continue
        voice = {"name": d.name, "is_default": d.name == default_voice}
        ref = d / "reference.wav"
        if ref.exists():
            try:
                import soundfile as sf
                info = sf.info(str(ref))
                voice["duration"] = f"{info.duration:.1f}s"
            except Exception:
                pass
        voices.append(voice)
    return jsonify(voices)


@app.route("/api/audio/<path:filename>")
def api_audio(filename):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File non trovato"}), 404
    return send_file(str(file_path), mimetype="audio/wav")


@app.route("/api/download/<path:filename>")
def api_download(filename):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File non trovato"}), 404
    return send_file(str(file_path), as_attachment=True)


@app.route("/api/upload-voice", methods=["POST"])
def api_upload_voice():
    if "file" not in request.files:
        return jsonify({"error": "Nessun file"}), 400
    file = request.files["file"]
    name = request.form.get("name", "mia_voce").strip()
    if not name:
        return jsonify({"error": "Nome voce obbligatorio"}), 400
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    dest = voice_dir / "reference.wav"
    file.save(str(dest))
    config_file = VOICES_DIR / "voices_config.json"
    config = {}
    if config_file.exists():
        with open(config_file, "r") as f:
            config = json.load(f)
    if "profiles" not in config:
        config["profiles"] = {}
    config["profiles"][name] = {"added": datetime.now().isoformat(), "source_file": file.filename}
    if config.get("default_voice") is None:
        config["default_voice"] = name
    with open(config_file, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok", "name": name})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  CANTAUTORE DIGITALE - Dashboard")
    print(f"  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
