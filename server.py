"""
ECG Simulator Web Server  ·  v6.0 (FastAPI Architecture)
LifeSigns Biomedical Engineering
"""

import os
import re
import json
import math
import queue
import asyncio
from datetime import datetime

import numpy as np
import sounddevice as sd
import wfdb

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── AUDIO / VISUAL CONFIGURATION ────────────────────────────────────────────
SAMPLE_RATE       = 48000
BLOCK_SIZE        = 2048
MIT_RESAMPLE_RATE = 360
JSON_DEFAULT_FS   = 250
VISUAL_DOWNSAMPLE = 80          # Decimation: 48000/80 = 600 visual Hz
WINDOW_SECONDS    = 6           

JSON_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_data")

# ─── MIT-BIH DATABASE MAP ────────────────────────────────────────────────────
MIT_RECORDS = {
    "100": {"desc": "Normal sinus rhythm", "tags": ["NSR"]},
    "106": {"desc": "Normal sinus rhythm with isolated PVCs", "tags": ["PVC"]},
    "109": {"desc": "Left bundle branch block (LBBB)", "tags": ["LBBB"]},
    "118": {"desc": "Right bundle branch block (RBBB)", "tags": ["RBBB"]},
    "119": {"desc": "Normal sinus rhythm with ventricular bigeminy", "tags": ["Bigeminy"]},
    "200": {"desc": "Ventricular bigeminy with multiformed PVCs", "tags": ["Bigeminy"]},
    "201": {"desc": "Atrial fibrillation and atrial flutter", "tags": ["AFib"]},
    "205": {"desc": "Ventricular tachycardia runs", "tags": ["VTach"]},
    "207": {"desc": "Ventricular fibrillation and asystole", "tags": ["VFib"]},
    "230": {"desc": "Normal sinus rhythm with WPW", "tags": ["WPW"]},
    "231": {"desc": "RBBB with first-degree AV block", "tags": ["RBBB"]},
}

# ─── JSON METADATA PARSER ────────────────────────────────────────────────────
def fast_scan_and_parse_directory() -> list:
    """High-performance scanner utilizing Regex to bypass heavy JSON deserialization."""
    os.makedirs(JSON_DATA_DIR, exist_ok=True)
    out = []
    try: 
        files = sorted(f for f in os.listdir(JSON_DATA_DIR) if f.lower().endswith(".json"))
    except Exception: 
        return out
        
    for fname in files:
        fpath = os.path.join(JSON_DATA_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                content = fh.read()
            
            # ADAPTIVE HANDLING: Skip files missing ecg_records or containing an empty array []
            if '"ecg_records"' not in content or re.search(r'"ecg_records"\s*:\s*\[\s*\]', content):
                continue
                
            meta = {"filepath": fpath, "fname": fname, "admission_id": "UNKNOWN", "facility_id": "—"}
            adm_match = re.search(r'"admission_id"\s*:\s*"([^"]+)"', content)
            if adm_match: meta["admission_id"] = adm_match.group(1)
            
            fac_match = re.search(r'"facility_id"\s*:\s*"([^"]+)"', content)
            if fac_match: meta["facility_id"] = fac_match.group(1)
            
            out.append(meta)
        except Exception as e:
            out.append({"filepath": fpath, "fname": fname, "error": str(e)})
            
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL ENGINE  (Stateful Beat-by-Beat Arrhythmia & EDR Generator)
# ═════════════════════════════════════════════════════════════════════════════

class SignalEngine:
    def __init__(self):
        self.mode             = "Synthetic"
        self.waveform         = "Normal Sinus Rhythm (NSR)"
        self.target_mv        = 1.0
        self.calibration_gain = 0.5
        self.is_calibrated    = False
        self.fs               = SAMPLE_RATE
        
        self.bpm              = 60.0
        self.p_amp            = 0.15
        self.q_amp            = -0.15
        self.t_amp            = 0.30
        self.st_shift         = 0.0
        self.ectopics_per_min = 5

        self.resp_rate        = 15.0  
        self.rsa_depth        = 0.0   
        self.resp_am_depth    = 0.0   
        self.resp_wander_mv   = 0.0   

        self.phase            = 0      
        self.plot_queue       = queue.Queue(maxsize=100) # Thread-safe queue for Audio -> WS pipe
        self.mit_data         = None
        self.mit_index        = 0

        self.beat_idx          = 0             
        self.beat_pos          = 0             
        self.beat_len          = int(self.fs)  
        self.base_len          = int(self.fs)  
        self.morphology        = "NORMAL"
        self.next_forced_state = None          
        
        self._next_beat()

    def get_output_gain(self) -> float:
        if not self.is_calibrated: return float(np.clip(self.target_mv / 5.0, 0.0, 1.0))
        return float(np.clip(self.target_mv * self.calibration_gain, 0.0, 1.0))

    def _next_beat(self):
        self.beat_pos = 0
        self.beat_idx += 1
        wf = self.waveform
        
        f_resp = self.resp_rate / 60.0
        t_sec = self.phase / self.fs
        
        inst_bpm = self.bpm
        if self.rsa_depth > 0.0:
            inst_bpm += self.rsa_depth * np.sin(2 * np.pi * f_resp * t_sec)

        base_len = int((60.0 / max(10, inst_bpm)) * self.fs)
        self.base_len = base_len 
        
        if self.next_forced_state:
            state = self.next_forced_state
            self.next_forced_state = None
            if state == "PVC_COMP_PAUSE":
                self.beat_len = int(base_len * 1.25)
                self.morphology = "PVC"
            elif state == "PVC_COMP_PAUSE_MULTIFOCAL":
                self.beat_len = int(base_len * 1.25)
                self.morphology = "PVC_MULTI"
            elif state == "PAC_RESET_PAUSE":
                self.beat_len = int(base_len * 1.35) 
                self.morphology = "PAC"
            return

        self.beat_len = base_len
        self.morphology = "NORMAL"

        if wf == "Normal Sinus Rhythm (NSR)": pass
        elif wf == "Atrial Fibrillation (AFib)":
            self.beat_len = int(base_len * np.random.uniform(0.7, 1.3))
            self.morphology = "AFIB"
        elif wf == "Atrial Flutter": self.morphology = "AFLUTTER"
        elif wf == "Ventricular Tachycardia (VTach)": self.morphology = "PVC" 
        elif wf == "Ventricular Fibrillation (VFib)": self.morphology = "VFIB"
        elif wf == "Asystole": self.morphology = "ASYSTOLE"

        elif wf == "Ventricular Bigeminy":
            if self.beat_idx % 2 == 1: 
                self.beat_len = int(base_len * 0.75) 
                self.next_forced_state = "PVC_COMP_PAUSE" 
        elif wf == "Ventricular Trigeminy":
            if self.beat_idx % 3 == 2:
                self.beat_len = int(base_len * 0.75)
                self.next_forced_state = "PVC_COMP_PAUSE"
        elif wf == "Atrial Bigeminy":
            if self.beat_idx % 2 == 1:
                self.beat_len = int(base_len * 0.75)
                self.next_forced_state = "PAC_RESET_PAUSE"
        elif wf == "Atrial Trigeminy":
            if self.beat_idx % 3 == 2:
                self.beat_len = int(base_len * 0.75)
                self.next_forced_state = "PAC_RESET_PAUSE"

        elif "Occasional PVC" in wf:
            prob = self.ectopics_per_min / max(10, self.bpm)
            if np.random.random() < prob:
                self.beat_len = int(base_len * 0.75)
                if "Multifocal" in wf and np.random.random() > 0.5:
                    self.next_forced_state = "PVC_COMP_PAUSE_MULTIFOCAL"
                else:
                    self.next_forced_state = "PVC_COMP_PAUSE"
        elif wf == "Occasional PAC":
            prob = self.ectopics_per_min / max(10, self.bpm)
            if np.random.random() < prob:
                self.beat_len = int(base_len * 0.75)
                self.next_forced_state = "PAC_RESET_PAUSE"

    def _generate_beat_chunk(self, t_array: np.ndarray, global_t_sec: np.ndarray) -> np.ndarray:
        cycle_t = (t_array / self.base_len) - 0.4
        buf = np.zeros_like(cycle_t)
        
        if self.morphology in ("NORMAL", "PAC"):
            params = [
                (self.p_amp, -0.20, 0.02),        
                (self.q_amp, -0.05, 0.01),        
                (1.00, 0.00, 0.01),               
                (-0.25, 0.05, 0.01),              
                (self.st_shift, 0.15, 0.06),      
                (self.t_amp, 0.30, 0.06)          
            ]
            for a, c, w in params: buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
                
        elif self.morphology == "PVC":
            params = [(1.2, 0.0, 0.04), (-0.5, 0.1, 0.04), (-0.4, 0.4, 0.08)]
            for a, c, w in params: buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
                
        elif self.morphology == "PVC_MULTI":
            params = [(0.2, 0.0, 0.04), (-1.2, 0.1, 0.05), (0.6, 0.4, 0.08)]
            for a, c, w in params: buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
                
        elif self.morphology == "AFIB":
            params = [
                (self.q_amp, -0.05, 0.01), (1.00, 0.00, 0.01), (-0.25, 0.05, 0.01),
                (self.st_shift, 0.15, 0.06), (self.t_amp, 0.30, 0.06)
            ]
            for a, c, w in params: buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
            f_wave = 0.04 * np.sin(2 * np.pi * 5.2 * global_t_sec) + 0.02 * np.sin(2 * np.pi * 8.1 * global_t_sec)
            buf += f_wave
            
        elif self.morphology == "AFLUTTER":
            params = [
                (self.q_amp, -0.05, 0.01), (1.00, 0.00, 0.01), (-0.25, 0.05, 0.01),
                (self.st_shift, 0.15, 0.06), (self.t_amp, 0.30, 0.06)
            ]
            for a, c, w in params: buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
            sawtooth = 0.08 * (2.0 * (global_t_sec * 5.0 - np.floor(global_t_sec * 5.0 + 0.5)))
            buf += sawtooth
            
        elif self.morphology == "VFIB":
            buf += 0.3 * np.sin(2 * np.pi * 3.1 * global_t_sec) * np.sin(2 * np.pi * 0.4 * global_t_sec)
            buf += 0.2 * np.sin(2 * np.pi * 5.5 * global_t_sec)
            buf += 0.05 * np.random.randn(len(t_array))
            
        elif self.morphology == "ASYSTOLE":
            buf += 0.02 * np.random.randn(len(t_array))
            
        return buf

    def load_mit_record(self, record_id: str):
        try:
            record = wfdb.rdrecord(record_id, pn_dir="mitdb", channels=[0])
            signal = record.p_signal.flatten()
            signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
            max_val = np.max(np.abs(signal))
            if max_val > 0: signal /= max_val
            orig = len(signal)
            target = int(orig / MIT_RESAMPLE_RATE * self.fs)
            self.mit_data  = np.interp(np.linspace(0, orig, target), np.arange(orig), signal).astype(np.float32)
            self.mit_index = 0
            return True, f"Record {record_id} loaded successfully."
        except Exception as exc: return False, str(exc)

    def load_json_record(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as fh: data = json.load(fh)
            records = data.get("ecg_records", [])
            samples = []
            for rec in records:
                outer = rec.get("value", [])
                if outer: samples.extend(outer[0])
            sig = np.array(samples, dtype=np.float32)
            sig = np.nan_to_num(sig, nan=0.0, posinf=1.0, neginf=-1.0)
            total = len(sig)
            native_fs = float(JSON_DEFAULT_FS)
            try:
                s, e = data.get("start_utc"), data.get("end_utc")
                dur = (datetime.fromisoformat(e) - datetime.fromisoformat(s)).total_seconds()
                inf = total / dur
                if 50.0 <= inf <= 4000.0: native_fs = inf
            except Exception: pass
            mv = np.max(np.abs(sig))
            if mv > 0: sig /= mv
            target = int(total / native_fs * SAMPLE_RATE)
            self.mit_data  = np.interp(np.linspace(0, total, target), np.arange(total), sig).astype(np.float32)
            self.mit_index = 0
            return True, f"Loaded {total} samples."
        except Exception as exc: return False, str(exc)

    def audio_callback(self, outdata, frames, time_info, status):
        buffer = np.zeros(frames, dtype=np.float64)

        if self.mode == "Calibration":
            t = (np.arange(frames) + self.phase) / self.fs
            buffer[:frames] = np.sign(np.sin(2.0 * np.pi * t))
            self.phase += frames

        elif self.mode == "Synthetic":
            t_sec = (np.arange(frames) + self.phase) / self.fs
            if self.waveform in ("Sine Wave", "Square Wave", "Sawtooth Wave"):
                freq = self.bpm
                if self.waveform == "Sine Wave": buffer[:frames] = np.sin(2.0 * np.pi * freq * t_sec)
                elif self.waveform == "Square Wave": buffer[:frames] = np.sign(np.sin(2.0 * np.pi * freq * t_sec))
                elif self.waveform == "Sawtooth Wave": buffer[:frames] = 2.0 * (t_sec * freq - np.floor(t_sec * freq + 0.5))
            else:
                out_idx = 0
                while out_idx < frames:
                    if self.beat_pos >= self.beat_len: self._next_beat()
                    rem = self.beat_len - self.beat_pos
                    chunk = min(frames - out_idx, rem)
                    t_array = np.arange(self.beat_pos, self.beat_pos + chunk)
                    global_t_sec = t_sec[out_idx : out_idx + chunk]
                    buffer[out_idx : out_idx + chunk] = self._generate_beat_chunk(t_array, global_t_sec)
                    self.beat_pos += chunk
                    out_idx += chunk
                    
            f_resp = self.resp_rate / 60.0
            if self.resp_am_depth > 0.0:
                am_env = 1.0 + (self.resp_am_depth * np.sin(2 * np.pi * f_resp * t_sec))
                buffer[:frames] *= am_env
                
            if self.resp_wander_mv > 0.0:
                wander = self.resp_wander_mv * np.sin(2 * np.pi * f_resp * t_sec)
                buffer[:frames] += wander

            self.phase += frames

        elif self.mode in ("MIT-BIH", "JSON-ECG"):
            if self.mit_data is not None:
                rem = len(self.mit_data) - self.mit_index
                if rem >= frames:
                    buffer[:frames]   = self.mit_data[self.mit_index:self.mit_index + frames]
                    self.mit_index += frames
                else:
                    wrap           = frames - rem
                    buffer[:rem]      = self.mit_data[self.mit_index:]
                    buffer[rem:frames]= self.mit_data[:wrap]
                    self.mit_index = wrap

        gain  = self.target_mv if self.mode == "Calibration" else self.get_output_gain()
        final = (buffer[:frames] * gain).astype(np.float32)
        outdata[:] = final.reshape(-1, 1)

        try:
            num_blocks = frames // VISUAL_DOWNSAMPLE
            if num_blocks > 0:
                reshaped = final[:num_blocks * VISUAL_DOWNSAMPLE].reshape(num_blocks, VISUAL_DOWNSAMPLE)
                max_v = np.max(reshaped, axis=1)
                min_v = np.min(reshaped, axis=1)
                chunk = np.where(np.abs(max_v) > np.abs(min_v), max_v, min_v).astype(np.float32)
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Push raw bytes for zero-overhead WebSocket transmission
                self.plot_queue.put_nowait(chunk.tobytes())
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
#  FASTAPI SERVER
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = SignalEngine()
audio_stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=engine.audio_callback, dtype="float32")

@app.on_event("startup")
async def startup_event():
    audio_stream.start()
    print("[Audio] SoundDevice Stream Started.")

@app.on_event("shutdown")
async def shutdown_event():
    audio_stream.stop()
    audio_stream.close()

# Models
class ParamUpdate(BaseModel):
    param: str
    value: float

class StringUpdate(BaseModel):
    param: str
    value: str

# Endpoints
@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.websocket("/ws/ecg")
async def websocket_ecg(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Poll the thread-safe queue rapidly and flush all available binary chunks
            while not engine.plot_queue.empty():
                chunk_bytes = engine.plot_queue.get_nowait()
                await websocket.send_bytes(chunk_bytes)
            await asyncio.sleep(0.016) # ~60fps polling
    except WebSocketDisconnect:
        pass

@app.post("/api/update_param")
async def update_param(data: ParamUpdate):
    setattr(engine, data.param, data.value)
    return {"status": "success"}

@app.post("/api/update_string")
async def update_string(data: StringUpdate):
    setattr(engine, data.param, data.value)
    return {"status": "success"}

@app.get("/api/mit_records")
async def get_mit_records():
    return MIT_RECORDS

@app.post("/api/load_mit")
async def load_mit(data: StringUpdate):
    success, msg = await asyncio.to_thread(engine.load_mit_record, data.value)
    if success:
        engine.mode = "MIT-BIH"
    return {"success": success, "message": msg}

@app.get("/api/scan_json")
async def scan_json():
    res = await asyncio.to_thread(fast_scan_and_parse_directory)
    return res

@app.post("/api/load_json")
async def load_json(data: StringUpdate):
    success, msg = await asyncio.to_thread(engine.load_json_record, data.value)
    if success:
        engine.mode = "JSON-ECG"
    return {"success": success, "message": msg}

@app.post("/api/toggle_calibration")
async def toggle_calib():
    if engine.mode == "Calibration":
        engine.mode = "Synthetic"
        engine.is_calibrated = True
        return {"mode": "Synthetic", "msg": "Calibrated successfully."}
    else:
        engine.mode = "Calibration"
        engine.target_mv = 0.1
        return {"mode": "Calibration", "msg": "Calibration mode active. Set slider so output is 1mVpp."}


if __name__ == "__main__":
    import uvicorn
    # Disable auto-reload as it will kill the audio stream thread repeatedly
    uvicorn.run("server:app", host="0.0.0.0", port=44321, reload=False)