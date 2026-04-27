"""
ECG Simulator Workstation  ·  v2.0.0 (EDR & Advanced Arrhythmia Edition)
LifeSigns Biomedical Engineering

Architecture
────────────
  Main process  CustomTkinter control panel + sounddevice audio output
  Plot process  PyQtGraph monitor (Patient Monitor Sweep Style)
  IPC           multiprocessing.Queue  (ECG float32 chunks)
"""

import multiprocessing as mp
import os
import sys
import json
import threading
import queue as _stdlib_queue
from datetime import datetime

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import wfdb

# ─── AUDIO / VISUAL CONFIGURATION ────────────────────────────────────────────
SAMPLE_RATE       = 48000
BLOCK_SIZE        = 2048
MIT_RESAMPLE_RATE = 360
JSON_DEFAULT_FS   = 250
VISUAL_DOWNSAMPLE = 80          # Decimation: 48000/80 = 600 visual Hz
WINDOW_SECONDS    = 10          # Screen width in seconds

_MONITOR_CFG = {
    "sample_rate":       SAMPLE_RATE,
    "visual_downsample": VISUAL_DOWNSAMPLE,
    "window_seconds":    WINDOW_SECONDS,
}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
JSON_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json_data")

SYNTH_WAVEFORMS = [
    "Normal Sinus Rhythm (NSR)",
    "Occasional PAC",
    "Occasional PVC (Unifocal)",
    "Occasional PVC (Multifocal)",
    "Atrial Bigeminy",
    "Atrial Trigeminy",
    "Ventricular Bigeminy",
    "Ventricular Trigeminy",
    "Atrial Fibrillation (AFib)",
    "Atrial Flutter",
    "Ventricular Tachycardia (VTach)",
    "Ventricular Fibrillation (VFib)",
    "Asystole",
    "Sine Wave",
    "Square Wave",
    "Sawtooth Wave"
]

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
def scan_json_directory() -> dict:
    os.makedirs(JSON_DATA_DIR, exist_ok=True)
    result = {}
    try: files = sorted(f for f in os.listdir(JSON_DATA_DIR) if f.lower().endswith(".json"))
    except Exception: return result
    for fname in files:
        fpath = os.path.join(JSON_DATA_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh: data = json.load(fh)
            if not data.get("ecg_records"): continue
            result[f"{data.get('admission_id','UNKNOWN')} | S{data.get('sample_no','?')}"] = fpath
        except Exception: pass
    return result

def parse_json_metadata(file_map: dict) -> list:
    out = []
    for _, fpath in file_map.items():
        try:
            with open(fpath, "r", encoding="utf-8") as fh: data = json.load(fh)
            recs = data.get("ecg_records", [])
            total = sum(len(r.get("value",[[]])[0]) if r.get("value") else 0 for r in recs)
            out.append({
                "filepath": fpath, "fname": os.path.basename(fpath),
                "admission_id": data.get("admission_id", "UNKNOWN"),
                "facility_id": data.get("facility_id", "—"),
                "samples": f"{total:,}"
            })
        except Exception as exc:
            out.append({"filepath": fpath, "fname": os.path.basename(fpath), "error": str(exc)})
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  PATIENT MONITOR PROCESS  (PyQtGraph SWEEP effect)
# ═════════════════════════════════════════════════════════════════════════════

def _monitor_process(ecg_queue: mp.Queue, cfg: dict) -> None:
    import sys
    import numpy as np

    try:
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtWidgets, QtCore
    except ImportError as exc:
        print(f"\n[Monitor] *** Import failed: {exc}")
        return

    VISUAL_FS  = cfg["sample_rate"] / cfg["visual_downsample"]
    WIN_SEC    = float(cfg["window_seconds"])
    MAX_PTS    = int(VISUAL_FS * WIN_SEC)
    UPDATE_HZ  = 30
    BATCH_CAP  = 20

    pg.setConfigOption("background", "#000000")
    pg.setConfigOption("foreground", "#2a2a2a")
    pg.setConfigOption("antialias",  True)

    qapp = QtWidgets.QApplication(sys.argv)
    qapp.setApplicationName("ECG Monitor")

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("ECG Simulator  ·  Lead II Monitor")
    win.resize(1300, 500)
    win.setStyleSheet("QMainWindow, QWidget { background-color: #000000; color: #cccccc; }")

    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    root_vbox = QtWidgets.QVBoxLayout(central)
    root_vbox.setContentsMargins(0, 0, 0, 0)
    root_vbox.setSpacing(0)

    hdr_widget = QtWidgets.QWidget()
    hdr_widget.setFixedHeight(40)
    hdr_widget.setStyleSheet("background-color: #111111; border-bottom: 1px solid #1a1a1a;")
    hdr_h = QtWidgets.QHBoxLayout(hdr_widget)
    hdr_h.setContentsMargins(18, 0, 18, 0)
    
    lbl = QtWidgets.QLabel("  LEAD II  ")
    lbl.setStyleSheet("color:#00ff00;font-size:14px;font-weight:bold;font-family:Consolas, monospace;")
    hdr_h.addWidget(lbl)
    hdr_h.addStretch()
    
    ind = QtWidgets.QLabel("● LIVE  ")
    ind.setStyleSheet("color:red;font-size:12px;font-weight:bold;font-family:Consolas, monospace;")
    hdr_h.addWidget(ind)
    root_vbox.addWidget(hdr_widget)

    plot_widget = pg.PlotWidget()
    plot_widget.setBackground("#000000")
    plot_widget.showGrid(x=True, y=True, alpha=0.10)
    plot_widget.setYRange(-1.6, 1.6, padding=0)
    plot_widget.setXRange(0, MAX_PTS, padding=0)
    plot_widget.setMouseEnabled(x=False, y=False)
    plot_widget.hideButtons()
    
    plot_widget.getAxis('bottom').setStyle(showValues=False)
    plot_widget.getAxis('bottom').setPen(pg.mkPen(color="#1c1c1c", width=1))
    plot_widget.getAxis('left').setPen(pg.mkPen(color="#1c1c1c", width=1))

    curve = plot_widget.plot(pen=pg.mkPen(color="#00ee00", width=1.6))
    root_vbox.addWidget(plot_widget)
    win.show()

    y_data = np.full(MAX_PTS, np.nan)
    scan_idx = [0]
    gap_size = int(VISUAL_FS * 0.3) 

    def _consume() -> None:
        updated = False
        consumed = 0
        while consumed < BATCH_CAP:
            try:
                chunk = ecg_queue.get_nowait()
                if chunk is None:
                    QtCore.QTimer.singleShot(800, qapp.quit)
                    return
            except Exception:
                break

            n = len(chunk)
            if n == 0: continue
            
            idx = scan_idx[0]
            if idx + n < MAX_PTS:
                y_data[idx : idx + n] = chunk
                gap_end = idx + n + gap_size
                if gap_end < MAX_PTS:
                    y_data[idx + n : gap_end] = np.nan
                else:
                    y_data[idx + n :] = np.nan
                    y_data[: gap_end - MAX_PTS] = np.nan
                scan_idx[0] += n
            else:
                space = MAX_PTS - idx
                y_data[idx :] = chunk[:space]
                rem = n - space
                y_data[:rem] = chunk[space:]
                
                gap_end = rem + gap_size
                if gap_end < MAX_PTS:
                    y_data[rem : gap_end] = np.nan
                else:
                    y_data[rem :] = np.nan
                    y_data[: gap_end - MAX_PTS] = np.nan
                scan_idx[0] = rem

            updated = True
            consumed += 1

        if updated:
            curve.setData(y_data, connect="finite")

    consumer_timer = QtCore.QTimer()
    consumer_timer.timeout.connect(_consume)
    consumer_timer.start(int(1000 / UPDATE_HZ))

    exec_fn = getattr(qapp, "exec", None) or getattr(qapp, "exec_", None)
    sys.exit(exec_fn())


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
        
        # User Morphology Controls
        self.bpm              = 60.0
        self.p_amp            = 0.15
        self.q_amp            = -0.15
        self.t_amp            = 0.30
        self.st_shift         = 0.0
        self.ectopics_per_min = 5

        # Respiratory Modulation (EDR) Controls
        self.resp_rate        = 15.0   # Breaths per minute
        self.rsa_depth        = 0.0    # Heart rate varies by ±BPM
        self.resp_am_depth    = 0.0    # QRS amplitude scales by ±percentage
        self.resp_wander_mv   = 0.0    # Isoelectric baseline shift in ±mV

        # Continuous / Hardware tracking
        self.phase            = 0      # Absolute sample counter (Master Clock)
        self.plot_queue       = None
        self.mit_data         = None
        self.mit_index        = 0

        # Beat State Machine
        self.beat_idx          = 0             # Sequential beat counter
        self.beat_pos          = 0             # Current sample index within the active beat
        self.beat_len          = int(self.fs)  # Length of current beat in samples
        self.base_len          = int(self.fs)  # Reference standard RR interval
        self.morphology        = "NORMAL"
        self.next_forced_state = None          # Lookahead state for compensatory pauses
        
        self._next_beat()

    def get_output_gain(self) -> float:
        if not self.is_calibrated: return float(np.clip(self.target_mv / 5.0, 0.0, 1.0))
        return float(np.clip(self.target_mv * self.calibration_gain, 0.0, 1.0))

    def _next_beat(self):
        """FSM calculates the morphology and timing of the upcoming beat."""
        self.beat_pos = 0
        self.beat_idx += 1
        
        wf = self.waveform
        
        # --- Respiratory Sinus Arrhythmia (RSA) ---
        # Evaluate respiratory phase at the start of this beat
        f_resp = self.resp_rate / 60.0
        t_sec = self.phase / self.fs
        
        inst_bpm = self.bpm
        if self.rsa_depth > 0.0:
            inst_bpm += self.rsa_depth * np.sin(2 * np.pi * f_resp * t_sec)

        base_len = int((60.0 / max(10, inst_bpm)) * self.fs)
        self.base_len = base_len # Store reference for morphology mapping
        
        # --- Handle Pre-Determined Compensatory / Reset Pauses ---
        if self.next_forced_state:
            state = self.next_forced_state
            self.next_forced_state = None
            if state == "PVC_COMP_PAUSE":
                self.beat_len = int(base_len * 1.25) # 0.75 + 1.25 = 2.0x normal RR
                self.morphology = "PVC"
            elif state == "PVC_COMP_PAUSE_MULTIFOCAL":
                self.beat_len = int(base_len * 1.25)
                self.morphology = "PVC_MULTI"
            elif state == "PAC_RESET_PAUSE":
                self.beat_len = int(base_len * 1.35) # SA Node Reset: slightly > 2.0x normal RR
                self.morphology = "PAC"
            return

        # --- Standard Reset ---
        self.beat_len = base_len
        self.morphology = "NORMAL"

        if wf == "Normal Sinus Rhythm (NSR)":
            pass
        elif wf == "Atrial Fibrillation (AFib)":
            self.beat_len = int(base_len * np.random.uniform(0.7, 1.3))
            self.morphology = "AFIB"
        elif wf == "Atrial Flutter":
            self.morphology = "AFLUTTER"
        elif wf == "Ventricular Tachycardia (VTach)":
            self.morphology = "PVC" 
        elif wf == "Ventricular Fibrillation (VFib)":
            self.morphology = "VFIB"
        elif wf == "Asystole":
            self.morphology = "ASYSTOLE"

        # --- Deterministic Bigeminy / Trigeminy Logic ---
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

        # --- Probabilistic Ectopy Logic ---
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
        """Renders the mathematical shape of the current active beat into the audio buffer."""
        # Normalize local cycle time
        cycle_t = (t_array / self.base_len) - 0.4
        buf = np.zeros_like(cycle_t)
        
        if self.morphology in ("NORMAL", "PAC"):
            params = [
                (self.p_amp, -0.20, 0.02),        # P
                (self.q_amp, -0.05, 0.01),        # Q
                (1.00, 0.00, 0.01),               # R
                (-0.25, 0.05, 0.01),              # S
                (self.st_shift, 0.15, 0.06),      # ST Shift
                (self.t_amp, 0.30, 0.06)          # T
            ]
            for a, c, w in params:
                buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
                
        elif self.morphology == "PVC":
            params = [
                (1.2, 0.0, 0.04),                 # Wide upright R
                (-0.5, 0.1, 0.04),                # Deep wide S
                (-0.4, 0.4, 0.08)                 # Deep inverted T
            ]
            for a, c, w in params:
                buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
                
        elif self.morphology == "PVC_MULTI":
            params = [
                (0.2, 0.0, 0.04),                 # Small R
                (-1.2, 0.1, 0.05),                # Deep sweeping S
                (0.6, 0.4, 0.08)                  # Tall broad T
            ]
            for a, c, w in params:
                buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
                
        elif self.morphology == "AFIB":
            # No P wave, but standard Q, R, S, ST, T
            params = [
                (self.q_amp, -0.05, 0.01),
                (1.00, 0.00, 0.01),
                (-0.25, 0.05, 0.01),
                (self.st_shift, 0.15, 0.06),
                (self.t_amp, 0.30, 0.06)
            ]
            for a, c, w in params:
                buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
            f_wave = 0.04 * np.sin(2 * np.pi * 5.2 * global_t_sec) + 0.02 * np.sin(2 * np.pi * 8.1 * global_t_sec)
            buf += f_wave
            
        elif self.morphology == "AFLUTTER":
            params = [
                (self.q_amp, -0.05, 0.01),
                (1.00, 0.00, 0.01),
                (-0.25, 0.05, 0.01),
                (self.st_shift, 0.15, 0.06),
                (self.t_amp, 0.30, 0.06)
            ]
            for a, c, w in params:
                buf += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))
            sawtooth = 0.08 * (2.0 * (global_t_sec * 5.0 - np.floor(global_t_sec * 5.0 + 0.5)))
            buf += sawtooth
            
        elif self.morphology == "VFIB":
            buf += 0.3 * np.sin(2 * np.pi * 3.1 * global_t_sec) * np.sin(2 * np.pi * 0.4 * global_t_sec)
            buf += 0.2 * np.sin(2 * np.pi * 5.5 * global_t_sec)
            buf += 0.05 * np.random.randn(len(t_array))
            
        elif self.morphology == "ASYSTOLE":
            buf += 0.02 * np.random.randn(len(t_array))
            
        return buf

    # --- Dataset Loaders ---
    def load_mit_record(self, record_id: str, progress_callback=None):
        try:
            if progress_callback: progress_callback(0.10)
            record = wfdb.rdrecord(record_id, pn_dir="mitdb", channels=[0])
            if progress_callback: progress_callback(0.50)
            signal = record.p_signal.flatten()
            signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=-1.0)
            max_val = np.max(np.abs(signal))
            if max_val > 0: signal /= max_val
            orig = len(signal)
            target = int(orig / MIT_RESAMPLE_RATE * self.fs)
            self.mit_data  = np.interp(np.linspace(0, orig, target), np.arange(orig), signal).astype(np.float32)
            self.mit_index = 0
            if progress_callback: progress_callback(1.00)
            return True, f"Record {record_id} loaded"
        except Exception as exc: return False, str(exc)

    def load_json_record(self, filepath: str, progress_callback=None):
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
            if progress_callback: progress_callback(1.00)
            return True, f"Loaded JSON Data"
        except Exception as exc: return False, str(exc)


    # --- Master Audio Callback ---
    def audio_callback(self, outdata, frames, time_info, status):
        buffer = np.zeros(frames, dtype=np.float64)

        if self.mode == "Calibration":
            t = (np.arange(frames) + self.phase) / self.fs
            buffer[:frames] = np.sign(np.sin(2.0 * np.pi * t))
            self.phase += frames

        elif self.mode == "Synthetic":
            t_sec = (np.arange(frames) + self.phase) / self.fs
            
            if self.waveform in ("Sine Wave", "Square Wave", "Sawtooth Wave"):
                # Continuous non-beat functions
                freq = self.bpm
                if self.waveform == "Sine Wave": buffer[:frames] = np.sin(2.0 * np.pi * freq * t_sec)
                elif self.waveform == "Square Wave": buffer[:frames] = np.sign(np.sin(2.0 * np.pi * freq * t_sec))
                elif self.waveform == "Sawtooth Wave": buffer[:frames] = 2.0 * (t_sec * freq - np.floor(t_sec * freq + 0.5))
            else:
                # Stateful Beat-by-Beat FSM pipeline
                out_idx = 0
                while out_idx < frames:
                    if self.beat_pos >= self.beat_len:
                        self._next_beat()
                    
                    rem = self.beat_len - self.beat_pos
                    chunk = min(frames - out_idx, rem)
                    t_array = np.arange(self.beat_pos, self.beat_pos + chunk)
                    global_t_sec = t_sec[out_idx : out_idx + chunk]
                    
                    buffer[out_idx : out_idx + chunk] = self._generate_beat_chunk(t_array, global_t_sec)
                    
                    self.beat_pos += chunk
                    out_idx += chunk
                    
            # Apply Respiratory Modulations (AM & Baseline Wander) Globally to Synthetic
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

        if self.plot_queue is not None:
            try:
                num_blocks = frames // VISUAL_DOWNSAMPLE
                if num_blocks > 0:
                    reshaped = final[:num_blocks * VISUAL_DOWNSAMPLE].reshape(num_blocks, VISUAL_DOWNSAMPLE)
                    max_v = np.max(reshaped, axis=1)
                    min_v = np.min(reshaped, axis=1)
                    chunk = np.where(np.abs(max_v) > np.abs(min_v), max_v, min_v).astype(np.float32)
                    chunk = np.nan_to_num(chunk, nan=0.0, posinf=1.0, neginf=-1.0)
                    self.plot_queue.put_nowait(chunk)
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════════════════════
#  BROWSER WINDOWS
# ═════════════════════════════════════════════════════════════════════════════

class _BrowserBase(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self._app_parent = parent
        self.withdraw()

    def _present(self):
        self.deiconify()
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))

class MITBrowserWindow(_BrowserBase):
    def __init__(self, parent, on_load_callback):
        super().__init__(parent)
        self.on_load = on_load_callback
        self.title("MIT-BIH Arrhythmia Database")
        self.geometry("1020x720")
        self._build_ui()
        self._refresh_list()
        self.after(20, self._present)

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(self, fg_color="#141424", height=68)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text="MIT-BIH DATABASE", text_color="#2CC985").grid(row=0, column=0, padx=20, pady=12)
        sbg = ctk.CTkFrame(self, fg_color="#0e1117", height=54)
        sbg.grid(row=1, column=0, sticky="ew")
        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", lambda *_: self.after(140, self._refresh_list))
        ctk.CTkEntry(sbg, textvariable=self._search_var, placeholder_text="Search...").grid(row=0, column=0, padx=14, pady=8, sticky="ew")
        self._cards = ctk.CTkScrollableFrame(self, fg_color="#0a0e14")
        self._cards.grid(row=2, column=0, sticky="nsew")
        self._cards.grid_columnconfigure(0, weight=1)

    def _refresh_list(self):
        for w in self._cards.winfo_children(): w.destroy()
        q = self._search_var.get().strip().lower()
        hits = [(r,i) for r,i in MIT_RECORDS.items() if not q or q in (r + " " + i["desc"] + " " + " ".join(i["tags"])).lower()]
        for idx, (rid, info) in enumerate(hits):
            card = ctk.CTkFrame(self._cards, fg_color="#131820", corner_radius=10)
            card.grid(row=idx, column=0, sticky="ew", padx=12, pady=4)
            card.grid_columnconfigure(3, weight=1)
            ctk.CTkLabel(card, text=rid, text_color="#2CC985").grid(row=0, column=1, rowspan=3, padx=12, pady=14)
            ctk.CTkLabel(card, text=info["desc"], text_color="#dce6f0").grid(row=0, column=3, sticky="w", padx=14)
            ctk.CTkButton(card, text="Load  ▶", command=lambda r=rid, d=info["desc"]: self._select(r, d)).grid(row=0, column=4, rowspan=3, padx=16)

    def _select(self, rid, desc):
        self.on_load(rid, desc)
        self._app_parent._close_browser("mit", self)

class JSONBrowserWindow(_BrowserBase):
    def __init__(self, parent, parsed_meta, on_load_cb):
        super().__init__(parent)
        self.on_load = on_load_cb
        self._meta = parsed_meta
        self.title("LifeSigns Protocol V1.1")
        self.geometry("1020x660")
        self._build_ui()
        self._refresh_list()
        self.after(20, self._present)

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(self, fg_color="#0e1a0e", height=68)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text="JSON DATA", text_color="#2CC985").grid(row=0, column=0, padx=20, pady=12)
        sbg = ctk.CTkFrame(self, fg_color="#080e08", height=54)
        sbg.grid(row=1, column=0, sticky="ew")
        self._search_var = ctk.StringVar()
        self._search_var.trace_add("write", lambda *_: self.after(140, self._refresh_list))
        ctk.CTkEntry(sbg, textvariable=self._search_var, placeholder_text="Search...").grid(row=0, column=0, padx=14, pady=8, sticky="ew")
        self._cards = ctk.CTkScrollableFrame(self, fg_color="#080e08")
        self._cards.grid(row=2, column=0, sticky="nsew")
        self._cards.grid_columnconfigure(0, weight=1)

    def _refresh_list(self):
        for w in self._cards.winfo_children(): w.destroy()
        q = self._search_var.get().strip().lower()
        valid = [m for m in self._meta if "error" not in m and (not q or q in m["admission_id"].lower() or q in m["facility_id"].lower())]
        for idx, m in enumerate(valid):
            card = ctk.CTkFrame(self._cards, fg_color="#0d1a0d", corner_radius=10)
            card.grid(row=idx, column=0, sticky="ew", padx=12, pady=4)
            card.grid_columnconfigure(3, weight=1)
            ctk.CTkLabel(card, text=m["admission_id"], text_color="#2CC985").grid(row=0, column=1, rowspan=4, padx=10, pady=12)
            r0 = ctk.CTkFrame(card, fg_color="transparent")
            r0.grid(row=0, column=3, sticky="w", padx=(14,8), pady=(12,2))
            ctk.CTkLabel(r0, text=f"Facility: {m['facility_id']}").pack(side="left")
            ctk.CTkButton(card, text="Load  ▶", command=lambda fp=m["filepath"], mt=m: self._select(fp, mt)).grid(row=0, column=4, rowspan=4, padx=(4,16))

    def _select(self, fpath, meta):
        self.on_load(fpath, meta)
        self._app_parent._close_browser("json", self)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Biomedical Signal Generator - Control Panel")
        self.geometry("1080x960")

        self.engine  = SignalEngine()
        self.stream  = None
        self.running = True

        self._json_map           = {}
        self._mit_browser_open   = False
        self._json_browser_open  = False

        self._ecg_queue  = mp.Queue(maxsize=300)
        self._plot_proc  = None
        self.engine.plot_queue = self._ecg_queue
        
        self._setup_ui()
        self._spawn_monitor()
        self._start_audio_stream()
        self.after(200, self._auto_scan_json)

    def _spawn_monitor(self):
        self._plot_proc = mp.Process(target=_monitor_process, args=(self._ecg_queue, _MONITOR_CFG), daemon=True)
        self._plot_proc.start()

    def _restart_monitor(self):
        try: self._ecg_queue.put_nowait(None)
        except Exception: pass
        if self._plot_proc and self._plot_proc.is_alive():
            self._plot_proc.join(timeout=1.2)
            if self._plot_proc.is_alive(): self._plot_proc.terminate()
        self._ecg_queue = mp.Queue(maxsize=300)
        self.engine.plot_queue = self._ecg_queue
        self._spawn_monitor()

    # ── UI Construction ──
    def _create_slider_row(self, parent, label, from_, to, initial, cmd):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=8)
        lbl = ctk.CTkLabel(row, text=label, width=120, anchor="w")
        lbl.pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{initial:.2f}", width=45)
        val_lbl.pack(side="right")
        
        def _wrapper(v, l=val_lbl, c=cmd):
            l.configure(text=f"{v:.2f}")
            c(v)
            
        sl = ctk.CTkSlider(row, from_=from_, to=to, command=_wrapper)
        sl.set(initial)
        sl.pack(side="left", fill="x", expand=True, padx=10)
        return sl, val_lbl

    def _setup_ui(self):
        self.grid_columnconfigure((0, 1), weight=1, uniform="col")
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=0)
        self.grid_rowconfigure(5, weight=1) # Elastic spacer for footer

        # Header
        hdr = ctk.CTkFrame(self, fg_color="#141424", height=60)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        ctk.CTkLabel(hdr, text="ECG SIMULATOR CONTROL PANEL", text_color="#2CC985", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=15)

        # ── Frame 1: Mode & Waveform ──
        f_mode = ctk.CTkFrame(self, fg_color="#0d1117")
        f_mode.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")
        ctk.CTkLabel(f_mode, text="OPERATION MODE", font=ctk.CTkFont(weight="bold")).pack(pady=(15,5))
        self.mode_seg = ctk.CTkSegmentedButton(f_mode, values=["Synthetic", "MIT-BIH", "JSON-ECG"], command=self._change_mode)
        self.mode_seg.set("Synthetic")
        self.mode_seg.pack(pady=5, padx=20, fill="x")
        self.wave_opt = ctk.CTkOptionMenu(f_mode, values=SYNTH_WAVEFORMS, command=self._change_waveform)
        self.wave_opt.pack(pady=(15, 10), padx=20, fill="x")

        # ── Frame 2: HR / BPM ──
        f_hr = ctk.CTkFrame(self, fg_color="#0d1117")
        f_hr.grid(row=1, column=1, padx=15, pady=10, sticky="nsew")
        ctk.CTkLabel(f_hr, text="HEART RATE / FREQUENCY", font=ctk.CTkFont(weight="bold")).pack(pady=(15,5))
        hr_row = ctk.CTkFrame(f_hr, fg_color="transparent")
        hr_row.pack(fill="x", padx=20, pady=10)
        self._bpm_slider = ctk.CTkSlider(hr_row, from_=10, to=180, command=self._update_bpm_slider)
        self._bpm_slider.set(60)
        self._bpm_slider.pack(side="left", fill="x", expand=True, padx=(0, 15))
        self._bpm_entry = ctk.CTkEntry(hr_row, width=60, justify="center")
        self._bpm_entry.insert(0, "60")
        self._bpm_entry.pack(side="right")
        self._bpm_entry.bind("<Return>", self._update_bpm_entry)
        self._bpm_entry.bind("<FocusOut>", self._update_bpm_entry)
        ctk.CTkLabel(f_hr, text="BPM / Hz", text_color="#888", font=ctk.CTkFont(size=11)).pack(side="bottom", pady=(0, 15))

        # ── Frame 3: Morphology (Synthetic) ──
        f_morph = ctk.CTkFrame(self, fg_color="#0d1117")
        f_morph.grid(row=2, column=0, padx=15, pady=10, sticky="nsew")
        ctk.CTkLabel(f_morph, text="MORPHOLOGY CONTROLS", font=ctk.CTkFont(weight="bold")).pack(pady=(15,0))
        self.sl_p_amp, self.lbl_p_amp = self._create_slider_row(f_morph, "P-Wave Amp", 0.0, 0.5, 0.15, lambda v: setattr(self.engine, 'p_amp', v))
        self.sl_q_amp, self.lbl_q_amp = self._create_slider_row(f_morph, "Q-Wave Depth", -1.0, 0.0, -0.15, lambda v: setattr(self.engine, 'q_amp', v))
        self.sl_t_amp, self.lbl_t_amp = self._create_slider_row(f_morph, "T-Wave Amp", -0.5, 1.0, 0.30, lambda v: setattr(self.engine, 't_amp', v))
        self.sl_st, self.lbl_st = self._create_slider_row(f_morph, "ST Elevation", -0.5, 0.5, 0.0, lambda v: setattr(self.engine, 'st_shift', v))

        # ── Frame 4: Arrhythmia Settings ──
        f_arr = ctk.CTkFrame(self, fg_color="#0d1117")
        f_arr.grid(row=2, column=1, padx=15, pady=10, sticky="nsew")
        ctk.CTkLabel(f_arr, text="ECTOPY SETTINGS", font=ctk.CTkFont(weight="bold")).pack(pady=(15,0))
        self.sl_ectopy, self.lbl_ectopy = self._create_slider_row(f_arr, "Ectopics per min", 0, 30, 5, lambda v: setattr(self.engine, 'ectopics_per_min', int(v)))
        self.lbl_ectopy.configure(text="5") # Format as int
        
        # ── Frame 5: Amplitude & Calibration ──
        f_amp = ctk.CTkFrame(self, fg_color="#0d1117")
        f_amp.grid(row=3, column=0, padx=15, pady=10, sticky="nsew")
        ctk.CTkLabel(f_amp, text="SIGNAL AMPLITUDE", font=ctk.CTkFont(weight="bold")).pack(pady=(15,0))
        amp_row = ctk.CTkFrame(f_amp, fg_color="transparent")
        amp_row.pack(fill="x", padx=20, pady=15)
        self.amp_val_label = ctk.CTkLabel(amp_row, text="Gain: 20%", width=90, anchor="e")
        self.amp_val_label.pack(side="right")
        self.amp_slider = ctk.CTkSlider(amp_row, from_=0, to=5, command=self._update_amp_slider)
        self.amp_slider.set(1.0)
        self.amp_slider.pack(side="left", fill="x", expand=True, padx=(0, 20))
        self.calib_btn = ctk.CTkButton(f_amp, text="Step 1: Calibrate Hardware", fg_color="#444", command=self._toggle_calibration_mode)
        self.calib_btn.pack(pady=(0, 15))

        # ── Frame 6: Respiratory Modulation (EDR) ──
        f_resp = ctk.CTkFrame(self, fg_color="#0d1117")
        f_resp.grid(row=3, column=1, padx=15, pady=10, sticky="nsew")
        ctk.CTkLabel(f_resp, text="RESPIRATORY MODULATION (EDR)", font=ctk.CTkFont(weight="bold")).pack(pady=(15,0))
        
        rpm_row = ctk.CTkFrame(f_resp, fg_color="transparent")
        rpm_row.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(rpm_row, text="Resp Rate (RPM)", width=120, anchor="w").pack(side="left")
        self._rpm_entry = ctk.CTkEntry(rpm_row, width=45, justify="center")
        self._rpm_entry.insert(0, "15")
        self._rpm_entry.pack(side="right")
        self._rpm_slider = ctk.CTkSlider(rpm_row, from_=0, to=40, command=self._update_rpm_slider)
        self._rpm_slider.set(15)
        self._rpm_slider.pack(side="left", fill="x", expand=True, padx=10)
        self._rpm_entry.bind("<Return>", self._update_rpm_entry)
        self._rpm_entry.bind("<FocusOut>", self._update_rpm_entry)

        self.sl_rsa, self.lbl_rsa = self._create_slider_row(f_resp, "RSA Depth (BPM)", 0.0, 20.0, 0.0, lambda v: setattr(self.engine, 'rsa_depth', v))
        
        def _update_am(v):
            self.engine.resp_am_depth = v
            self.lbl_am.configure(text=f"{int(v*100)}%")
            
        self.sl_am, self.lbl_am = self._create_slider_row(f_resp, "Thoracic AM", 0.0, 0.5, 0.0, _update_am)
        self.lbl_am.configure(text="0%")
        self.sl_wander, self.lbl_wander = self._create_slider_row(f_resp, "Base Wander (mV)", 0.0, 1.0, 0.0, lambda v: setattr(self.engine, 'resp_wander_mv', v))

        # ── Frame 7: Databases ──
        db_container = ctk.CTkFrame(self, fg_color="transparent")
        db_container.grid(row=4, column=0, columnspan=2, padx=5, pady=10, sticky="nsew")
        db_container.grid_columnconfigure((0,1), weight=1, uniform="c")

        mit = ctk.CTkFrame(db_container, fg_color="#12151e")
        mit.grid(row=0, column=0, padx=10, pady=0, sticky="nsew")
        ctk.CTkLabel(mit, text="MIT-BIH DATABASE", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self._mit_status = ctk.CTkLabel(mit, text="No record loaded", text_color="#aaa")
        self._mit_status.pack(fill="x", padx=14, pady=4)
        ctk.CTkButton(mit, text="Browse All 48 Records  ▶", command=self._open_mit_browser).pack(fill="x", padx=20, pady=(5, 15))

        jsn = ctk.CTkFrame(db_container, fg_color="#0d1a0d")
        jsn.grid(row=0, column=1, padx=10, pady=0, sticky="nsew")
        ctk.CTkLabel(jsn, text="LIFESIGNS JSON DATA", text_color="#2CC985", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 0))
        self._json_badge = ctk.CTkLabel(jsn, text=" scanning… ", text_color="#888")
        self._json_badge.pack()
        self._json_status = ctk.CTkLabel(jsn, text="No file loaded", text_color="#aaa")
        self._json_status.pack(fill="x", padx=14, pady=4)
        self._json_browse_btn = ctk.CTkButton(jsn, text="Browse JSON Files  ▶", state="disabled", command=self._open_json_browser, fg_color="#1e5c1e", hover_color="#2a7a2a")
        self._json_browse_btn.pack(fill="x", padx=20, pady=(5, 15))

        # ── Footer ──
        ftr = ctk.CTkFrame(self, fg_color="transparent")
        ftr.grid(row=5, column=0, columnspan=2, sticky="s", pady=10)
        ctk.CTkButton(ftr, text="Restart Plot Monitor", command=self._restart_monitor, fg_color="#004488", width=200).pack()

        # Initialize UI state locks
        self._change_waveform(SYNTH_WAVEFORMS[0])

    # ── Control Callbacks ──
    def _update_bpm_slider(self, value):
        val = int(value)
        self.engine.bpm = val
        self._bpm_entry.delete(0, "end")
        self._bpm_entry.insert(0, str(val))

    def _update_bpm_entry(self, event=None):
        try:
            val = float(self._bpm_entry.get())
            val = max(10.0, min(180.0, val))
            self.engine.bpm = val
            self._bpm_slider.set(val)
            self._bpm_entry.delete(0, "end")
            self._bpm_entry.insert(0, str(int(val)))
        except ValueError:
            self._bpm_entry.delete(0, "end")
            self._bpm_entry.insert(0, str(int(self.engine.bpm)))
            
    def _update_rpm_slider(self, value):
        val = int(value)
        self.engine.resp_rate = val
        self._rpm_entry.delete(0, "end")
        self._rpm_entry.insert(0, str(val))

    def _update_rpm_entry(self, event=None):
        try:
            val = float(self._rpm_entry.get())
            val = max(0.0, min(40.0, val))
            self.engine.resp_rate = val
            self._rpm_slider.set(val)
            self._rpm_entry.delete(0, "end")
            self._rpm_entry.insert(0, str(int(val)))
        except ValueError:
            self._rpm_entry.delete(0, "end")
            self._rpm_entry.insert(0, str(int(self.engine.resp_rate)))

    def _update_amp_slider(self, value):
        self.engine.target_mv = float(value)
        if self.engine.is_calibrated:
            self.amp_val_label.configure(text=f"Output: {value:.2f} mV")
        else:
            self.amp_val_label.configure(text=f"Gain: {int((value / 5.0) * 100)}%")

    def _toggle_calibration_mode(self):
        if self.engine.mode == "Calibration":
            self.engine.mode = "Synthetic"
            self.engine.calibration_gain = self.amp_slider.get()
            self.engine.is_calibrated = True
            self.calib_btn.configure(text="Recalibrate System", fg_color="#444")
            self.amp_slider.configure(from_=0, to=5)
            self.amp_slider.set(1.0)
            self._update_amp_slider(1.0)
            messagebox.showinfo("Success", "System Calibrated.\nSlider now sets actual Millivolts (0-5 mV).")
        else:
            self.engine.mode = "Calibration"
            self.calib_btn.configure(text="CONFIRM 1mV ON SCOPE", fg_color="red")
            self.amp_slider.configure(from_=0, to=1.0)
            self.amp_slider.set(0.1)
            self._update_amp_slider(0.1)
            messagebox.showinfo("Calibration", "1. Connect Scope.\n2. Signal is now 1Hz Square Wave.\n3. Adjust Slider until Scope shows exactly 1mVpp.\n4. Click 'CONFIRM' button.")

    def _change_mode(self, value):
        self.engine.mode = value
        resp_widgets = [self._rpm_slider, self._rpm_entry, self.sl_rsa, self.sl_am, self.sl_wander]
        
        if value in ("MIT-BIH", "JSON-ECG"):
            self.wave_opt.configure(state="disabled")
            self._bpm_slider.configure(state="disabled")
            self._bpm_entry.configure(state="disabled")
            for w in (self.sl_p_amp, self.sl_q_amp, self.sl_t_amp, self.sl_st, self.sl_ectopy): 
                w.configure(state="disabled")
            for w in resp_widgets: w.configure(state="disabled")
        else:
            self.wave_opt.configure(state="normal")
            self._bpm_slider.configure(state="normal")
            self._bpm_entry.configure(state="normal")
            self._change_waveform(self.engine.waveform) 
            for w in resp_widgets: w.configure(state="normal")

    def _change_waveform(self, value):
        self.engine.waveform = value
        
        if value in ("Sine Wave", "Square Wave", "Sawtooth Wave"):
            for w in (self.sl_p_amp, self.sl_q_amp, self.sl_t_amp, self.sl_st): 
                w.configure(state="disabled")
        else:
            for w in (self.sl_p_amp, self.sl_q_amp, self.sl_t_amp, self.sl_st): 
                w.configure(state="normal")
            
        if "Occasional" in value:
            self.sl_ectopy.configure(state="normal")
        else:
            self.sl_ectopy.configure(state="disabled")

    def _close_browser(self, which, win):
        setattr(self, f"_{which}_browser_open", False)
        try: win.destroy()
        except Exception: pass

    # MIT
    def _open_mit_browser(self):
        if self._mit_browser_open: return
        self._mit_browser_open = True
        w = MITBrowserWindow(self, self._on_mit_selected)
        w.protocol("WM_DELETE_WINDOW", lambda: self._close_browser("mit", w))

    def _on_mit_selected(self, rid, desc):
        self._mit_status.configure(text=f"Downloading {rid}…")
        def _t():
            ok, msg = self.engine.load_mit_record(rid)
            if ok: self.after(0, lambda: self._on_mit_loaded(rid, desc))
        threading.Thread(target=_t, daemon=True).start()

    def _on_mit_loaded(self, rid, desc):
        self.mode_seg.set("MIT-BIH"); self._change_mode("MIT-BIH")
        self._mit_status.configure(text=f"{rid}  —  {desc}")

    # JSON
    def _auto_scan_json(self):
        def _s():
            r = scan_json_directory()
            self.after(0, lambda: self._apply_scan(r))
        threading.Thread(target=_s, daemon=True).start()

    def _apply_scan(self, result):
        self._json_map = result
        self._json_badge.configure(text=f" {len(result)} files ")
        self._json_browse_btn.configure(state="normal" if result else "disabled")

    def _open_json_browser(self):
        if self._json_browser_open: return
        self._json_browser_open = True
        self._json_browse_btn.configure(text="Parsing Metadata...", state="disabled")
        
        def _parse_and_open():
            parsed_meta = parse_json_metadata(self._json_map)
            self.after(0, lambda: self._show_json_ui(parsed_meta))
            
        threading.Thread(target=_parse_and_open, daemon=True).start()

    def _show_json_ui(self, parsed_meta):
        self._json_browse_btn.configure(text="Browse JSON Files  ▶", state="normal")
        try:
            w = JSONBrowserWindow(self, parsed_meta, on_load_cb=self._on_json_selected)
            w.protocol("WM_DELETE_WINDOW", lambda: self._close_browser("json", w))
        except Exception:
            self._json_browser_open = False

    def _on_json_selected(self, fpath, meta):
        self._json_status.configure(text="Loading…")
        def _t():
            ok, msg = self.engine.load_json_record(fpath)
            if ok: self.after(0, lambda: self._on_json_loaded(meta))
        threading.Thread(target=_t, daemon=True).start()

    def _on_json_loaded(self, meta):
        self.mode_seg.set("JSON-ECG"); self._change_mode("JSON-ECG")
        self._json_status.configure(text=f"{meta.get('admission_id','?')} loaded")

    def _start_audio_stream(self):
        self.stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=self.engine.audio_callback, dtype="float32")
        self.stream.start()

    def on_close(self):
        self.running = False
        try: self._ecg_queue.put_nowait(None)
        except Exception: pass
        if self._plot_proc and self._plot_proc.is_alive():
            self._plot_proc.join(timeout=1.0)
            if self._plot_proc.is_alive(): self._plot_proc.terminate()
        if self.stream: self.stream.stop(); self.stream.close()
        self.quit(); self.destroy(); sys.exit()

if __name__ == "__main__":
    mp.freeze_support()
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()