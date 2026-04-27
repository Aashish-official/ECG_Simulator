import tkinter as tk
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wfdb
import threading
import queue
import time
import sys
import os
import json
from datetime import datetime

# --- CONFIGURATION ---
SAMPLE_RATE = 48000         # Audio DAC Sample Rate
BLOCK_SIZE = 2048           # Audio Buffer Size
MIT_RESAMPLE_RATE = 360     # Native MIT-BIH Hz
JSON_DEFAULT_FS = 250       # Fallback sample rate if timestamps unavailable

# Visualization Settings
VISUAL_DOWNSAMPLE = 80
WINDOW_SECONDS = 10
PLOT_POINTS = int((SAMPLE_RATE / VISUAL_DOWNSAMPLE) * WINDOW_SECONDS)

# Set Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# JSON data directory (sibling of script — create if absent)
JSON_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json_data')

# Curated MIT-BIH record map
MIT_RECORD_MAP = {
    "100: Normal Sinus Rhythm":       "100",
    "101: Normal Sinus Rhythm (Ex 2)":"101",
    "106: PVCs (Ventricular Ectopy)": "106",
    "109: LBBB":                      "109",
    "118: RBBB":                      "118",
    "119: Ventricular Bigeminy":      "119",
    "200: Ventricular Bigeminy (2)":  "200",
    "201: Atrial Fibrillation":       "201",
    "203: Multi-modal Arrhythmia":    "203",
    "205: Ventricular Tachycardia":   "205",
    "230: WPW Syndrome":              "230",
    "231: RBBB & 1st Deg Block":      "231",
}


# ---------------------------------------------------------------------------
#  JSON Directory Scanner
# ---------------------------------------------------------------------------

def scan_json_directory():
    """
    Scan JSON_DATA_DIR for LifeSigns Protocol V1.1 JSON files.

    Returns  { display_label: filepath }  ordered alphabetically.
    Silently skips:
      - Files that fail to parse
      - Files where ecg_records is absent or empty  (e.g. sample3)
    Creates JSON_DATA_DIR if it does not exist.
    """
    os.makedirs(JSON_DATA_DIR, exist_ok=True)
    result = {}

    try:
        files = sorted(f for f in os.listdir(JSON_DATA_DIR) if f.lower().endswith('.json'))
    except Exception as exc:
        print(f"[JSON Scan] Cannot list directory: {exc}")
        return result

    for fname in files:
        fpath = os.path.join(JSON_DATA_DIR, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            ecg_records = data.get('ecg_records', [])
            if not ecg_records:
                # Empty file — omit from dropdown
                print(f"[JSON Scan] Omitting '{fname}': ecg_records is empty.")
                continue

            adm_id    = data.get('admission_id', 'UNKNOWN')
            sample_no = data.get('sample_no', '?')
            start_ist = data.get('start_ist', '')
            pkt_count = data.get('packet_count') or len(ecg_records)

            label = f"{adm_id}  |  S{sample_no}  |  {start_ist}  ({pkt_count} pkts)"
            result[label] = fpath

        except Exception as exc:
            print(f"[JSON Scan] Skipping '{fname}': {exc}")

    return result


# ---------------------------------------------------------------------------
#  Signal Engine
# ---------------------------------------------------------------------------

class SignalEngine:
    def __init__(self):
        self.mode   = "Synthetic"
        self.waveform = "NSR"
        self.target_mv        = 1.0
        self.calibration_gain = 0.5
        self.is_calibrated    = False

        self.bpm   = 60
        self.fs    = SAMPLE_RATE
        self.phase = 0

        # Synthetic Lead-II ECG gaussian params: (amplitude, centre, width)
        self.ecg_params = {
            'P': ( 0.15, -0.20, 0.02),
            'Q': (-0.15, -0.05, 0.01),
            'R': ( 1.00,  0.00, 0.01),
            'S': (-0.25,  0.05, 0.01),
            'T': ( 0.30,  0.30, 0.06),
        }

        # Shared playback buffer — used by both MIT-BIH and JSON-ECG modes
        self.mit_data  = None   # np.float32 array, normalised ±1.0, at SAMPLE_RATE
        self.mit_index = 0
        self.plot_queue = queue.Queue(maxsize=100)

    # ------------------------------------------------------------------
    def get_output_gain(self):
        if not self.is_calibrated:
            return np.clip(self.target_mv / 5.0, 0.0, 1.0)
        return np.clip(self.target_mv * self.calibration_gain, 0.0, 1.0)

    # ------------------------------------------------------------------
    def load_mit_record(self, record_id, progress_callback=None):
        """Download and resample a MIT-BIH record into self.mit_data."""
        try:
            if progress_callback: progress_callback(0.10)
            record = wfdb.rdrecord(record_id, pn_dir='mitdb', channels=[0])
            if progress_callback: progress_callback(0.50)

            signal  = record.p_signal.flatten()
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val

            original_len = len(signal)
            duration_sec = original_len / MIT_RESAMPLE_RATE
            target_len   = int(duration_sec * self.fs)

            if progress_callback: progress_callback(0.70)

            self.mit_data = np.interp(
                np.linspace(0, original_len, target_len),
                np.arange(original_len),
                signal,
            ).astype(np.float32)
            self.mit_index = 0

            if progress_callback: progress_callback(1.00)
            return True, f"Record {record_id}: {original_len} samples @ {MIT_RESAMPLE_RATE} Hz"
        except Exception as exc:
            print(f"[MIT Loader] {exc}")
            return False, str(exc)

    # ------------------------------------------------------------------
    def load_json_record(self, filepath, progress_callback=None):
        """
        Load a LifeSigns Protocol V1.1 JSON file.

        JSON structure handled:
          {
            "admission_id": ...,
            "start_utc": "2026-04-21T07:45:00",
            "end_utc":   "2026-04-21T07:47:00",
            "ecg_records": [
              {
                "utcTimestamp": ...,
                "value": [ [ s0, s1, s2, ... ] ],   # outer=channels, inner=samples
                "packetNo": ...,
              },
              ...
            ]
          }

        Pipeline:
          1. Parse & validate ecg_records is non-empty
          2. Flatten value[][] in record order  (first channel only for multi-channel)
          3. Infer native FS from (start_utc, end_utc); fallback = JSON_DEFAULT_FS
          4. Normalise to ±1.0
          5. Linear-resample to SAMPLE_RATE
          6. Store in self.mit_data, reset self.mit_index
        """
        try:
            if progress_callback: progress_callback(0.05)

            with open(filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            ecg_records = data.get('ecg_records', [])
            if not ecg_records:
                return False, "ecg_records array is empty — nothing to simulate."

            if progress_callback: progress_callback(0.20)

            # ---- Step 1: Flatten samples ------------------------------------
            # value = [ [s0, s1, ...], [ch2_s0, ...], ... ]
            # We concatenate channel-0 data from every record in time order.
            all_samples: list = []
            for rec in ecg_records:
                value_outer = rec.get('value', [])
                if value_outer:
                    # Take first (and usually only) channel
                    all_samples.extend(value_outer[0])

            if not all_samples:
                return False, "All ecg_records have empty value arrays."

            signal        = np.array(all_samples, dtype=np.float32)
            total_samples = len(signal)

            if progress_callback: progress_callback(0.40)

            # ---- Step 2: Infer native sample rate ---------------------------
            native_fs = float(JSON_DEFAULT_FS)
            try:
                s_str = data.get('start_utc')
                e_str = data.get('end_utc')
                if s_str and e_str:
                    duration_sec = (
                        datetime.fromisoformat(e_str) - datetime.fromisoformat(s_str)
                    ).total_seconds()
                    if duration_sec > 0:
                        inferred = total_samples / duration_sec
                        if 50.0 <= inferred <= 4000.0:
                            native_fs = inferred
                            print(f"[JSON Loader] Inferred FS = {native_fs:.1f} Hz "
                                  f"({total_samples} samples / {duration_sec:.1f} s)")
                        else:
                            print(f"[JSON Loader] Inferred FS {inferred:.1f} Hz out of "
                                  f"[50, 4000] range — using {JSON_DEFAULT_FS} Hz default.")
            except Exception as exc:
                print(f"[JSON Loader] FS inference error: {exc} — using {JSON_DEFAULT_FS} Hz")

            if progress_callback: progress_callback(0.60)

            # ---- Step 3: Normalise ------------------------------------------
            max_val = np.max(np.abs(signal))
            if max_val > 0.0:
                signal = signal / max_val

            # ---- Step 4: Resample to SAMPLE_RATE ----------------------------
            original_len  = len(signal)
            duration_data = original_len / native_fs
            target_len    = int(duration_data * SAMPLE_RATE)

            if progress_callback: progress_callback(0.80)

            self.mit_data = np.interp(
                np.linspace(0, original_len, target_len),
                np.arange(original_len),
                signal,
            ).astype(np.float32)
            self.mit_index = 0

            if progress_callback: progress_callback(1.00)

            adm_id = data.get('admission_id', 'UNKNOWN')
            msg = (
                f"{len(ecg_records)} records  |  {total_samples} samples  |  "
                f"~{native_fs:.0f} Hz native  |  {duration_data:.1f} s  [{adm_id}]"
            )
            return True, msg

        except Exception as exc:
            print(f"[JSON Loader] {exc}")
            return False, str(exc)

    # ------------------------------------------------------------------
    def audio_callback(self, outdata, frames, time_info, status):
        buffer = np.zeros(frames, dtype=np.float64)

        # Frequency / rate mapping
        freq = (self.bpm / 60.0) if self.waveform in ("NSR", "PVC") else float(self.bpm)

        # ---- Calibration: 1 Hz square wave ----------------------------------
        if self.mode == "Calibration":
            t = (np.arange(frames) + self.phase) / self.fs
            buffer = np.sign(np.sin(2 * np.pi * 1.0 * t))
            self.phase += frames

        # ---- Synthetic waveform generators ----------------------------------
        elif self.mode == "Synthetic":
            t = (np.arange(frames) + self.phase) / self.fs
            if self.waveform == "Sine":
                buffer = np.sin(2 * np.pi * freq * t)
            elif self.waveform == "Square":
                buffer = np.sign(np.sin(2 * np.pi * freq * t))
            elif self.waveform == "Sawtooth":
                buffer = 2.0 * (t * freq - np.floor(t * freq + 0.5))
            elif self.waveform in ("NSR", "PVC"):
                period_samples = max(1, int(self.fs / freq))
                local_t   = np.arange(frames) + self.phase
                cycle_t   = (local_t % period_samples) / period_samples - 0.5
                p_list = (
                    [(0.0, -0.2, 0.02), (1.2, 0.0, 0.04), (-0.5, 0.1, 0.04), (-0.4, 0.4, 0.08)]
                    if self.waveform == "PVC"
                    else list(self.ecg_params.values())
                )
                for a, c, w in p_list:
                    buffer += a * np.exp(-((cycle_t - c) ** 2) / (2.0 * w ** 2))
            self.phase += frames

        # ---- MIT-BIH or JSON-ECG: shared looping playback -------------------
        elif self.mode in ("MIT-BIH", "JSON-ECG"):
            if self.mit_data is None:
                buffer[:] = 0.0
            else:
                remaining = len(self.mit_data) - self.mit_index
                if remaining >= frames:
                    buffer[:] = self.mit_data[self.mit_index:self.mit_index + frames]
                    self.mit_index += frames
                else:
                    # Seamless wrap-around loop
                    buffer[:remaining]  = self.mit_data[self.mit_index:]
                    wrap_len            = frames - remaining
                    buffer[remaining:]  = self.mit_data[:wrap_len]
                    self.mit_index      = wrap_len

        # ---- Apply output gain ----------------------------------------------
        gain         = self.target_mv if self.mode == "Calibration" else self.get_output_gain()
        final_output = (buffer * gain).astype(np.float32)
        outdata[:]   = final_output.reshape(-1, 1)

        try:
            self.plot_queue.put_nowait(final_output[::VISUAL_DOWNSAMPLE])
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
#  Application
# ---------------------------------------------------------------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Biomedical Signal Generator - PhD Workstation")
        self.geometry("1200x880")

        self.engine = SignalEngine()
        self.stream  = None
        self.running = True

        # { display_label: filepath }  populated by rescan_json_directory()
        self.json_file_map: dict = {}

        self._setup_ui()
        self._start_audio_stream()
        self._start_scan_plot()

        # Auto-scan after window is fully initialised
        self.after(200, self.rescan_json_directory)

    # ======================================================================
    #  UI Layout
    # ======================================================================

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Sidebar -------------------------------------------------------
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)   # elastic spacer

        # Header
        ctk.CTkLabel(
            self.sidebar, text="ECG SIMULATOR",
            font=ctk.CTkFont(size=24, weight="bold"), text_color="#2CC985",
        ).grid(row=0, column=0, padx=20, pady=(20, 5))
        ctk.CTkLabel(
            self.sidebar, text="v1.1 Release",
            font=ctk.CTkFont(size=10),
        ).grid(row=1, column=0, padx=20, pady=(0, 20))

        # 1. Mode switch  (Synthetic / MIT-BIH / JSON-ECG)
        self.mode_label = ctk.CTkLabel(
            self.sidebar, text="OPERATION MODE", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.mode_label.grid(row=2, column=0, padx=20, sticky="w")
        self.mode_seg = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["Synthetic", "MIT-BIH", "JSON-ECG"],
            command=self.change_mode,
        )
        self.mode_seg.set("Synthetic")
        self.mode_seg.grid(row=3, column=0, padx=20, pady=5)

        # 2. Waveform selector
        self.wave_opt = ctk.CTkOptionMenu(
            self.sidebar,
            values=["NSR", "PVC", "Square", "Sine", "Sawtooth"],
            command=self.change_waveform,
        )
        self.wave_opt.grid(row=4, column=0, padx=20, pady=10)

        # 3. Frequency / BPM
        ctk.CTkLabel(
            self.sidebar, text="FREQUENCY / RATE", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        bpm_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bpm_row.grid(row=6, column=0, padx=20, pady=0, sticky="ew")
        self.bpm_val_label = ctk.CTkLabel(bpm_row, text="60 BPM")
        self.bpm_val_label.pack(side="right")
        self.bpm_slider = ctk.CTkSlider(
            self.sidebar, from_=10, to=180, number_of_steps=170,
            command=self.update_bpm_slider,
        )
        self.bpm_slider.set(60)
        self.bpm_slider.grid(row=7, column=0, padx=20, pady=(0, 10))

        # 4. Amplitude & calibration
        ctk.CTkLabel(
            self.sidebar, text="SIGNAL AMPLITUDE", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=8, column=0, padx=20, pady=(10, 0), sticky="w")
        amp_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        amp_row.grid(row=9, column=0, padx=20, pady=0, sticky="ew")
        self.amp_val_label = ctk.CTkLabel(amp_row, text="Gain: 20%")
        self.amp_val_label.pack(side="right")
        self.amp_slider = ctk.CTkSlider(
            self.sidebar, from_=0, to=5, number_of_steps=100,
            command=self.update_amp_slider,
        )
        self.amp_slider.set(1.0)
        self.amp_slider.grid(row=10, column=0, padx=20, pady=(0, 10))
        self.calib_btn = ctk.CTkButton(
            self.sidebar, text="Step 1: Calibrate Output", fg_color="#444",
            command=self.toggle_calibration_mode,
        )
        self.calib_btn.grid(row=11, column=0, padx=20, pady=5)

        # row 12 = elastic spacer

        # ---- MIT-BIH panel -------------------------------------------------
        mit_frame = ctk.CTkFrame(self.sidebar, fg_color="#2b2b2b")
        mit_frame.grid(row=13, column=0, padx=10, pady=(10, 4), sticky="ew")

        ctk.CTkLabel(
            mit_frame, text="MIT-BIH DATABASE",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(pady=(8, 4))
        self.mit_opt = ctk.CTkOptionMenu(mit_frame, values=list(MIT_RECORD_MAP.keys()))
        self.mit_opt.pack(pady=4, padx=10, fill="x")
        self.mit_progress = ctk.CTkProgressBar(mit_frame)
        self.mit_progress.set(0)
        self.mit_progress.pack(pady=4, padx=10, fill="x")
        self.load_btn = ctk.CTkButton(
            mit_frame, text="Download Record", command=self.load_mit_data,
        )
        self.load_btn.pack(pady=(4, 10), padx=10)

        # ---- LifeSigns JSON-ECG panel --------------------------------------
        json_frame = ctk.CTkFrame(self.sidebar, fg_color="#1a2a1a")
        json_frame.grid(row=14, column=0, padx=10, pady=(4, 14), sticky="ew")

        ctk.CTkLabel(
            json_frame, text="LIFESIGNS JSON DATA",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#2CC985",
        ).pack(pady=(10, 2))

        self.json_scan_label = ctk.CTkLabel(
            json_frame, text="Scanning json_data/ …",
            font=ctk.CTkFont(size=9), text_color="#888888",
        )
        self.json_scan_label.pack(pady=(0, 4))

        self.json_opt = ctk.CTkOptionMenu(json_frame, values=["— scanning —"])
        self.json_opt.pack(pady=4, padx=10, fill="x")

        json_btn_row = ctk.CTkFrame(json_frame, fg_color="transparent")
        json_btn_row.pack(fill="x", padx=10, pady=(2, 4))

        self.json_refresh_btn = ctk.CTkButton(
            json_btn_row, text="↺ Rescan", width=80,
            fg_color="#2d3d2d", hover_color="#3d5a3d",
            command=self.rescan_json_directory,
        )
        self.json_refresh_btn.pack(side="left", padx=(0, 5))

        self.json_load_btn = ctk.CTkButton(
            json_btn_row, text="Load & Simulate",
            fg_color="#1e5c1e", hover_color="#2a7a2a",
            command=self.load_json_data,
            state="disabled",
        )
        self.json_load_btn.pack(side="left", fill="x", expand=True)

        self.json_progress = ctk.CTkProgressBar(json_frame)
        self.json_progress.set(0)
        self.json_progress.pack(pady=(4, 2), padx=10, fill="x")

        self.json_status_label = ctk.CTkLabel(
            json_frame, text="",
            font=ctk.CTkFont(size=9), text_color="#aaaaaa",
            wraplength=270, justify="left",
        )
        self.json_status_label.pack(pady=(2, 10), padx=10)

        # ---- Main monitor area ---------------------------------------------
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#000000")
        self.main_frame.grid(row=0, column=1, sticky="nsew")

        header = ctk.CTkFrame(self.main_frame, height=40, fg_color="#111111")
        header.pack(fill="x", side="top")
        ctk.CTkLabel(
            header, text="  LEAD II  ",
            font=ctk.CTkFont(family="Consolas", weight="bold"), text_color="#00ff00",
        ).pack(side="left")
        self.status_ind = ctk.CTkLabel(header, text="● LIVE  ", text_color="red")
        self.status_ind.pack(side="right")

        self.fig, self.ax = plt.subplots(facecolor='black')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_facecolor('black')
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_xlim(0, PLOT_POINTS)
        self.ax.grid(True, color='#222222', linestyle='-', linewidth=0.5)
        self.ax.set_xticks(np.arange(0, PLOT_POINTS, PLOT_POINTS / 10))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        self.line, = self.ax.plot(np.zeros(PLOT_POINTS), color='#00ff00', linewidth=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ======================================================================
    #  Control callbacks
    # ======================================================================

    def update_bpm_slider(self, value):
        self.engine.bpm = value
        if self.engine.waveform in ("NSR", "PVC"):
            self.bpm_val_label.configure(text=f"{int(value)} BPM")
        else:
            self.bpm_val_label.configure(text=f"{int(value)} Hz")

    def update_amp_slider(self, value):
        self.engine.target_mv = value
        if self.engine.is_calibrated:
            self.amp_val_label.configure(text=f"Output: {value:.2f} mV")
        else:
            self.amp_val_label.configure(text=f"Gain: {int((value / 5.0) * 100)}%")

    def change_mode(self, value):
        self.engine.mode = value
        if value in ("MIT-BIH", "JSON-ECG"):
            self.wave_opt.configure(state="disabled")
            self.bpm_slider.configure(state="disabled")
        else:
            self.wave_opt.configure(state="normal")
            self.bpm_slider.configure(state="normal")

    def change_waveform(self, value):
        self.engine.waveform = value
        self.update_bpm_slider(self.bpm_slider.get())

    def toggle_calibration_mode(self):
        if self.engine.mode == "Calibration":
            self.engine.mode = "Synthetic"
            self.engine.calibration_gain = self.amp_slider.get()
            self.engine.is_calibrated    = True
            self.calib_btn.configure(text="Recalibrate System", fg_color="#444")
            self.mode_label.configure(text="OPERATION MODE")
            self.amp_slider.configure(from_=0, to=5)
            self.amp_slider.set(1.0)
            self.engine.target_mv = 1.0
            self.amp_val_label.configure(text="Output: 1.00 mV")
            tk.messagebox.showinfo("Success", "System Calibrated.\nSlider now sets actual Millivolts (0–5 mV).")
        else:
            self.engine.mode = "Calibration"
            self.calib_btn.configure(text="CONFIRM 1mV ON SCOPE", fg_color="red")
            self.mode_label.configure(text="MODE: CALIBRATION (1Hz SQ)")
            self.amp_slider.configure(from_=0, to=1.0)
            self.amp_slider.set(0.1)
            self.engine.target_mv = 0.1
            self.amp_val_label.configure(text="DAC Level (Raw)")
            tk.messagebox.showinfo(
                "Calibration",
                "1. Connect Scope/Monitor.\n"
                "2. Signal is now 1Hz Square Wave.\n"
                "3. Adjust Slider until Scope shows exactly 1mVpp.\n"
                "4. Click 'CONFIRM' button.",
            )

    # ======================================================================
    #  MIT-BIH loader
    # ======================================================================

    def load_mit_data(self):
        selection = self.mit_opt.get()
        rec_id    = MIT_RECORD_MAP[selection]
        self.load_btn.configure(state="disabled", text="Downloading…")
        self.mit_progress.set(0)

        def _thread():
            success, _msg = self.engine.load_mit_record(
                rec_id, progress_callback=lambda v: self.mit_progress.set(v),
            )
            if success:
                self.after(0, lambda: (
                    self.mode_seg.set("MIT-BIH"),
                    self.change_mode("MIT-BIH"),
                    self.load_btn.configure(text="Loaded ✓", state="normal"),
                ))
            else:
                self.after(0, lambda: self.load_btn.configure(text="Error — Retry", state="normal"))

        threading.Thread(target=_thread, daemon=True).start()

    # ======================================================================
    #  JSON-ECG: directory scanner
    # ======================================================================

    def rescan_json_directory(self):
        """Kick off a background scan and refresh the dropdown."""
        self.json_scan_label.configure(text="Scanning json_data/ …", text_color="#888888")
        self.json_opt.configure(values=["— scanning —"])
        self.json_opt.set("— scanning —")
        self.json_load_btn.configure(state="disabled")

        def _scan():
            result = scan_json_directory()
            self.after(0, lambda: self._apply_scan_result(result))

        threading.Thread(target=_scan, daemon=True).start()

    def _apply_scan_result(self, result: dict):
        self.json_file_map = result
        count = len(result)

        if count == 0:
            self.json_scan_label.configure(
                text="No valid files found — drop .json files into  json_data/",
                text_color="#cc6644",
            )
            self.json_opt.configure(values=["No valid files found"])
            self.json_opt.set("No valid files found")
            self.json_load_btn.configure(state="disabled")
        else:
            s = "s" if count != 1 else ""
            self.json_scan_label.configure(
                text=f"{count} valid file{s} found", text_color="#2CC985",
            )
            labels = list(result.keys())
            self.json_opt.configure(values=labels)
            self.json_opt.set(labels[0])
            self.json_load_btn.configure(state="normal")

    # ======================================================================
    #  JSON-ECG: loader
    # ======================================================================

    def load_json_data(self):
        selection = self.json_opt.get()
        if selection not in self.json_file_map:
            return

        filepath = self.json_file_map[selection]
        self.json_load_btn.configure(state="disabled", text="Loading…")
        self.json_progress.set(0)
        self.json_status_label.configure(text="Parsing ECG records…", text_color="#aaaaaa")

        def _thread():
            success, msg = self.engine.load_json_record(
                filepath,
                progress_callback=lambda v: self.json_progress.set(v),
            )
            if success:
                self.after(0, lambda: self._on_json_loaded(msg))
            else:
                self.after(0, lambda: self._on_json_error(msg))

        threading.Thread(target=_thread, daemon=True).start()

    def _on_json_loaded(self, msg: str):
        self.mode_seg.set("JSON-ECG")
        self.change_mode("JSON-ECG")
        self.json_load_btn.configure(text="Loaded ✓", state="normal")
        self.json_status_label.configure(text=msg, text_color="#2CC985")

    def _on_json_error(self, msg: str):
        self.json_load_btn.configure(text="Error — Retry", state="normal")
        self.json_status_label.configure(text=f"Error: {msg}", text_color="#ff4444")

    # ======================================================================
    #  Audio + Plot loop
    # ======================================================================

    def _start_audio_stream(self):
        self.stream = sd.OutputStream(
            channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            callback=self.engine.audio_callback,
        )
        self.stream.start()

    def _start_scan_plot(self):
        self.y_data   = np.zeros(PLOT_POINTS)
        self.scan_idx = 0
        self._animate_scan()

    def _animate_scan(self):
        if not self.running:
            return
        try:
            while not self.engine.plot_queue.empty():
                chunk     = self.engine.plot_queue.get_nowait()
                chunk_len = len(chunk)

                if self.scan_idx + chunk_len < PLOT_POINTS:
                    self.y_data[self.scan_idx:self.scan_idx + chunk_len] = chunk
                    gap_end = min(self.scan_idx + chunk_len + 50, PLOT_POINTS)
                    self.y_data[self.scan_idx + chunk_len:gap_end] = np.nan
                    self.scan_idx += chunk_len
                else:
                    space = PLOT_POINTS - self.scan_idx
                    self.y_data[self.scan_idx:] = chunk[:space]
                    rem = chunk_len - space
                    self.y_data[:rem] = chunk[space:]
                    self.y_data[rem:rem + 50] = np.nan
                    self.scan_idx = rem

            self.line.set_ydata(self.y_data)
            self.canvas.draw_idle()
        except Exception:
            pass

        if self.running:
            self.after(20, self._animate_scan)

    def on_close(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        plt.close('all')
        self.quit()
        self.destroy()
        sys.exit()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()