# Lifesigns ECG Simulator Workstation (v2.0.0)

A high-fidelity, multithreaded biomedical signal generator designed for ECG simulation, ECG-Derived Respiration (EDR) testing, and arrhythmia algorithm validation.

Built with an advanced Finite State Machine (FSM) engine, this workstation generates mathematically accurate beat-by-beat morphologies, physiologic compensatory pauses, and real-time analog audio output (48kHz) synchronized with a continuous-sweep PyQtGraph patient monitor.

## Installation and Setup Guidelines

### Step 1: Repository Acquisition

The source code can be acquired via the following methods:

**Method A: Git Clone (Recommended)**
Execute the following commands in your terminal or command prompt:

```bash
git clone [https://github.com/Aashish-official/ECG_Simulator.git](https://github.com/Aashish-official/ECG_Simulator.git)
cd ECG_Simulator
```

**Method B: ZIP Archive Download**

1. Select the **Code** dropdown at the top of the GitHub repository.
2. Select **Download ZIP**.
3. Extract the archive to your preferred local directory and navigate to it via your terminal.

### Step 2: Operating System Dependencies (Linux Environments Only)

The audio generation engine (`sounddevice`) utilizes the `portaudio` C-library to interface with system audio hardware.

* **Windows / macOS:** Proceed to Step 3 (handled automatically by standard package managers).
* **Linux (Ubuntu/Debian):** Execute the following commands to install required system packages:

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Step 3: Python Environment Configuration

Ensure the system is running **Python 3.8 or higher**. Install the required dependencies using the provided configuration file:

```bash
pip install -r requirements.txt
```

## System Initialization

To initialize the simulator, execute the primary Python application:

```bash
python ECG_Workstation.py
```

*Note: Verify that the intended audio output device (speakers or line-out) is configured as the default system audio device prior to initialization.*

## Standard Operating Procedure

The user interface is segmented into dedicated control panels. Operational instructions for each parameter are detailed below:

### 1. Operation Mode

Select the primary data source for signal generation:

* **Synthetic:** Utilizes the FSM to generate mathematically precise, highly customizable cardiac rhythms in real-time.
* **MIT-BIH:** Downloads and streams verified clinical data from the PhysioNet arrhythmia database.
* **JSON-ECG:** Parses and plays back proprietary local LifeSigns protocol data.

### 2. Heart Rate / Frequency

Regulates the operational speed of the FSM.

* Utilize the **slider** for continuous adjustments, or select the **text input field**, enter a specific integer (e.g., `135`), and press **Enter** for discrete calibration.
* *Note: This module is automatically disabled during MIT or JSON playback, as clinical datasets possess fixed native sampling rates and heart rates.*

### 3. Morphology Controls (Synthetic Mode Only)

Adjust the structural parameters of the generated FSM complexes:

* **P-Wave Amp:** Increase to simulate atrial enlargement; decrease to simulate flattened P-waves.
* **Q-Wave Depth:** Shift left to generate deep, pathological Q-waves (indicative of prior myocardial infarction).
* **T-Wave Amp:** Shift right for peaked T-waves (hyperkalemia); shift left for inverted T-waves (myocardial ischemia).
* **ST Elevation:** Displace the ST segment vertically (STEMI) or horizontally (ST depression).

### 4. Arrhythmia and Ectopy Configuration

When an intermittent waveform (e.g., *Occasional PVC*) is active, this parameter governs the statistical probability of the ectopic event.

* Example: Setting the value to `5` will average 5 ectopic events per minute, distributed stochastically.

### 5. Respiratory Modulation / EDR

This module is designed for the validation of ECG-Derived Respiration (EDR) algorithms. It synthesizes three distinct physiological respiratory artifacts:

* **Resp Rate (RPM):** The frequency of the simulated respiratory cycle (default: 15 breaths/min).
* **RSA Depth:** Respiratory Sinus Arrhythmia. Induces vagal modulation, causing the heart rate to accelerate during inspiration and decelerate during expiration by the specified BPM variance.
* **Thoracic AM:** Amplitude Modulation. Induces proportional scaling of the QRS complex amplitude synchronously with lung inflation (configurable from 0% to 50%).
* **Base Wander:** Induces a low-frequency isoelectric baseline shift (measured in mV).

### 6. Hardware Calibration

To guarantee that the analog voltage output correlates precisely to a 1mV amplitude on connected clinical hardware:

1. Select **Step 1: Calibrate Hardware**. The interface will lock and output a stable 1Hz Square Wave.
2. Monitor the connected physical oscilloscope or patient monitor.
3. Adjust the **Master Gain** slider until the rendered square wave measures exactly **1mV peak-to-peak**.
4. Select **CONFIRM 1mV ON SCOPE**. The system will log this calibration scaling factor and resume standard simulation.

## Proprietary JSON Data Integration

To utilize proprietary LifeSigns recordings, allocate the respective JSON files into a directory named `json_data/` located in the root application folder.

### Expected JSON Format (Protocol V1.1)

The parser requires a top-level dictionary containing an `admission_id` string and an `ecg_records` array. The raw analog voltage values must be structured within a nested array under the `"value"` key.

```json
{
  "admission_id": "PT-99812",
  "sample_no": 1,
  "start_utc": "2024-01-01T10:00:00Z",
  "end_utc": "2024-01-01T10:05:00Z",
  "facility_id": "ICU-A",
  "packet_count": 300,
  "ecg_records": [
    {
      "timestamp": 1704103200,
      "value": [
        [0.0, 0.15, 0.85, -0.2, 0.0, 0.1, 0.0]
      ]
    }
  ]
}
```

*The software automatically computes the native sample rate by evaluating the `start_utc` and `end_utc` boundaries against the total sample count. It defaults to 250Hz if temporal data is unavailable.*

## Troubleshooting and Diagnostics

**1. "No Audio Device Found" / PortAudio Exceptions**

* *Cause:* Python cannot detect an active audio output interface.
* *Resolution:* Connect an audio output device or enable a virtual audio cable in the OS settings. On Linux environments, verify that `portaudio19-dev` is installed.

**2. Plot Monitor Fails to Render (Blank/Black Screen)**

* *Cause:* The data buffer processed `NaN` (Not a Number) or `Inf` values, causing the PyQtGraph visual bounds to collapse.
* *Resolution:* Select the **Restart Plot Monitor** button in the footer. The v5.1 DSP pipeline automatically sanitizes mathematical anomalies, restricting this failure mode primarily to extreme window resize events.

**3. Application Hangs During "Browse JSON Files"**

* *Cause:* High latency disk I/O when processing large volumes of JSON files.
* *Resolution:* The v5.1 FSM delegates this process to a background daemon thread. Allow several seconds for execution. Review the terminal output for specific JSON parsing exceptions if the window fails to instantiate.

**4. Legacy Matplotlib Backend Errors (`_tkinter.TclError`)**

* *Resolution:* This application relies exclusively on `pyqtgraph` to prevent Tkinter event-loop starvation. Ensure legacy Matplotlib integrations are fully uninstalled or operate within a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

## Technical Support and Developer Contact

For technical guidance, algorithmic logic inquiries, or anomaly reporting within the FSM pipeline, please contact the primary developer:

**Aashish Srinivasan**
Senior Product Development Engineer | Lifesigns
Email: aashish.srinivasan@lifesigns.in