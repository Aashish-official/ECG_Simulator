# Lifesigns ECG Simulator Workstation (v3.0.0 Web Edition)

A high-fidelity, multithreaded biomedical signal generator designed for ECG simulation, ECG-Derived Respiration (EDR) testing, and arrhythmia algorithm validation.

Transitioned to a modern client-server architecture, this workstation utilizes an advanced Python Finite State Machine (FSM) engine to generate mathematically accurate beat-by-beat morphologies and real-time analog audio output (48kHz). The visual telemetry is streamed via zero-overhead binary WebSockets to a high-performance HTML5 Canvas, recreating an authentic continuous-sweep patient monitor within any standard web browser.

## Installation and Setup Guidelines

### Step 1: Repository Acquisition

The source code can be acquired via the following methods:

**Method A: Git Clone (Recommended)**
Execute the following commands in your terminal or command prompt:

git clone https://github.com/Aashish-official/ECG_Simulator.git
cd ECG_Simulator

**Method B: ZIP Archive Download**

1. Select the **Code** dropdown at the top of the GitHub repository.
2. Select **Download ZIP**.
3. Extract the archive to your preferred local directory and navigate to it via your terminal.

### Step 2: Operating System Dependencies (Linux Environments Only)

The audio generation engine (`sounddevice`) utilizes the `portaudio` C-library to interface with system audio hardware.

* **Windows / macOS:** Proceed to Step 3 (handled automatically by standard Python package managers).
* **Linux (Ubuntu/Debian):** Execute the following commands to install required system packages:

sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio

### Step 3: Python Environment Configuration

Ensure the system is running **Python 3.8 or higher**. Install the required asynchronous web and signal processing dependencies using the provided configuration file:

pip install -r requirements.txt

## System Initialization

The workstation now operates as a local web server. To initialize the simulator, execute the primary Python server application:

python server.py

*Note: Verify that the intended audio output device (speakers or line-out) is configured as the default system audio device prior to initialization.*

**Accessing the User Interface:**
Once the terminal indicates `Uvicorn running on http://0.0.0.0:44321`, open any modern web browser (Google Chrome, Microsoft Edge, or Safari) and navigate to:
**http://localhost:44321**

## Standard Operating Procedure

The web interface is segmented into dedicated control panels situated on the left sidebar. Operational instructions for each parameter are detailed below:

### 1. Operation Mode

Select the primary data source for signal generation:

* **Synthetic:** Utilizes the FSM to generate mathematically precise, highly customizable cardiac rhythms in real-time.
* **MIT-BIH:** Downloads and streams verified clinical data from the PhysioNet arrhythmia database.
* **JSON-ECG:** Parses and plays back proprietary local LifeSigns protocol data.

*Use the dropdown menu immediately below the mode selector to dictate the specific underlying clinical rhythm (e.g., Normal Sinus Rhythm, Ventricular Bigeminy, Atrial Fibrillation).*

### 2. Heart Rate / Frequency

Regulates the operational speed of the FSM.

* Utilize the **slider** for continuous adjustments, or select the **text input field**, enter a specific integer (e.g., `135`), and press **Enter** (or click outside the box) for discrete calibration.
* *Note: This module is automatically disabled during MIT or JSON playback, as clinical datasets possess fixed native sampling rates and heart rates.*

### 3. Morphology Controls (Synthetic Mode Only)

Adjust the structural parameters of the generated FSM complexes:

* **P-Wave Amp:** Increase to simulate atrial enlargement; decrease to simulate flattened P-waves.
* **Q-Wave Depth:** Shift left to generate deep, pathological Q-waves (indicative of prior myocardial infarction).
* **T-Wave Amp:** Shift right for peaked T-waves (hyperkalemia); shift left for inverted T-waves (myocardial ischemia).
* **ST Elevation:** Displace the ST segment vertically (STEMI) or horizontally (ST depression).
* **Ectopics/Min:** When an intermittent waveform (e.g., *Occasional PVC*) is active, this parameter governs the statistical probability of the ectopic event per minute.

### 4. Signal Amplitude & Hardware Calibration

To guarantee that the analog voltage output correlates precisely to a 1mV amplitude on connected clinical hardware:

1. Select the **Step 1: Calibrate Hardware** button. The interface will lock, and the audio hardware will output a stable 1Hz Square Wave.
2. Monitor the connected physical oscilloscope or patient monitor.
3. Adjust the **Gain** slider until the rendered square wave measures exactly **1mV peak-to-peak** on your physical hardware.
4. Select **CONFIRM 1mV ON SCOPE**. The system will log this calibration scaling factor, restrict the slider to millivolt values, and resume standard simulation.

### 5. Respiratory Modulation / EDR

This module is designed for the validation of ECG-Derived Respiration (EDR) algorithms. It synthesizes three distinct physiological respiratory artifacts overlaid onto the primary ECG vector:

* **Resp Rate (RPM):** The frequency of the simulated respiratory cycle (default: 15 breaths/min).
* **RSA Depth (BPM):** Respiratory Sinus Arrhythmia. Induces vagal modulation, causing the heart rate to accelerate during inspiration and decelerate during expiration.
* **Thoracic AM:** Amplitude Modulation. Induces proportional scaling of the QRS complex amplitude synchronously with lung inflation (configurable from 0% to 50%).
* **Wander (mV):** Induces a low-frequency isoelectric baseline shift.

### 6. Database Loaders

* **MIT-BIH Database:** Select a clinical record from the dropdown and click **Load**. The server will fetch the data from PhysioNet, resample it to 48kHz, and immediately begin simulation.
* **JSON Data:** To utilize proprietary LifeSigns recordings, allocate the respective JSON files into the `json_data/` directory. Click **Rescan** to populate the dropdown, select a file, and click **Load**.

## Proprietary JSON Data Integration

### Expected JSON Format (Protocol V1.1)

The parser requires a top-level dictionary containing an `admission_id` string and an `ecg_records` array. The raw analog voltage values must be structured within a nested array under the `"value"` key.

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

*The backend server automatically computes the native sample rate by evaluating the `start_utc` and `end_utc` boundaries against the total sample count. It defaults to 250Hz if temporal data is unavailable.*

## Troubleshooting and Diagnostics

**1. Audio Output Failure or PortAudio Exceptions**
* *Cause:* Python cannot detect an active audio output interface.
* *Resolution:* Connect an audio output device or enable a virtual audio cable in the OS settings. On Linux environments, verify that `portaudio19-dev` is installed.

**2. "Address already in use" Error on Startup**
* *Cause:* Port 8000 is currently occupied by another application or a ghost process from a previous simulator run.
* *Resolution:* Terminate the process utilizing the port, or restart your terminal instance.

**3. Status Badge Indicates "DISCONNECTED"**
* *Cause:* The browser has lost the WebSocket connection to the Python server.
* *Resolution:* Ensure `server.py` is actively running in your terminal. The browser will automatically attempt to reconnect every 1000ms. If the server was halted, restart it and refresh the webpage.

**4. JSON Files Do Not Appear in Dropdown**
* *Cause:* The files are either missing the `ecg_records` key or contain an empty array (`[]`).
* *Resolution:* The high-speed Regex scanner deliberately skips empty or malformed files to prevent runtime errors. Verify the integrity of the JSON payload.

## Technical Support and Developer Contact

For technical guidance, algorithmic logic inquiries, or anomaly reporting within the FSM pipeline, please contact the primary developer:

**Aashish Srinivasan**

Senior Product Development Engineer | Lifesigns

Email: aashish.srinivasan@lifesigns.in