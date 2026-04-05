# Requirements Document: Indic Speech Translation System

## 1. Project Overview

**Project Name:** Indic Speech Translation System  
**Version:** 2.0  
**Purpose:** Real-time English speech to Indian language translation with text-to-speech synthesis

The system provides an end-to-end pipeline for capturing English speech, translating it to Indian languages (with focus on Kannada), and generating natural-sounding speech output in the target language.

---

## 2. Functional Requirements

### 2.1 Audio Input & Processing

**FR-1.1:** The system shall capture audio input from the default microphone at 16kHz sample rate  
**FR-1.2:** The system shall support real-time audio recording with start/stop controls  
**FR-1.3:** The system shall apply spectral-based noise reduction to enhance audio quality  
**FR-1.4:** The system shall use Voice Activity Detection (VAD) to distinguish speech from noise  
**FR-1.5:** The system shall allow users to enable/disable audio enhancement  
**FR-1.6:** The system shall provide adjustable noise reduction levels (0.5 to 3.0)

### 2.2 Speech Recognition

**FR-2.1:** The system shall transcribe English speech to text using Whisper ASR  
**FR-2.2:** The system shall process enhanced audio for improved transcription accuracy  
**FR-2.3:** The system shall display transcribed English text in the UI  
**FR-2.4:** The system shall handle empty or inaudible recordings gracefully

### 2.3 Language Translation

**FR-3.1:** The system shall translate English text to the following Indian languages:
- Kannada (kan_Knda)
- Telugu (tel_Telu)
- Hindi (hin_Deva)
- Tamil (tam_Taml)
- Gujarati (guj_Gujr)
- Bengali (ben_Beng)

**FR-3.2:** The system shall use AI4Bharat's IndicTrans2 model for translation  
**FR-3.3:** The system shall display translated text in the target language script  
**FR-3.4:** The system shall support language selection via dropdown menu

### 2.4 Text-to-Speech Synthesis

**FR-4.1:** The system shall provide two TTS engines:
- Fast Kannada TTS (optimized, CPU-only)
- Standard Indic Parler-TTS (multi-language support)

**FR-4.2:** The system shall automatically select Fast TTS for Kannada when available  
**FR-4.3:** The system shall support multiple speaker voices per language  
**FR-4.4:** The system shall allow speaker selection via dropdown menu  
**FR-4.5:** The system shall generate audio output in WAV or MP3 format  
**FR-4.6:** The system shall provide auto-play option after translation

### 2.5 User Interface

**FR-5.1:** The system shall provide a graphical user interface using Tkinter  
**FR-5.2:** The UI shall display:
- Model loading status
- Recording status
- English transcription
- Translated text
- Audio enhancement settings
- TTS method indicator
- Progress bar during processing

**FR-5.3:** The UI shall provide controls for:
- Start/stop recording
- Language selection
- Speaker selection
- Audio enhancement toggle
- Noise reduction level adjustment
- Auto-play toggle
- Manual audio playback

**FR-5.4:** The UI shall indicate which TTS engine is being used (Fast/Standard)

---

## 3. Non-Functional Requirements

### 3.1 Performance

**NFR-1.1:** Model loading shall complete within 60 seconds on standard hardware  
**NFR-1.2:** Audio transcription shall process within 5 seconds for 10-second recordings  
**NFR-1.3:** Translation shall complete within 3 seconds for typical sentences  
**NFR-1.4:** Fast Kannada TTS shall synthesize speech faster than real-time (RTF < 1.0)  
**NFR-1.5:** The system shall remain responsive during background processing

### 3.2 Usability

**NFR-2.1:** The UI shall be intuitive for non-technical users  
**NFR-2.2:** Error messages shall be clear and actionable  
**NFR-2.3:** The system shall provide visual feedback for all operations  
**NFR-2.4:** Audio quality indicators shall inform users about enhancement results

### 3.3 Reliability

**NFR-3.1:** The system shall handle microphone access failures gracefully  
**NFR-3.2:** The system shall provide fallback TTS when Fast TTS fails  
**NFR-3.3:** The system shall validate audio file creation before processing  
**NFR-3.4:** The system shall recover from model loading errors with clear messages

### 3.4 Compatibility

**NFR-4.1:** The system shall run on macOS, Linux, and Windows  
**NFR-4.2:** The system shall work with CPU-only hardware (no GPU required)  
**NFR-4.3:** The system shall support Python 3.8+  
**NFR-4.4:** The system shall be compatible with standard audio input devices

### 3.5 Maintainability

**NFR-5.1:** Code shall be modular with clear separation of concerns  
**NFR-5.2:** Audio enhancement shall use documented spectral subtraction methods  
**NFR-5.3:** TTS engines shall be pluggable for easy extension  
**NFR-5.4:** Configuration shall be centralized in language mapping dictionaries

---

## 4. System Requirements

### 4.1 Hardware Requirements

**Minimum:**
- CPU: Dual-core processor (2.0 GHz+)
- RAM: 4GB
- Storage: 5GB free space
- Audio: Microphone input device

**Recommended:**
- CPU: Quad-core processor (2.5 GHz+)
- RAM: 8GB
- Storage: 10GB free space
- Audio: Quality USB microphone

### 4.2 Software Requirements

**Operating System:**
- macOS 10.14+
- Ubuntu 18.04+ / Linux
- Windows 10/11

**Python Environment:**
- Python 3.8 or higher
- pip package manager

**System Dependencies:**
- PortAudio (for PyAudio)
- FFmpeg (for audio format conversion)
- Python Tkinter (usually bundled)

### 4.3 Python Dependencies

**Core ML/AI:**
- torch >= 2.0.0
- transformers >= 4.30.0
- openai-whisper >= 20230918
- parler-tts >= 0.1.0

**Audio Processing:**
- librosa >= 0.10.0
- soundfile >= 0.12.0
- pyaudio >= 0.2.11
- webrtcvad >= 2.0.10
- scipy >= 1.10.0
- pydub >= 0.25.0

**Indian Languages:**
- IndicTransToolkit >= 1.0.0
- indic-nlp-library >= 0.81

---

## 5. Data Requirements

### 5.1 Model Files

**Required Models:**
- Whisper base model (~140MB)
- IndicTrans2 en-indic-1B model (~2GB)
- Indic Parler-TTS model (~1GB)
- Kannada FastPitch model (~100MB)
- Kannada HiFi-GAN vocoder (~50MB)

**Model Storage:**
- Models cached in user's home directory
- Kannada TTS models in `kannada_tts_fast/kn/` directory

### 5.2 Audio Files

**Temporary Files:**
- `audio_files/temp_recording.wav` - Raw recording
- `audio_files/enhanced_recording.wav` - Noise-reduced audio
- `audio_files/output.mp3` - Final TTS output
- `audio_files/fast_kannada_*.wav` - Fast TTS outputs

**File Formats:**
- Input: WAV (16kHz, mono, 16-bit PCM)
- Output: WAV or MP3

---

## 6. Security & Privacy Requirements

**SR-1:** Audio recordings shall be stored locally only  
**SR-2:** No audio data shall be transmitted to external servers (except model downloads)  
**SR-3:** Temporary audio files shall be overwritten on each recording  
**SR-4:** Users shall have control over microphone access  
**SR-5:** Model downloads shall use HTTPS connections

---

## 7. Constraints & Assumptions

### 7.1 Constraints

- Internet connection required for initial model downloads
- CPU-only operation (CUDA disabled for compatibility)
- Single-user desktop application
- Synchronous processing (one recording at a time)

### 7.2 Assumptions

- Users have working microphone hardware
- Users speak clearly in English for input
- Target language text rendering is supported by system fonts
- Audio playback is handled by system default player

---

## 8. Future Enhancements

**FE-1:** Support for additional Indian languages (Malayalam, Marathi, Punjabi)  
**FE-2:** Real-time streaming translation (no recording required)  
**FE-3:** Batch processing of audio files  
**FE-4:** Custom voice training for personalized TTS  
**FE-5:** GPU acceleration option for faster processing  
**FE-6:** Web-based interface for remote access  
**FE-7:** Conversation mode with bidirectional translation  
**FE-8:** Audio quality metrics and visualization

---

## 9. Acceptance Criteria

**AC-1:** System successfully records and processes 10-second English speech  
**AC-2:** Transcription accuracy > 90% for clear speech  
**AC-3:** Translation produces grammatically correct target language text  
**AC-4:** TTS output is intelligible and natural-sounding  
**AC-5:** Audio enhancement reduces background noise noticeably  
**AC-6:** Fast Kannada TTS works on CPU-only systems  
**AC-7:** UI remains responsive during all operations  
**AC-8:** Error messages guide users to resolution  
**AC-9:** All 6 target languages produce valid output  
**AC-10:** System runs on all three supported operating systems
