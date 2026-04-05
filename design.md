# Design Document: Indic Speech Translation System

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                    (Tkinter GUI - main2.py)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                   Application Logic Layer                    │
│                  (SpeechTranslatorApp Class)                 │
├──────────────────┬──────────────────┬──────────────────────┤
│  Audio Capture   │  Model Manager   │  File Management     │
└────────┬─────────┴────────┬─────────┴──────────┬───────────┘
         │                  │                     │
┌────────┴─────────┐ ┌─────┴──────────┐ ┌───────┴────────────┐
│ Audio Processing │ │ ML Models      │ │ Audio Storage      │
│ Layer            │ │ Layer          │ │ Layer              │
├──────────────────┤ ├────────────────┤ ├────────────────────┤
│ • AudioEnhancer  │ │ • Whisper ASR  │ │ • WAV files        │
│ • VAD            │ │ • IndicTrans2  │ │ • MP3 conversion   │
│ • Spectral       │ │ • Parler-TTS   │ │ • Temp storage     │
│   Subtraction    │ │ • Kannada TTS  │ │                    │
└──────────────────┘ └────────────────┘ └────────────────────┘
```

### 1.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SpeechTranslatorApp                       │
├─────────────────────────────────────────────────────────────┤
│ - language_mapping: dict                                     │
│ - is_recording: bool                                         │
│ - audio_enhancer: AudioEnhancer                              │
│ - whisper_model: WhisperModel                                │
│ - model: IndicTransModel                                     │
│ - tts_model: ParlerTTSModel                                  │
│ - kannada_tts: SurgicalKannadaTTS                            │
├─────────────────────────────────────────────────────────────┤
│ + __init__(root)                                             │
│ + load_models()                                              │
│ + toggle_recording()                                         │
│ + record_audio()                                             │
│ + enhance_recorded_audio()                                   │
│ + process_recording()                                        │
│ + synthesize_with_fast_kannada_tts()                         │
│ + play_audio()                                               │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      AudioEnhancer                           │
├─────────────────────────────────────────────────────────────┤
│ - samplerate: int                                            │
│ - frame_length: int                                          │
│ - hop_length: int                                            │
│ - alpha: float                                               │
│ - noise_threshold: float                                     │
│ - vad: WebRTCVAD                                             │
│ - noise_profile: ndarray                                     │
├─────────────────────────────────────────────────────────────┤
│ + enhance_audio_chunk(audio_chunk)                           │
│ + enhance_full_audio(audio_data)                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   SurgicalKannadaTTS                         │
├─────────────────────────────────────────────────────────────┤
│ - base_dir: Path                                             │
│ - kn_dir: Path                                               │
│ - cpu_dir: Path                                              │
├─────────────────────────────────────────────────────────────┤
│ + setup_paths()                                              │
│ + prepare_models()                                           │
│ + convert_model_cpu()                                        │
│ + synthesize(text, output_path)                              │
│ + synthesize_with_original_models()                          │
│ + synthesize_with_converted_models()                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Design

### 2.1 Main Processing Pipeline

```
[User Speaks] 
    ↓
[Microphone Capture] → PyAudio (16kHz, mono)
    ↓
[Raw Audio Buffer] → temp_recording.wav
    ↓
[Audio Enhancement] → AudioEnhancer
    ├─ STFT Transform
    ├─ VAD Detection
    ├─ Noise Profile Update
    ├─ Spectral Subtraction
    └─ ISTFT Transform
    ↓
[Enhanced Audio] → enhanced_recording.wav
    ↓
[Speech Recognition] → Whisper Model
    ↓
[English Text] → Display in UI
    ↓
[Translation] → IndicTrans2
    ├─ Preprocessing (IndicProcessor)
    ├─ Tokenization
    ├─ Model Inference
    └─ Postprocessing
    ↓
[Target Language Text] → Display in UI
    ↓
[TTS Selection]
    ├─ If Kannada + Fast TTS enabled
    │   └─ SurgicalKannadaTTS
    └─ Else
        └─ Indic Parler-TTS
    ↓
[Audio Generation]
    ↓
[Output Audio] → output.mp3 / fast_kannada_*.wav
    ↓
[Auto-play (optional)] → System Audio Player
```

### 2.2 Audio Enhancement Flow

```
Input Audio (time domain)
    ↓
[STFT] → Magnitude + Phase
    ↓
[VAD Check (20ms frame)]
    ├─ Speech Detected → Update speech counter
    └─ Noise Detected → Update noise profile
    ↓
[Spectral Subtraction]
    mask = mag / (mag + threshold * noise_profile)
    mag_enhanced = mag * mask
    ↓
[Smoothing] → Uniform filter (size=3)
    ↓
[ISTFT] → Enhanced audio (time domain)
    ↓
[Normalization] → Final output
```

### 2.3 Fast Kannada TTS Flow

```
[Kannada Text Input]
    ↓
[Model Selection]
    ├─ Try Converted CPU Models
    │   ├─ Load FastPitch (CPU)
    │   ├─ Load HiFi-GAN (CPU)
    │   └─ Load Speakers
    └─ Fallback: Original Models
    ↓
[Text Processing]
    ├─ Speaker Selection (if multi-speaker)
    └─ Text Normalization
    ↓
[FastPitch Synthesis]
    Text → Mel-spectrogram
    ↓
[HiFi-GAN Vocoding]
    Mel-spectrogram → Waveform
    ↓
[Audio Output] → WAV file
    ↓
[Format Conversion (if needed)] → MP3
```

---

## 3. Module Design

### 3.1 AudioEnhancer Module

**Purpose:** Real-time noise reduction using spectral subtraction

**Key Parameters:**
- `samplerate`: 16000 Hz (Whisper requirement)
- `frame_length`: 2048 samples (STFT window)
- `hop_length`: 512 samples (STFT hop)
- `alpha`: 0.98 (noise profile smoothing factor)
- `noise_threshold`: 1.2 (adjustable, 0.5-3.0)
- `vad_aggressiveness`: 2 (WebRTC VAD mode)

**Algorithm:**
1. Convert audio to frequency domain using STFT
2. Use VAD to detect speech vs. noise frames
3. Update noise profile during non-speech frames
4. Apply spectral mask: `mask = mag / (mag + threshold * noise_profile)`
5. Smooth enhanced magnitude with uniform filter
6. Convert back to time domain using ISTFT

**Design Rationale:**
- Based on proven spectral3.py implementation
- VAD prevents speech from being treated as noise
- Adaptive noise profile handles varying environments
- Smoothing reduces musical noise artifacts

### 3.2 SpeechTranslatorApp Module

**Purpose:** Main application controller and UI manager

**Responsibilities:**
- UI creation and event handling
- Model lifecycle management
- Audio recording coordination
- Processing pipeline orchestration
- User feedback and status updates

**Threading Model:**
- Main thread: UI event loop
- Background thread: Model loading
- Recording thread: Audio capture
- Processing thread: Translation pipeline

**State Management:**
- `is_recording`: Controls recording state
- `recorded_audio`: Buffer for audio chunks
- Model references: Lazy-loaded on startup
- UI variables: Tkinter StringVar/BooleanVar

### 3.3 SurgicalKannadaTTS Module

**Purpose:** CPU-optimized Kannada text-to-speech

**Key Features:**
- CUDA patching for CPU-only operation
- Model conversion to CPU tensors
- Dual synthesis strategies (converted/original)
- Multi-speaker support with fallbacks

**Model Preparation:**
1. Check for required model files
2. Create CPU-patched directory structure
3. Patch config files (disable CUDA)
4. Convert model checkpoints to CPU tensors
5. Cache converted models for reuse

**Synthesis Strategy:**
1. Try converted CPU models first (faster)
2. Fallback to original models if needed
3. Handle multi-speaker models gracefully
4. Return success/failure status

---

## 4. Database Design

**Note:** This system does not use a traditional database. All data is file-based.

### 4.1 File Storage Structure

```
project_root/
├── audio_files/                    # Runtime audio storage
│   ├── temp_recording.wav          # Raw microphone input
│   ├── enhanced_recording.wav      # Noise-reduced audio
│   ├── output.mp3                  # Standard TTS output
│   └── fast_kannada_*.wav          # Fast TTS outputs
│
├── kannada_tts_fast/               # Kannada TTS module
│   ├── kn/                         # Original models
│   │   ├── fastpitch/
│   │   │   ├── best_model.pth
│   │   │   ├── config.json
│   │   │   └── speakers.pth
│   │   └── hifigan/
│   │       ├── best_model.pth
│   │       └── config.json
│   └── kn_cpu_patched/             # Converted CPU models
│       ├── fastpitch/
│       └── hifigan/
│
└── ~/.cache/                       # System cache (models)
    ├── huggingface/                # Transformers models
    ├── whisper/                    # Whisper models
    └── torch/                      # PyTorch cache
```

### 4.2 Configuration Data

**Language Mapping (in-memory dictionary):**
```python
{
    "language_name": {
        "code": "lang_Script",
        "speakers": ["Speaker1", "Speaker2"],
        "default_speaker": "Speaker1",
        "fast_tts": bool
    }
}
```

---

## 5. Interface Design

### 5.1 User Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Enhanced English to Indian Language Translator             │
│  🚀 Fast Kannada TTS Available                              │
├─────────────────────────────────────────────────────────────┤
│  Status: Ready to record                                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─ Audio Enhancement Settings ──────────────────────────┐  │
│  │  ☑ Enable Audio Enhancement                           │  │
│  │  Noise Reduction Level: [━━━━━━━━━━] 1.2              │  │
│  │  ☑ Auto-play after translation                        │  │
│  └────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌─ TTS Settings ─────────────────────────────────────────┐  │
│  │  ☑ Use Fast Kannada TTS (when available)              │  │
│  └────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Target Language: [Kannada ▼]  🚀 Fast TTS                 │
├─────────────────────────────────────────────────────────────┤
│                   [ Start Recording ]                        │
│                   [████████████████]                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─ Results ──────────────────────────────────────────────┐  │
│  │  English:                                              │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ [Transcribed English text appears here]          │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  │                                                        │  │
│  │  Kannada:                                              │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │ [Translated Kannada text appears here]           │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Speaker: [Suresh ▼]                                        │
│                   [ Play Translation ]                       │
│  🚀 Fast Kannada TTS synthesis completed                    │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 UI State Transitions

```
[Initial State]
    ↓
[Loading Models] → Progress bar active, buttons disabled
    ↓
[Ready] → Record button enabled
    ↓
[Recording] → Button shows "Stop Recording", status updates
    ↓
[Processing] → Progress bar active, status shows steps
    ↓
[Complete] → Play button enabled, results displayed
    ↓
[Playing Audio] → System audio player opens
    ↓
[Ready] → Can record again
```

### 5.3 API Interfaces

**AudioEnhancer API:**
```python
class AudioEnhancer:
    def __init__(self, samplerate, frame_length, hop_length, 
                 alpha, noise_threshold, vad_aggressiveness)
    
    def enhance_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray
    def enhance_full_audio(self, audio_data: np.ndarray) -> np.ndarray
```

**SurgicalKannadaTTS API:**
```python
class SurgicalKannadaTTS:
    def __init__(self)
    
    def synthesize(self, text: str, output_path: str) -> bool
    def synthesize_with_original_models(self, text: str, output_path: str) -> bool
    def synthesize_with_converted_models(self, text: str, output_path: str) -> bool
```

---

## 6. Algorithm Design

### 6.1 Spectral Subtraction Algorithm

```
Input: audio_chunk (time-domain signal)
Output: enhanced_audio (time-domain signal)

1. Apply STFT to convert to frequency domain
   stft = librosa.stft(audio_chunk, n_fft=frame_length, hop_length=hop_length)
   magnitude = |stft|
   phase = angle(stft)

2. Compute average magnitude across time
   avg_magnitude = mean(magnitude, axis=time)

3. Voice Activity Detection
   vad_frame = audio_chunk[0:320]  # First 20ms
   is_speech = vad.is_speech(vad_frame, samplerate)

4. Update noise profile
   if not is_speech:
       noise_profile = alpha * noise_profile + (1 - alpha) * avg_magnitude
   
5. Compute spectral mask
   mask = magnitude / (magnitude + noise_threshold * noise_profile + epsilon)

6. Apply mask and smooth
   magnitude_enhanced = magnitude * mask
   magnitude_enhanced = uniform_filter(magnitude_enhanced, size=3, axis=time)

7. Reconstruct signal
   stft_enhanced = magnitude_enhanced * exp(j * phase)
   enhanced_audio = librosa.istft(stft_enhanced, hop_length=hop_length)

8. Normalize
   enhanced_audio = enhanced_audio / max(|enhanced_audio|)

Return enhanced_audio
```

### 6.2 Translation Pipeline Algorithm

```
Input: enhanced_audio_file
Output: translated_text, audio_file

1. Transcription
   result = whisper_model.transcribe(enhanced_audio_file)
   english_text = result["text"]
   
2. Preprocessing
   batch = indic_processor.preprocess_batch(
       [english_text], 
       src_lang="eng_Latn", 
       tgt_lang=target_lang_code
   )

3. Tokenization
   inputs = tokenizer(batch, return_tensors="pt")

4. Translation
   output_tokens = model.generate(
       **inputs,
       num_beams=5,
       max_length=256
   )

5. Decoding
   decoded = tokenizer.batch_decode(output_tokens)

6. Postprocessing
   translated_text = indic_processor.postprocess_batch(
       decoded, 
       lang=target_lang_code
   )[0]

7. TTS Selection
   if language == "Kannada" and use_fast_tts:
       success = kannada_tts.synthesize(translated_text, output_path)
   else:
       success = parler_tts.generate(translated_text, output_path)

Return translated_text, output_path
```

---

## 7. Security Design

### 7.1 Threat Model

**Threats:**
- Unauthorized microphone access
- Audio data interception
- Model tampering
- Malicious audio input

**Mitigations:**
- OS-level microphone permissions required
- Local-only processing (no network transmission)
- Model checksums verification (via Hugging Face)
- Input validation and sanitization

### 7.2 Data Privacy

**Privacy Principles:**
- No cloud processing (all local)
- No audio logging or persistence beyond session
- Temporary files overwritten on each use
- No telemetry or analytics

**Data Lifecycle:**
1. Audio captured → stored in temp file
2. Processing → enhanced version created
3. Translation → text displayed (not saved)
4. TTS → output audio created
5. Session end → temp files remain (user can delete)

---

## 8. Performance Design

### 8.1 Optimization Strategies

**Model Loading:**
- Lazy loading in background thread
- Models cached in memory after first load
- CPU-only operation for compatibility

**Audio Processing:**
- Chunk-based processing for real-time capability
- Vectorized NumPy operations
- Efficient STFT/ISTFT using librosa

**TTS Optimization:**
- Fast Kannada TTS uses quantized models
- CPU-optimized tensor operations
- Model conversion done once, cached

### 8.2 Performance Targets

| Operation | Target Time | Actual (Typical) |
|-----------|-------------|------------------|
| Model Loading | < 60s | 30-45s |
| 10s Audio Recording | 10s | 10s |
| Audio Enhancement | < 2s | 0.5-1s |
| Transcription (10s) | < 5s | 2-3s |
| Translation | < 3s | 1-2s |
| TTS (Fast Kannada) | < 2s | 0.5-1.5s |
| TTS (Standard) | < 5s | 3-4s |
| **Total Pipeline** | **< 20s** | **10-15s** |

### 8.3 Resource Usage

**Memory:**
- Base application: ~500MB
- Whisper model: ~140MB
- IndicTrans model: ~2GB
- TTS models: ~1GB
- Peak usage: ~4GB

**CPU:**
- Idle: < 5%
- Recording: 10-15%
- Processing: 60-90%
- TTS: 70-95%

**Storage:**
- Models: ~5GB
- Temporary audio: ~10MB per session
- Application code: ~50MB

---

## 9. Error Handling Design

### 9.1 Error Categories

**User Errors:**
- No microphone detected
- Empty recording
- Inaudible speech
- Unsupported language

**System Errors:**
- Model loading failure
- Out of memory
- Disk space exhausted
- Audio device busy

**Processing Errors:**
- Transcription failure
- Translation timeout
- TTS synthesis failure
- File I/O errors

### 9.2 Error Handling Strategy

```python
try:
    # Operation
    result = perform_operation()
except SpecificError as e:
    # Log error
    logger.error(f"Operation failed: {e}")
    
    # Update UI
    self.update_status(f"Error: {user_friendly_message}")
    
    # Show dialog
    messagebox.showerror("Error", detailed_message)
    
    # Attempt recovery
    if recoverable:
        fallback_operation()
    else:
        reset_to_ready_state()
```

### 9.3 Fallback Mechanisms

**TTS Fallback:**
```
Fast Kannada TTS fails
    ↓
Try Standard Parler-TTS
    ↓
If still fails, show error
```

**Audio Enhancement Fallback:**
```
Enhancement fails
    ↓
Copy original audio
    ↓
Continue with unenhanced audio
```

**Model Loading Fallback:**
```
Model download fails
    ↓
Check local cache
    ↓
Retry with different mirror
    ↓
Show error if all fail
```

---

## 10. Testing Strategy

### 10.1 Unit Testing

**AudioEnhancer Tests:**
- Test noise profile updates
- Test VAD integration
- Test spectral subtraction math
- Test edge cases (silence, pure noise)

**TTS Tests:**
- Test model loading
- Test CPU conversion
- Test synthesis with various texts
- Test speaker selection

### 10.2 Integration Testing

**Pipeline Tests:**
- End-to-end recording → translation → TTS
- Test all language combinations
- Test with various audio qualities
- Test error recovery paths

### 10.3 Performance Testing

**Benchmarks:**
- Measure processing time for standard inputs
- Test memory usage under load
- Test with long recordings (60s+)
- Test rapid successive recordings

### 10.4 User Acceptance Testing

**Scenarios:**
- First-time user setup
- Typical translation workflow
- Error handling (no mic, bad audio)
- Language switching
- Speaker selection

---

## 11. Deployment Design

### 11.1 Installation Process

```
1. Clone repository
2. Install system dependencies (PortAudio, FFmpeg)
3. Create Python virtual environment
4. Install Python dependencies (pip install -r requirements.txt)
5. Download Kannada TTS models
6. Run application (python main2.py)
```

### 11.2 Configuration

**User Configuration:**
- Language preferences (in-memory)
- Audio enhancement settings (UI controls)
- Speaker selection (UI controls)

**System Configuration:**
- Model paths (hardcoded, relative)
- Audio parameters (constants in code)
- Device selection (OS default)

### 11.3 Updates & Maintenance

**Model Updates:**
- Models downloaded from Hugging Face
- Automatic caching in user directory
- Manual update: delete cache, restart app

**Code Updates:**
- Git pull for latest code
- Reinstall dependencies if requirements.txt changed
- No database migrations needed

---

## 12. Design Patterns Used

### 12.1 Singleton Pattern
- Model instances (Whisper, IndicTrans, TTS) loaded once per session

### 12.2 Strategy Pattern
- TTS selection (Fast vs. Standard)
- Audio enhancement (enabled vs. disabled)

### 12.3 Observer Pattern
- UI updates via Tkinter variables
- Status updates via callback methods

### 12.4 Template Method Pattern
- Audio processing pipeline (fixed steps, variable implementations)

### 12.5 Facade Pattern
- SpeechTranslatorApp provides simple interface to complex subsystems

---

## 13. Design Decisions & Rationale

### 13.1 Why CPU-Only?
**Decision:** Force CPU operation, disable CUDA  
**Rationale:** 
- Broader hardware compatibility
- Avoid CUDA installation complexity
- Acceptable performance for speech tasks
- Reduces memory requirements

### 13.2 Why Two TTS Engines?
**Decision:** Fast Kannada TTS + Standard Parler-TTS  
**Rationale:**
- Fast TTS optimized for Kannada (primary use case)
- Standard TTS supports all languages
- Fallback mechanism for reliability
- Performance vs. flexibility trade-off

### 13.3 Why Spectral Subtraction?
**Decision:** Use spectral subtraction for noise reduction  
**Rationale:**
- Proven effective for speech enhancement
- Computationally efficient (real-time capable)
- No training required (unsupervised)
- Works well with VAD for adaptive noise profiling

### 13.4 Why Tkinter?
**Decision:** Use Tkinter for GUI  
**Rationale:**
- Bundled with Python (no extra dependency)
- Cross-platform (Windows, Mac, Linux)
- Sufficient for desktop application needs
- Lightweight and fast

### 13.5 Why Local Processing?
**Decision:** All processing done locally  
**Rationale:**
- Privacy (no audio sent to cloud)
- No API costs
- Works offline (after model download)
- Lower latency

---

## 14. Future Architecture Considerations

### 14.1 Microservices Architecture
- Separate services for ASR, Translation, TTS
- REST API for remote access
- Scalable deployment

### 14.2 Real-Time Streaming
- WebSocket for live audio streaming
- Incremental transcription
- Streaming translation

### 14.3 Multi-User Support
- User accounts and preferences
- Session management
- Audio history

### 14.4 Cloud Integration
- Optional cloud TTS for better quality
- Model serving via API
- Distributed processing

---

## 15. Appendix

### 15.1 Technology Stack

| Layer | Technology |
|-------|-----------|
| UI | Tkinter |
| Audio I/O | PyAudio |
| Audio Processing | librosa, scipy, webrtcvad |
| ASR | OpenAI Whisper |
| Translation | AI4Bharat IndicTrans2 |
| TTS | Parler-TTS, Custom Kannada TTS |
| ML Framework | PyTorch |
| Language | Python 3.8+ |

### 15.2 External Dependencies

- **Hugging Face Transformers:** Model loading and inference
- **AI4Bharat IndicTransToolkit:** Indian language processing
- **WebRTC VAD:** Voice activity detection
- **librosa:** Audio signal processing
- **PyAudio:** Microphone access

### 15.3 Model Sources

- **Whisper:** OpenAI (MIT License)
- **IndicTrans2:** AI4Bharat (MIT License)
- **Parler-TTS:** Hugging Face (Apache 2.0)
- **Kannada TTS:** AI4Bharat Indic-TTS (MIT License)
