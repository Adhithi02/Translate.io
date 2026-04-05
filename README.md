# 🎙️ Indic Speech Translation System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI4Bharat](https://img.shields.io/badge/Powered%20by-AI4Bharat-orange.svg)](https://ai4bharat.org/)

A comprehensive real-time speech translation system that converts English speech to multiple Indian languages with advanced noise reduction and high-quality text-to-speech synthesis. Built with state-of-the-art models from [AI4Bharat](https://ai4bharat.org/) optimised for speed.

Perfect for voice-based interfaces, accessibility tools, multilingual communication systems, and language learning applications.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **🎤 Real-Time Audio Capture** - Record directly from your microphone with live feedback
- **🔇 Advanced Noise Reduction** - Spectral subtraction with Voice Activity Detection (VAD)
- **�️ Speech Recognition** - Powered by OpenAI Whisper for accurate English transcription
- **🌐 Multi-Language Translation** - Support for 6 Indian languages using IndicTrans2
- **🔊 Dual TTS Engines** - Fast optimized Kannada TTS + Standard multi-language TTS
- **�️ User-Friendly GUI** - Intuitive Tkinter interface with real-time status updates

### 🎨 Advanced Features
- ✅ Adjustable noise reduction levels (0.5x to 3.0x)
- ✅ Multiple speaker voices per language
- ✅ Auto-play option after translation
- ✅ CPU-only operation (no GPU required)
- ✅ Audio quality indicators and enhancement metrics
- ✅ Cross-platform support (macOS, Linux, Windows)

---

## 🌍 Supported Languages

| Language | Script | Code | Speakers | Fast TTS |
|----------|--------|------|----------|----------|
| **Kannada** | ಕನ್ನಡ | kan_Knda | Suresh, Anu, Chetan, Vidya | ✅ Yes |
| **Telugu** | తెలుగు | tel_Telu | Prakash, Lalitha, Kiran | ❌ No |
| **Hindi** | हिन्दी | hin_Deva | Ravi, Priya, Amit | ❌ No |
| **Tamil** | தமிழ் | tam_Taml | Arun, Meena | ❌ No |
| **Gujarati** | ગુજરાતી | guj_Gujr | Jignesh, Kavita | ❌ No |
| **Bengali** | বাংলা | ben_Beng | Rahul, Mou | ❌ No |

---

## 🏗️ Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│  Microphone │ -> │ Noise Filter │ -> │   Whisper   │ -> │ English  │
│   Input     │    │   (VAD+STFT) │    │     ASR     │    │   Text   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│   Audio     │ <- │  TTS Engine  │ <- │ IndicTrans2 │ <- │  Target  │
│   Output    │    │ (Fast/Std)   │    │ Translation │    │ Language │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
```

---

## 🛠️ Installation

### Prerequisites

**System Requirements:**
- Python 3.8 or higher
- 4GB+ RAM (8GB recommended)
- 5GB+ free disk space
- Microphone input device
- Internet connection (for initial model downloads)

**System Dependencies:**

<details>
<summary><b>macOS</b></summary>

```bash
brew install portaudio ffmpeg
```
</details>

<details>
<summary><b>Ubuntu/Debian</b></summary>

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-tk ffmpeg
```
</details>

<details>
<summary><b>Windows</b></summary>

1. Install FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Download PyAudio wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)
3. Install: `pip install <downloaded-wheel-file>`
</details>

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arushi-vaidya/Indic-Speech-Translation.git
   cd Indic-Speech-Translation
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   
   # macOS/Linux
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download Kannada TTS models** (for Fast TTS):
   ```bash
   cd kannada_tts_fast
   wget https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/kn.zip
   unzip kn.zip
   
   # Create model directory structure
   mkdir -p models/v1
   cp -r kn models/v1/
   python3 mac_optimize.py
   python3 kannada_tts.py
   cd ..
   ```

5. **Run the application:**
   ```bash
   cp main2.py IndicTrans/
   cd IndicTrans
   python3 main2.py
   ```

---

## 🚀 Quick Start Guide

### Basic Usage

1. **Launch the application:**
   ```bash
   python3 main2.py
   ```

2. **Wait for models to load** (30-60 seconds on first run)

3. **Select your target language** from the dropdown menu

4. **Click "Start Recording"** and speak in English

5. **Click "Stop Recording"** when finished

6. **View results:**
   - English transcription appears in the first text box
   - Translated text appears in the second text box
   - Audio automatically plays (if auto-play is enabled)

### Advanced Options

**Audio Enhancement Settings:**
- Toggle noise reduction on/off
- Adjust noise reduction level (1.2 is default)
- Enable/disable auto-play

**TTS Settings:**
- Choose between Fast TTS (Kannada only) or Standard TTS
- Select different speaker voices
- Output format: WAV or MP3

---

## 📁 Project Structure

```
Indic-Speech-Translation/
├── main2.py                      # Main application entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── requirements.md               # Detailed requirements specification
├── design.md                     # System design documentation
│
├── Noise_Supression/             # Audio enhancement module
│   ├── spectral3.py              # Spectral subtraction implementation
│   ├── requirements.txt          # Module-specific dependencies
│   └── README.md                 # Noise suppression documentation
│
├── kannada_tts_fast/             # Fast Kannada TTS module
│   ├── kannada_tts.py            # Main TTS implementation
│   ├── mac_optimize.py           # macOS optimization script
│   ├── universal_optimize.py     # Cross-platform optimization
│   ├── kn/                       # Original Kannada models
│   │   ├── fastpitch/            # FastPitch TTS model
│   │   └── hifigan/              # HiFi-GAN vocoder
│   └── models/v1/kn/             # Model copies for compatibility
│
├── IndicTrans2/                  # Translation toolkit (git submodule)
├── IndicTransToolkit/            # Indian language processing
│
└── audio_files/                  # Runtime audio storage (created on run)
    ├── temp_recording.wav        # Raw microphone input
    ├── enhanced_recording.wav    # Noise-reduced audio
    └── output.mp3                # Final TTS output
```

---

## 🔧 Configuration

### Audio Enhancement Parameters

Edit these in `main2.py` or adjust via UI:

```python
AudioEnhancer(
    samplerate=16000,           # Audio sample rate (Hz)
    frame_length=2048,          # STFT window size
    hop_length=512,             # STFT hop size
    alpha=0.98,                 # Noise profile smoothing (0-1)
    noise_threshold=1.2,        # Noise reduction strength (0.5-3.0)
    vad_aggressiveness=2        # VAD sensitivity (0-3)
)
```

### Language Configuration

Add new languages in the `language_mapping` dictionary:

```python
"NewLanguage": {
    "code": "lang_Script",
    "speakers": ["Speaker1", "Speaker2"],
    "default_speaker": "Speaker1",
    "fast_tts": False
}
```

---

## 🧪 How It Works

### 1. Audio Capture & Enhancement
- Captures audio at 16kHz mono from microphone
- Applies Short-Time Fourier Transform (STFT)
- Uses WebRTC VAD to detect speech vs. noise
- Updates noise profile during non-speech frames
- Applies spectral subtraction mask
- Converts back to time domain with ISTFT

### 2. Speech Recognition
- Enhanced audio fed to OpenAI Whisper model
- Transcribes English speech to text
- Handles various accents and speaking styles

### 3. Translation
- English text preprocessed with IndicProcessor
- Tokenized and fed to IndicTrans2 model
- Beam search decoding (5 beams) for quality
- Postprocessed to target language script

### 4. Text-to-Speech
- **Fast Kannada TTS:** CPU-optimized FastPitch + HiFi-GAN
- **Standard TTS:** Indic Parler-TTS for all languages
- Automatic fallback if primary TTS fails
- Output saved as WAV/MP3

---

## 📊 Performance Benchmarks

| Operation | Time (Typical) | Hardware |
|-----------|----------------|----------|
| Model Loading | 30-45s | First run only |
| 10s Recording | 10s | Real-time |
| Audio Enhancement | 0.5-1s | CPU |
| Transcription | 2-3s | CPU |
| Translation | 1-2s | CPU |
| Fast Kannada TTS | 0.5-1.5s | CPU |
| Standard TTS | 3-4s | CPU |
| **Total Pipeline** | **10-15s** | **CPU-only** |

*Tested on: MacBook Pro M1, 16GB RAM*

---

## 🐛 Troubleshooting

<details>
<summary><b>No audio recorded / Microphone not working</b></summary>

- Check microphone permissions in system settings
- Verify microphone is set as default input device
- Test microphone with other applications
- Try running with `sudo` (Linux only)
</details>

<details>
<summary><b>Model loading fails</b></summary>

- Ensure stable internet connection
- Check available disk space (need 5GB+)
- Clear Hugging Face cache: `rm -rf ~/.cache/huggingface`
- Retry download
</details>

<details>
<summary><b>Fast Kannada TTS not working</b></summary>

- Verify models are in `kannada_tts_fast/kn/` directory
- Run optimization: `cd kannada_tts_fast && python3 mac_optimize.py`
- Check for error messages in console
- Fallback to Standard TTS will be used automatically
</details>

<details>
<summary><b>Poor transcription quality</b></summary>

- Enable audio enhancement
- Increase noise reduction level
- Speak clearly and closer to microphone
- Reduce background noise
- Check microphone quality
</details>

<details>
<summary><b>Out of memory errors</b></summary>

- Close other applications
- Use smaller Whisper model (change `base` to `tiny` in code)
- Reduce recording length
- Increase system swap space
</details>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit your changes:** `git commit -m 'Add amazing feature'`
4. **Push to branch:** `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Areas for Contribution
- Additional language support
- UI/UX improvements
- Performance optimizations
- Bug fixes and testing
- Documentation improvements

---

## 📚 Documentation

- **[requirements.md](requirements.md)** - Detailed functional and non-functional requirements
- **[design.md](design.md)** - System architecture and design decisions
- **[Noise_Supression/README.md](Noise_Supression/README.md)** - Audio enhancement details
- **[kannada_tts_fast/README.md](kannada_tts_fast/README.md)** - Fast TTS implementation

---

## 🙏 Acknowledgments

This project is built on excellent work from:

- **[AI4Bharat](https://ai4bharat.org/)** - IndicTrans2 and Indic-TTS models
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Speech recognition
- **[Hugging Face](https://huggingface.co/)** - Model hosting and Parler-TTS
- **[Coqui TTS](https://github.com/coqui-ai/TTS)** - TTS framework

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- Whisper: MIT License
- IndicTrans2: MIT License
- Parler-TTS: Apache 2.0 License
- Indic-TTS: MIT License

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/arushi-vaidya/Indic-Speech-Translation/issues)
- **Discussions:** [GitHub Discussions](https://github.com/arushi-vaidya/Indic-Speech-Translation/discussions)

---

## 🗺️ Roadmap

- [ ] Add support for Malayalam, Marathi, Punjabi
- [ ] Real-time streaming translation (no recording needed)
- [ ] Web-based interface
- [ ] Mobile app (iOS/Android)
- [ ] Bidirectional translation (Indian languages to English)
- [ ] Conversation mode with turn-taking
- [ ] GPU acceleration option
- [ ] Docker containerization
- [ ] REST API for integration

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the Indian language community**
