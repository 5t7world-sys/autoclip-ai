# 🎬 AutoClip AI

Paste a YouTube link → get ready-to-post highlight clips with AI-picked moments and
word-synced, styled captions. Built for Shorts, Reels and TikTok.

**Stack:** Streamlit · yt-dlp · OpenAI Whisper · GPT-4o(-mini) · FFmpeg · pycaps-ai · mediapipe

---

## 🇬🇧 English

### What it does

| Step | What happens |
|------|--------------|
| 1. Download | yt-dlp grabs `bestvideo[height<=1080]+bestaudio` and merges to MP4 |
| 2. Transcribe | OpenAI Whisper returns **word-level** timestamps |
| 3. Detect | GPT-4o-mini ranks the most viral 20–60 s moments; optional **AI Vision Mode** samples 1 frame / 5 s and scores visual interest with GPT-4o |
| 4. Cut | FFmpeg cuts each clip and crops it to 16:9, 9:16 or 1:1 |
| 5. Frame | mediapipe face detection keeps the speaker centered in vertical clips |
| 6. Caption | Word-synced captions burned in: Viral Bold, Minimal Clean or Karaoke Highlight |
| 7. Package | `clip_1.mp4`, `clip_1.srt`, … plus `autoclip_clips.zip` in `./output` |

### Requirements

- **Python 3.10+**
- **FFmpeg** (with libass) on your `PATH`
- An **OpenAI API key** — only needed for Whisper + LLM highlights
  (Demo Mode and the offline heuristic work without one)

### 1. Install FFmpeg

```bash
# Windows (PowerShell)
winget install Gyan.FFmpeg
# …or with Chocolatey
choco install ffmpeg
# …or with Scoop
scoop install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Fedora
sudo dnf install ffmpeg

# Conda (any OS)
conda install -c conda-forge ffmpeg
```

Verify: `ffmpeg -version` and `ffprobe -version`.

> The caption renderer needs an FFmpeg build **with libass**. The packages above
> all include it. Check with `ffmpeg -filters | grep subtitles`.

### 2. Install AutoClip AI

```bash
git clone <your-repo-url> autoclip-ai
cd autoclip-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Optional: the pycaps-ai caption templates (the built-in FFmpeg/ASS
# renderer works without them, and is used automatically as a fallback).
pip install pycaps-ai
playwright install chromium
```

### 3. Add your API key

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY=sk-...`, **or** paste the key into the sidebar
(it is kept in memory only, never written to disk).

Get a key: <https://platform.openai.com/api-keys> → *Create new secret key*.
Make sure the account has credit — Whisper bills per audio minute.

#### Optional `.env` settings

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | Key used when the sidebar field is empty |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat/vision model for highlight detection |
| `AUTOCLIP_OUTPUT_DIR` | `output` | Where clips, SRTs and the ZIP are written (relative paths resolve next to `app.py`) |
| `AUTOCLIP_MAX_HEIGHT` | `1080` | Maximum download height; lower it to save bandwidth |

### 4. Run

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### First run without a key

Tick **🧪 Demo Mode** in the sidebar and press **🚀 Generate Clips**. It renders a
synthetic 30-second video locally, so you can verify FFmpeg, caption rendering and
the ZIP export before spending any credits.

### Troubleshooting

| Problem | Fix |
|---|---|
| `FFmpeg not found` | Install it (step 1) and reopen the terminal so `PATH` refreshes. `ffmpeg -version` must work. |
| `This FFmpeg build has no libass support` | Install a full build: `brew install ffmpeg`, `winget install Gyan.FFmpeg`, `sudo apt install ffmpeg`. |
| `Download failed: Video unavailable` | The video is private, region-locked or removed. Try another URL. |
| `Sign in to confirm you're not a bot` | Update yt-dlp: `pip install -U yt-dlp`. |
| `openai.AuthenticationError` | Wrong/expired key, or no credit on the account. |
| `openai.RateLimitError` | You hit a quota. Wait, or lower the number of clips / disable Vision Mode. |
| `Could not parse the model's JSON answer` | Usually transient — press Generate again. |
| Transcript is empty | The video has no speech (music only). Highlights then fall back to even sampling. |
| `mediapipe` install fails | Not fatal. Auto-framing falls back to a skin-tone heuristic; or `pip install mediapipe --upgrade`. |
| pycaps errors | Just uncheck **Use pycaps-ai templates** — the built-in FFmpeg/ASS renderer is used. |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| Vertical clips look cropped | Turn off **Auto-framing** to use a plain center crop, or leave it on for face tracking. |
| Very slow on long videos | Use 16:9, turn off Vision Mode, and pick fewer clips. 30 min+ videos trigger a confirmation. |

### Project layout

```
autoclip-ai/
├── app.py            # Streamlit UI (dark theme, step progress, clip cards)
├── clipper.py        # download → transcribe → detect → cut → frame → caption → zip
├── requirements.txt
├── .env.example
├── README.md
└── output/           # generated clips, SRT files and ZIP (created on first run)
```

### Notes & limits

- Clips are generated **locally**; nothing is uploaded except audio (Whisper) and
  sampled frames (Vision Mode) to OpenAI.
- Downloading YouTube videos may violate YouTube's Terms of Service. Only clip
  content you own or are allowed to reuse.
- `cut_clip(..., fast=True)` stream-copies for speed; the pipeline re-encodes when
  a crop or rescale is required so frames stay accurate.
- Caption timing uses Whisper's word timestamps. If you run without a key, captions
  are skipped for silent footage rather than guessed.

---

## 🇮🇳 हिन्दी

### यह ऐप क्या करता है

YouTube लिंक डालिए और तैयार क्लिप्स पाइए — AI सबसे वायरल हिस्से चुनता है और
शब्द-दर-शब्द सिंक किए कैप्शन लगाता है। Shorts, Reels और TikTok के लिए।

1. **डाउनलोड** — yt-dlp वीडियो को 1080p तक MP4 में लाता है।
2. **ट्रांसक्राइब** — Whisper हर शब्द का टाइमस्टैम्प देता है।
3. **हाइलाइट डिटेक्शन** — GPT-4o-mini 20–60 सेकंड के सबसे दमदार हिस्से चुनता है।
   **AI Vision Mode** चालू हो तो हर 5 सेकंड में एक फ्रेम भी जाँचा जाता है।
4. **कटिंग** — FFmpeg क्लिप काटता है और 16:9 / 9:16 / 1:1 में क्रॉप करता है।
5. **ऑटो-फ्रेमिंग** — mediapipe चेहरा पहचानकर वर्टिकल क्लिप में स्पीकर को बीच में रखता है।
6. **कैप्शन** — Viral Bold, Minimal Clean या Karaoke Highlight स्टाइल में कैप्शन बर्न होते हैं।
7. **पैकेज** — `./output` फोल्डर में `clip_1.mp4`, `clip_1.srt` और एक ZIP फाइल।

### ज़रूरी चीज़ें

- **Python 3.10 या नया**
- **FFmpeg** (libass के साथ) — इंस्टॉल होना ज़रूरी है
- **OpenAI API key** — सिर्फ Whisper और AI हाइलाइट्स के लिए
  (Demo Mode और ऑफलाइन मोड बिना key भी चलते हैं)

### 1. FFmpeg इंस्टॉल करें

```bash
# Windows (PowerShell)
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg

# Fedora
sudo dnf install ffmpeg

# Conda (किसी भी OS पर)
conda install -c conda-forge ffmpeg
```

जाँचें: `ffmpeg -version` — अगर वर्ज़न दिखे तो सही है।
अगर `command not found` आए तो टर्मिनल बंद करके दोबारा खोलें।

### 2. ऐप इंस्टॉल करें

```bash
cd autoclip-ai

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# वैकल्पिक: pycaps-ai कैप्शन टेम्पलेट (बिल्ट-इन FFmpeg/ASS रेंडरर इसके बिना
# भी काम करता है और एरर होने पर अपने आप इस्तेमाल हो जाता है)।
pip install pycaps-ai
playwright install chromium
```

### 3. API key डालें

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

`.env` फाइल खोलकर `OPENAI_API_KEY=sk-...` लिखें।
या सीधे साइडबार में key पेस्ट कर दें (यह सिर्फ मेमोरी में रहती है, डिस्क पर सेव नहीं होती)।

Key यहाँ से बनाएँ: <https://platform.openai.com/api-keys> → *Create new secret key*।
ध्यान दें: अकाउंट में बैलेंस होना चाहिए, वरना Whisper काम नहीं करेगा।

#### वैकल्पिक `.env` सेटिंग्स

| वेरिएबल | डिफ़ॉल्ट | क्या करता है |
|---|---|---|
| `OPENAI_API_KEY` | — | साइडबार खाली होने पर यही key इस्तेमाल होती है |
| `OPENAI_MODEL` | `gpt-4o-mini` | हाइलाइट डिटेक्शन का मॉडल |
| `AUTOCLIP_OUTPUT_DIR` | `output` | क्लिप, SRT और ZIP कहाँ सेव हों (relative path `app.py` के पास बनेगा) |
| `AUTOCLIP_MAX_HEIGHT` | `1080` | डाउनलोड की अधिकतम ऊँचाई — घटाने से डेटा बचेगा |

### 4. ऐप चलाएँ

```bash
streamlit run app.py
```

ब्राउज़र में <http://localhost:8501> खुल जाएगा।

### बिना API key आज़माएँ

साइडबार में **🧪 Demo Mode** टिक करें और **🚀 Generate Clips** दबाएँ।
यह एक नकली 30 सेकंड का वीडियो बनाकर पूरा pipeline टेस्ट करता है —
बिना पैसा खर्च किए।

### समस्याएँ और हल

| समस्या | हल |
|---|---|
| `FFmpeg not found` | FFmpeg इंस्टॉल करें और टर्मिनल दोबारा खोलें। `ffmpeg -version` चलना चाहिए। |
| `has no libass support` | पूरा FFmpeg बिल्ड इंस्टॉल करें (`apt install ffmpeg` / `brew install ffmpeg`)। |
| `Video unavailable` | वीडियो private, region-locked या हटाया गया है। दूसरा लिंक डालें। |
| `Sign in to confirm you're not a bot` | `pip install -U yt-dlp` चलाकर yt-dlp अपडेट करें। |
| `AuthenticationError` | API key गलत है या अकाउंट में बैलेंस नहीं है। |
| `RateLimitError` | लिमिट खत्म — थोड़ा रुकें, क्लिप्स की संख्या घटाएँ या Vision Mode बंद करें। |
| ट्रांसक्रिप्ट खाली | वीडियो में बोली नहीं है (सिर्फ म्यूजिक)। ऐसे में बराबर अंतराल पर क्लिप्स बनेंगे। |
| mediapipe इंस्टॉल नहीं हो रहा | कोई दिक्कत नहीं — ऑटो-फ्रेमिंग दूसरा तरीका इस्तेमाल कर लेगा। |
| pycaps में एरर | साइडबार में **Use pycaps-ai templates** हटा दें — बिल्ट-इन रेंडरर चल जाएगा। |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| बहुत धीमा चल रहा है | Vision Mode बंद करें, क्लिप्स कम चुनें, 16:9 इस्तेमाल करें। |

### ज़रूरी बातें

- सारी प्रोसेसिंग **आपके कंप्यूटर पर** होती है। सिर्फ ऑडियो (Whisper) और कुछ फ्रेम
  (Vision Mode) OpenAI को भेजे जाते हैं।
- YouTube से वीडियो डाउनलोड करना YouTube की शर्तों के खिलाफ हो सकता है।
  सिर्फ अपना या अनुमति वाला कंटेंट क्लिप करें।
- 30 मिनट से लंबे वीडियो पर ऐप चेतावनी दिखाकर पुष्टि माँगता है।

---

## License

MIT — use it, fork it, ship it. Attribution appreciated.
