# AI Video Assistant with RAG

Turn any video or audio into a searchable, readable meeting record.

Point it at a YouTube link or upload a file, and it will transcribe the audio,
generate a title and summary, extract action items / key decisions / open
questions, and let you chat with the content using retrieval-augmented
generation (RAG).

---

## Features

| Capability | What it does |
|---|---|
| **Flexible input** | YouTube URL (via `yt-dlp`) or a local `mp3 / wav / m4a / mp4 / mov / mkv / webm` file |
| **Dual transcription** | Local **Whisper** for English, **Sarvam AI** STT-translate for Hinglish → English |
| **Map-reduce summary** | Long transcripts are chunked, summarized per-chunk, then combined |
| **Insight extraction** | Action items (with owner + deadline), key decisions, unresolved questions |
| **RAG chat** | Ask questions about the video; answers grounded in the transcript via Chroma + local embeddings |
| **Two interfaces** | A Streamlit web app and a terminal CLI |
| **Export** | Download the full transcript as `.txt` |

---

## Tech Stack

- **UI** — Streamlit
- **Acquisition** — `yt-dlp`, `pydub`
- **Transcription** — `openai-whisper` (local), Sarvam `speech-to-text-translate` API
- **Orchestration** — LangChain (LCEL runnables)
- **LLM** — Groq, running `openai/gpt-oss-120b`
- **Retrieval** — ChromaDB + `all-MiniLM-L6-v2` (HuggingFace, CPU)
- **Config** — `python-dotenv`

---

## Requirements

- **Python 3.11** (developed on 3.11.15)
- **FFmpeg** — required on your `PATH` for audio extraction and conversion
- **API keys** — Groq (always) and Sarvam (only for the Hinglish path)

<details>
<summary>Installing FFmpeg on Windows</summary>

```bash
winget install Gyan.FFmpeg
```

Then restart your terminal so the new `PATH` takes effect.
</details>

---

## Installation

```bash
git clone https://github.com/Akki282/-AI-Video-Assistant-With-RAG.git
cd -AI-Video-Assistant-With-RAG

python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install streamlit==1.38.0 yt-dlp==2024.8.6 openai-whisper==20231117 ffmpeg-python==0.2.0 pydub==0.25.1 requests==2.34.2 langchain==0.2.16 langchain-core==0.2.38 langchain-community==0.2.16 langchain-groq==0.1.5 langchain-text-splitters==0.2.4 langchain-chroma==0.1.4 chromadb==0.5.23 sentence-transformers==3.0.1 langchain-huggingface==0.0.3 transformers==4.46.3 python-dotenv==1.0.1 pydantic==2.8.2 numpy==1.26.4 torch==2.3.1
```

> **Why these versions are pinned** — the `langchain-chroma==0.1.4` /
> `chromadb==0.5.23` pair constrains `tokenizers`, so `transformers` must stay at
> `4.46.3`. Letting pip resolve `transformers` freely pulls `tokenizers>=0.22`
> and breaks the Chroma install. Install the list above as-is.

> `torch` is a large download (~2 GB on CPU builds). It is required even on the
> Hinglish/Sarvam path because the pinned `transformers` and
> `sentence-transformers` versions depend on it.

---

## Configuration

Create a `.env` file in the project root:

```bash
GROQ_API_KEY="gsk_..."
SARVAM_API_KEY="sk_..."
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Groq LLM access for summaries, extraction, and chat |
| `SARVAM_API_KEY` | Hinglish only | — | Sarvam speech-to-text-translate |
| `WHISPER_MODEL` | No | `small` | Any Whisper size: `tiny`, `base`, `small`, `medium`, `large` |
| `SARVAM_STT_MODEL` | No | `saaras:v2.5` | Sarvam STT model, e.g. `Bulbul:v3` |

`.env` is gitignored, so your keys stay local.

> **Note on model size** — `small` and above transcribe noticeably better but
> run much slower on CPU. `base` is a reasonable starting point for quick tests.

---

## Usage

### Streamlit web app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Pick a source in the sidebar, choose a
language, and click **🚀 Run Pipeline**. Results are split across three tabs:
**Summary & Insights**, **Full Transcript**, and **Chat**.

### Command line

```bash
python main.py
```

You are prompted for a source and language. After the pipeline finishes the CLI
drops into an interactive chat loop — type `exit`, `quit`, or `q` to leave.

```
Enter YouTube URL or local file path: https://youtube.com/watch?v=dQw4w9WgXcQ
Language (english/hinglish): english
```

---

## How It Works

```
source (URL or file)
      │
      ▼
utils/audio_processor.py ── download / convert to WAV (mono, 44.1 kHz)
      │                        and split into 10-minute chunks
      ▼
core/transcriber.py ────── routes by language:
      │                     english → local Whisper
      │                     hinglish → Sarvam (25 s pieces, translated to English)
      ▼
core/summarizer.py ─────── map-reduce summary + title generation
core/extractor.py ──────── action items, key decisions, open questions
core/vector_store.py ───── embed chunks → ChromaDB (persisted in ./vector_db)
      │
      ▼
core/rag_engine.py ─────── similarity retriever (k=4) → grounded chat answers
```

**Why Sarvam is chunked into 25 s pieces** — Sarvam's synchronous
speech-to-text-translate endpoint rejects audio longer than 30 seconds, so
`transcribe_chunk_sarvam` slices each 10-minute chunk into 25-second pieces
(a 5-second safety margin), sends them one at a time, and joins the results.

**RAG grounding** — the chat chain is told to answer *only* from the retrieved
context and to reply `I could not find that in the meeting.` when the answer is
absent, which keeps it from inventing content.

---

## Project Structure

```
.
├── app.py                  # Streamlit web interface
├── main.py                 # CLI entry point
├── .env                    # Your API keys (gitignored)
├── core/
│   ├── transcriber.py      # Whisper / Sarvam routing + chunk assembly
│   ├── summarizer.py       # Map-reduce summary, title generation
│   ├── extractor.py        # Action items, decisions, open questions
│   ├── vector_store.py     # Chroma + HuggingFace embeddings
│   └── rag_engine.py       # RAG chain + extraction chains
├── utils/
│   └── audio_processor.py  # Download, convert, chunk
├── vector_db/              # Persisted Chroma data (gitignored)
└── downloads/              # yt-dlp output (gitignored)
```

> There is no `requirements.txt` in this repo — the pinned install command lives
> in [Installation](#installation) so the exact working versions travel with the
> documentation.

---

## Troubleshooting

**`FileNotFoundError: 'ffmpeg'`**
FFmpeg is not installed or not on your `PATH`. See [Requirements](#requirements).

**`Sarvam returned 400`**
The audio piece exceeded Sarvam's 30-second limit, or the model name in
`SARVAM_STT_MODEL` is not valid for your key.

**`RuntimeError: SARVAM_API_KEY is not set`**
You ran the Hinglish path without a Sarvam key in `.env`. Switch the language to
`english`, or add the key.

**Whisper is very slow**
It runs on CPU. Use a smaller `WHISPER_MODEL` (e.g. `base` or `tiny`), or switch
to the Hinglish path to use the Sarvam API instead.

**Out of memory on long videos**
The map-reduce summary sends 3000-character chunks to the LLM. Reduce
`chunk_size` in `core/summarizer.py:16`, or split the source into shorter clips.

**Duplicate insight functions**
`core/extractor.py` and `core/rag_engine.py` both define
`extract_action_items` / `extract_key_decisions` / `extract_questions`. Both
entry points import from `core.extractor`, so the copies in `rag_engine.py` are
unused and can be deleted.

---

## License

MIT
