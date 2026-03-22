# Armenian Voice AI Support Agent

An end-to-end voice AI customer support agent for Armenian banks, built with the open-source LiveKit framework. The agent understands and speaks Armenian, answers questions about **loans**, **deposits** and **branch locations** for three banks: **Evoca**, **Ameria** and **ACBA**.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Flow](#pipeline-flow)
- [Model Selection Process](#model-selection-process)
- [Data Pipeline](#data-pipeline)
- [RAG System](#rag-system)
- [Known Limitations](#known-limitations)
- [Scalability](#scalability)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Costs](#costs)

---

## Architecture Overview

The system follows a modular pipeline architecture:

```
User Speech (Armenian)
        ▼
Silero VAD  - Voice Activity Detection - detects when user starts/stops speaking
         ▼
NeMo STT  - NVIDIA FastConformer - transcribes Armenian speech to text
         ▼
Topic Detection -  Keyword matching - classifies query as loans/deposits/branches
         ▼
RAG Retrieval - ChromaDB + Armenian embeddings - fetches relevant bank data chunks
         ▼
Groq LLM - Llama 3.3 70B - generates Armenian response using retrieved context
         ▼
Google Cloud TTS - hy-AM voice - synthesizes Armenian speech from text
         ▼
   Agent Speech (Armenian)
```

**Framework:** LiveKit open-source Agents SDK orchestrates the entire pipeline — VAD, STT, LLM and TTS are wired together as a real-time voice agent.

---

## Pipeline Flow

1. **User speaks** in Armenian into the microphone
2. **Silero VAD** detects when the user finishes speaking
3. **NeMo STT** transcribes the complete utterance to Armenian text (~1s on CPU)
4. **Topic detection** classifies the query using Armenian keyword matching
5. **RAG retrieval** fetches the top relevant chunks from ChromaDB using Armenian text embeddings
6. **Groq LLM** generates a concise Armenian answer based only on the retrieved context (~1-2s)
7. **Google Cloud TTS** synthesizes the response as Armenian speech (~0.7s)
8. **Agent speaks** the response back to the user

**Typical end-to-end latency: 3-5 seconds** (measured on CPU, no GPU).

---

## Model Selection Process

Building a voice agent for Armenian presented many challenges. Most AI models have limited or no Armenian support. Below is a detailed account of every model evaluated, why each was chosen or rejected, and the reasoning behind the final selections.

### Speech-to-Text (STT)

| Model | Status | Reason |
|-------|--------|--------|
| Groq Whisper (whisper-large-v3) | Rejected | Very poor Armenian accuracy, required speaking extremely clearly and slowly. Even with Armenian prompt hints, it frequently misheard words. |
| Google Cloud Speech-to-Text v2 | Rejected | Sometimes romanized Armenian text (output Latin characters instead of Armenian script), making it unusable for the downstream Armenian LLM pipeline. |
| OpenAI Whisper API | Rejected | Too expensive for continuous voice agent use. |
| Gladia | Rejected | Poor Armenian transcription quality. |
| ElevenLabs Scribe v2 | Rejected | Tested but NeMo provided better Armenian accuracy. |
| **NVIDIA NeMo FastConformer** | **Selected** | `nvidia/stt_hy_fastconformer_hybrid_large_pc` — the best available open-source Armenian ASR model. Trained specifically for Armenian with high accuracy. Uses CTC decoder mode for faster inference. ~1s transcription time on CPU. |

**Key decision:** NeMo was the only model that consistently produced accurate Armenian transcriptions without requiring exaggerated enunciation. Since it's a local model, it required writing a custom LiveKit STT plugin (`nemo_stt.py`) to integrate it into the pipeline.

### Large Language Model (LLM)

| Model | Status | Reason |
|-------|--------|--------|
| OpenAI GPT-4.1-nano | Rejected | Too expensive for a prototype with budget constraints. |
| HyGPT-10b-it | Rejected | No hosted API available. Requires ~20GB GPU VRAM to run locally, not feasible. |
| mGPT Armenian 1.3B | Rejected | Decoder-only model, not instruction-tuned, no chat capability. Cannot follow system prompts or answer questions. |
| Gemini 1.5 Flash | Rejected | Deprecated Python package at time of testing, quota issues on free tier. |
| **Groq Llama 3.3 70B Versatile** | **Selected** | Fast inference (~1-2s), strong multilingual capability including Armenian, follows system prompts well. Developer Tier at $5/month limit — pay-as-you-go at ~$0.59/M input tokens + $0.79/M output tokens. Each question costs roughly half a cent. |

**Key decision:** Groq was selected for its combination of speed (fastest LLM inference available), cost-effectiveness, and surprisingly good Armenian language understanding despite not being Armenian-specific. The model reliably follows the system prompt's guardrails (topic restriction, number formatting).

### Text-to-Speech (TTS)

| Model | Status | Reason |
|-------|--------|--------|
| OpenAI TTS | Rejected | Too expensive for continuous voice agent use. |
| gTTS (Google Translate TTS) | Rejected | Robotic, unnatural voice quality, unacceptable for a customer support agent. |
| ArmTTS via RapidAPI | Rejected | Endpoint was unclear, turned out to not actually provide TTS functionality. |
| ElevenLabs TTS | Rejected | No Armenian voice/language support available. |
| **Google Cloud TTS (hy-AM)** | **Selected** | Via `livekit-plugins-google`. Provides a natural-sounding Armenian voice. Integrated directly with LiveKit's TTS interface. Uses Google Cloud service account credentials. |

**Key decision:** Google Cloud TTS was the only viable option that provided both Armenian language support and acceptable voice quality. It integrates cleanly with LiveKit via the official plugin.

---

## Data Pipeline

### Scraping

Data was scraped from three Armenian bank websites:

| Bank | Loans | Deposits | Branches | Scraping Method |
|------|-------|----------|----------|----------------|
| **Evoca** | 39 | 4 | 17 | `requests` + BeautifulSoup (static HTML) |
| **Ameria** | 24 | 7 | 30 | Selenium (dynamic JS content) |
| **ACBA** | 39 | 4 | 66 | `requests` + BeautifulSoup (static HTML with tabs) |
| **Total** | **102** | **15** | **113** | |

**230 items** scraped across 3 banks and 3 topics.

Each bank required a different scraping approach due to different website architectures:

- **Evoca:** Simple static HTML. Product pages use consistent CSS classes (`cards-info__currency-item`, `cards-info__desc`). Branch data includes region/community metadata in HTML data attributes.

- **Ameria:** Content loaded dynamically via JavaScript (WebSitesCreative CMS modules). Required Selenium to render pages, click tabs to reveal hidden content and cycle through a region dropdown for branches. Branch sidebar uses `.sidebar-item` cards filtered by visibility per region.

- **ACBA:** Content is in the HTML source (including tab content), but uses a different structure. Product pages have a tab UI (`tabs__tpl1__tabs__item` / `tabs__tpl1__bodys__item`) where tab titles pair with tab bodies in DOM order. Branch page lists all 66 branches in static HTML with CSS class `fb_branch`.

I manually explored each bank's page source (Ctrl+U) to understand the HTML structure, identify the relevant CSS classes, and determine whether Selenium was needed.

### Ingestion

The ingestion pipeline (`ingest.py`) processes all scraped JSON files into ChromaDB:

1. **Short items** (description < 1000 chars) — branches, simple product summaries are compacted to one line each and grouped into chunks of up to 1500 characters. This keeps related items together for better retrieval.

2. **Long items** (description ≥ 1000 chars) — detailed product pages with tab content, legal info pages are split individually at Armenian sentence boundaries. Each chunk gets a header with the product's key fields (bank name, product name) for context.

**Result:** 230 items → **468 chunks** stored in ChromaDB with metadata (bank, topic) for filtered retrieval.

**Embeddings:** `Metric-AI/armenian-text-embeddings-2-base` — a HuggingFace model trained specifically for Armenian text. Produces 768-dimensional vectors used for both ingestion and query-time semantic search.

---

## RAG System

The agent uses Retrieval-Augmented Generation to answer questions based only on scraped bank data:

1. **Topic detection:** Armenian keyword matching classifies each query:
   - "ավանդ" (deposit), "խնայողություն" (savings) → deposits
   - "վարկ" (loan), "հիփոթեք" (mortgage) → loans
   - "մասնաճյուղ" (branch), "հասցե" (address) → branches

2. **Retrieval:** ChromaDB semantic search using Armenian embeddings, filtered by detected topic. Returns top 3 most relevant chunks.

3. **Context truncation:** Retrieved chunks are combined up to 4000 characters to fit within LLM token limits.

4. **LLM generation:** Groq Llama 3.3 70B generates a response using a strict Armenian system prompt that:
   - Restricts answers to loans, deposits and branches only
   - Refuses off-topic questions with a polite Armenian message
   - Requires all numbers to be written as Armenian words (for TTS pronunciation)
   - Forbids use of external knowledge

---

## Known Limitations

### Number Pronunciation
The Google Cloud TTS reads digit sequences individually (e.g., "100000" is pronounced as "one-zero-zero-zero-zero-zero" in Armenian). To work around this, the system prompt instructs the LLM to write all numbers as Armenian words (e.g., "հարյուր հազար" instead of "100000"). This works in most cases but occasionally fails․ The LLM sometimes still outputs digits, especially for percentages and complex numbers, resulting in awkward pronunciation.

### STT Latency on CPU
The NeMo FastConformer model takes ~1 second per transcription on CPU. This is acceptable but not instant. With a GPU, inference would be ~0.2s.

### TTS Length Limit
Very long LLM responses (especially numbered lists) can cause the Google Cloud TTS to time out. The system prompt instructs the LLM to keep answers concise, but this occasionally fails.

### Ameria Scraping Fragility
Ameria's dynamic JavaScript-rendered content depends on specific CMS module selectors. Website redesigns could break the scraper.

---

## Scalability

The system was designed with scalability in mind:

- **`BaseScraper` abstract class** defines a standard interface (`scrape_loans()`, `scrape_deposits()`, `scrape_branches()`) that all bank scrapers implement. Adding a new bank requires creating a single new file in `banks/` that extends `BaseScraper`.

- **Ingestion is bank-agnostic** — `ingest.py` processes any JSON files in `Data/raw/<bank_name>/` regardless of which bank produced them. The chunking strategy adapts automatically based on content length.

- **RAG retrieval uses metadata filtering** — each chunk is tagged with its bank and topic, allowing the system to scale to many banks without degrading retrieval quality.

- **However**, each bank scraper is necessarily unique. Armenian bank websites have vastly different architectures (static HTML vs. dynamic JS, different CSS class naming, different tab/dropdown implementations). Each new bank requires manually inspecting the page source to understand its structure. This is an inherent limitation of web scraping, and makes the project not easily scalable.

---

## Project Structure

```
Armenian-Voice-AI-Support-Agent/
├── agent.py                 # Main voice agent w/ LiveKit pipeline (STT→LLM→TTS)
├── nemo_stt.py              # Custom LiveKit STT plugin for NVIDIA NeMo Armenian
├── ingest.py                # Data ingestion - chunks + embeds scraped data into ChromaDB
├── scraper.py               # Entry point for running all bank scrapers
├── generate_token.py        # LiveKit room token generator (for playground testing)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
├── banks/                   # Bank-specific scrapers
│   ├── __init__.py
│   ├── base_scraper.py      # Abstract base class - defines scraping interface
│   ├── evoca.py             # Evoca Bank scraper (requests + BeautifulSoup)
│   ├── ameria.py            # Ameria Bank scraper (Selenium for dynamic content)
│   └── acba.py              # ACBA Bank scraper (requests + BeautifulSoup)
├── Data/
│   ├── raw/                 # Scraped JSON data (committed)
│   │   ├── evoca/           # {loans,deposits,branches}.json
│   │   ├── ameria/          # {loans,deposits,branches}.json
│   │   └── acba/            # {loans,deposits,branches}.json
│   └── chroma_db/           # ChromaDB vector store (generated by ingest.py, not committed)
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Google Cloud account with Text-to-Speech API enabled
- Groq API key (Developer Tier, free trial with $5 limit)
- LiveKit server binary (for non-console mode)
- Chrome browser (for Ameria scraper's Selenium)

### 1. Clone and Install

```bash
git clone https://github.com/awinnnie/Armenian-Voice-AI-Support-Agent.git
cd Armenian-Voice-AI-Support-Agent
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_APPLICATION_CREDENTIALS=your_google_service_account.json
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

**Google Cloud setup:**
1. Create a project in Google Cloud Console
2. Enable the Cloud Text-to-Speech API
3. Create a service account and download the JSON key file
4. Place the JSON key file in the project root
5. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to the filename

### 3. Scrape Bank Data (Optional, pre-scraped data is included)

```bash
python scraper.py
```

This scrapes all three banks. Ameria requires Chrome/ChromeDriver (installed automatically via `webdriver-manager`). Takes ~10-15 minutes.

### 4. Ingest Data into ChromaDB

```bash
python ingest.py
```

This processes the scraped JSON files, generates Armenian text embeddings, and stores 468 chunks in ChromaDB. Takes ~5-10 minutes (embedding generation).

### 5. Run the Agent

```bash
python agent.py console
```

This starts the voice agent in console mode. It uses your microphone for input and speakers for output. The NeMo model loads on first use (~30 seconds).

---

## Usage

Once running in console mode:

- **Speak in Armenian** ask about loans, deposits or branches
---

## Costs

| Service | Cost | Notes |
|---------|------|-------|
| Groq LLM | ~$0.005/question | Developer Tier, $5/month limit, pay-as-you-go |
| Google Cloud TTS | Free tier | Well within free tier limits for testing |
| NVIDIA NeMo STT | Free | Open-source, runs locally |
| ChromaDB | Free | Open-source, runs locally |
| LiveKit | Free | Open-source server, runs locally |
| **Total for testing** | **< $1** | ~200 questions at $0.005 each |
