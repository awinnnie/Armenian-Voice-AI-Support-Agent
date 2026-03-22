"""
Armenian Voice AI Banking Support Agent.

End-to-end voice pipeline using the LiveKit Agents SDK:
    STT:  NVIDIA NeMo FastConformer (Armenian) — via custom nemo_stt.py
    LLM:  Groq Llama 3.3 70B with RAG (ChromaDB + Armenian embeddings)
    TTS:  Google Cloud Text-to-Speech (hy-AM)
    VAD:  Silero Voice Activity Detection

The agent answers questions about loans, deposits and branch locations
for three Armenian banks - Evoca, Ameria, and ACBA. All responses are
in Armenian. Off-topic questions are politely refused.

RAG Pipeline:
    1. User speaks in Armenian
    2. NeMo STT transcribes speech to text
    3. detect_topic() classifies query as deposits/loans/branches
    4. retrieve() fetches relevant chunks from ChromaDB using
       Metric-AI Armenian embeddings for semantic search
    5. get_answer() sends context + query to Groq LLM
    6. Google Cloud TTS synthesizes the Armenian response to speech

Usage:
    python agent.py console    # Local testing with microphone + speaker
    python agent.py start      # Connect to LiveKit server for WebRTC
"""

import os
import asyncio
import chromadb
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
from groq import Groq
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent
from livekit.agents import llm as agents_llm
from livekit.plugins import google as google_plugin
from livekit.plugins import groq as groq_plugin
from livekit.plugins import silero

load_dotenv()

# Armenian text embeddings for RAG retrieval
# Uses Metric-AI's model trained specifically for Armenian text

MODEL_NAME = "Metric-AI/armenian-text-embeddings-2-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embed_model = AutoModel.from_pretrained(MODEL_NAME)

# ChromaDB - persistent vector store
# Pre-populated by ingest.py with 468 chunks from 3 banks

chroma_client = chromadb.PersistentClient(path="Data/chroma_db")
collection = chroma_client.get_collection("bank_data")

# Groq LLM client - Llama 3.3 70B
# Developer Tier: $5 limit, ~$0.59/M input + $0.79/M output tokens
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# RAG configuration
MAX_CONTEXT_CHARS = 4000
N_RESULTS = 3


def embed(text):
    """Generates embedding vector for a text query using Armenian embeddings model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = embed_model(**inputs)
    return output.last_hidden_state[:, 0, :].squeeze().tolist()


def detect_topic(query):
    """
    Classifies a user query into one of three topics based on Armenian keywords.
    """
    
    if any(kw in query for kw in ["ավանդ", "խնայողություն", "դեպոզիտ"]):
        return "deposits"
    if any(kw in query for kw in ["վարկ", "հիփոթեք", "ավտո", "վարկային"]):
        return "loans"
    if any(kw in query for kw in ["մասնաճյուղ", "հասցե", "բաժանմունք", "գրասենյակ"]):
        return "branches"
    return None


def retrieve(query, n=N_RESULTS):
    """
    Retrieves the most relevant text chunks from ChromaDB.

    Uses Armenian text embeddings for semantic similarity search.
    If a topic is detected, filters results to that topic only
    for more precise retrieval.

    query: User's question in Armenian
    n: Number of chunks to retrieve

    Returns list of text strings (the matching document chunks)
    """
    topic = detect_topic(query)
    query_embedding = embed(query)
    where = {"topic": topic} if topic else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where,
    )
    return results["documents"][0]


def truncate_context(chunks, max_chars=MAX_CONTEXT_CHARS):
    """
    Combines retrieved chunks into a context string, capped at max_chars.

    Joins chunks with double newlines. If adding the next chunk would
    exceed the limit, truncates it and adds "..." suffix. This prevents
    sending too much context to the LLM (which would waste tokens and
    potentially confuse the response).
    """
    context = ""
    for chunk in chunks:
        if len(context) + len(chunk) + 2 > max_chars:
            remaining = max_chars - len(context) - 5
            if remaining > 200:
                context += "\n\n" + chunk[:remaining] + "..."
            break
        context += ("\n\n" + chunk) if context else chunk
    return context

# System prompt instructs the LLM to
# 1. Only answer about loans, deposits and branches
# 2. Refuse off-topic questions politely in Armenian
# 3. Write ALL numbers as Armenian words (for TTS pronunciation)
# 4. Never use external knowledge, only the provided context

SYSTEM_PROMPT = """Դու հայկական բանկերի աջակցության գործակալ ես։
Պատասխանիր ՄԻԱՅՆ հետևյալ տեղեկատվության հիման վրա։
Պատասխանիր կարճ և հստակ՝ ձայնային պատասխանին հարմար։
 
ԿԱՐևՈՐ ԿԱՆՈՆՆԵՐ՝
- Դու պատասխանում ես ՄԻԱՅՆ վարկերի, ավանդների և մասնաճյուղերի մասին։
- Եթե հարցը դուրս է այս թեմաներից (օրինակ՝ եղանակ, սպորտ, քաղաքականություն, բաղադրատոմս, բանկերի մասին այլ ինֆորմացիա), պատասխանիր՝ "Ներեցեք, ես կարող եմ օգնել միայն վարկերի, ավանդների և մասնաճյուղերի վերաբերյալ։"
- ՄԻ օգտագործիր արտաքին գիտելիքները։
 
Թվերի մասին՝
- ԲՈԼՈՐ թվերը գրիր բառերով հայերեն։
- 18 գրիր տասնութ, 95% գրիր իննսուն հինգ տոկոս, 50% գրիր հիսուն տոկոս։
- Օրինակ՝ 100000 դրամ գրիր "հարյուր հազար դրամ", 9.5% գրիր "ինն ու կես տոկոս"։"""


def get_answer(query):
    """
    Generates an answer using the RAG pipeline.
    1. Retrieves relevant chunks from ChromaDB
    2. Truncates context to fit within LLM token limits
    3. Builds prompt with system instructions + context + question
    4. Sends to Groq Llama 3.3 70B and return the response
    """
    chunks = retrieve(query)
    context = truncate_context(chunks)
    prompt = f"""{SYSTEM_PROMPT}

Տեղեկատվություն:
{context}

Հարց: {query}
Պատասխան:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content

# LiveKit LLM integration
class RAGLLMStream(agents_llm.LLMStream):
    """Custom LLM stream that wraps the RAG pipeline."""

    def __init__(self, llm, answer, *, chat_ctx, tools, conn_options):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._answer = answer

    async def _run(self):
        """Send the complete RAG answer as one ChatChunk."""
        self._event_ch.send_nowait(
            agents_llm.ChatChunk(
                id="rag-response",
                delta=agents_llm.ChoiceDelta(
                    role="assistant",
                    content=self._answer,
                ),
            )
        )


class RAGLLMPlugin(agents_llm.LLM):
    """
    Custom LLM plugin that routes LiveKit's LLM calls through our RAG pipeline.

    LiveKit's AgentSession calls llm.chat() when it needs a response.
    Instead of calling an LLM API directly, we
    1. Extract the latest user message from the chat context
    2. Run it through our RAG pipeline (retrieve + Groq LLM)
    3. Return the answer wrapped in a RAGLLMStream
    """
    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        user_msg = ""
        for msg in reversed(list(chat_ctx.items)):
            if not hasattr(msg, "role"):
                continue
            if msg.role == "user":
                for content in msg.content:
                    if hasattr(content, "text"):
                        user_msg = content.text
                        break
                    else:
                        user_msg = str(content)
                        break
                break

        # If no user message (e.g. initial greeting), return the instructions directly
        if not user_msg.strip():
            # Find the system/instructions text to use as greeting
            answer = ""
            for msg in chat_ctx.items:
                if hasattr(msg, "role") and msg.role == "assistant":
                    for content in msg.content:
                        if hasattr(content, "text"):
                            answer = content.text
                            break
                    break
            if not answer:
                answer = "Բարև ձեզ, ես բանկային աջակցության գործակալն եմ։ Ինչպե՞ս կարող եմ օգնել։"
        else:
            answer = get_answer(user_msg)

        if conn_options is None:
            conn_options = agents_llm.llm.APIConnectOptions()

        return RAGLLMStream(
            self,
            answer,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class BankAssistant(Agent):
    """LiveKit Agent with Armenian banking assistant instructions."""
    def __init__(self):
        super().__init__(instructions="Դու հայկական բանկային աջակցության գործակալ ես։\nՊատասխանիր միայն վարկերի, ավանդների և մասնաճյուղերի մասին հարցերին։\nՊատասխանիր հայերեն։")

# LiveKit server setup and session handler
server = AgentServer()


@server.rtc_session(agent_name="armenian-bank-agent")
async def my_agent(ctx: agents.JobContext):
    """
    Main agent session handler, creates the full voice pipeline

    Pipeline:
        Microphone -> VAD (Silero) -> STT (NeMo Armenian) ->
        LLM (RAG + Groq) -> TTS (Google Cloud hy-AM) -> Speaker

    The NeMo STT model is imported inside this function to avoid
    loading the heavy model (~1GB) at module import time.
    """
    from nemo_stt import NeMoArmenianSTT
    session = AgentSession(
        stt=NeMoArmenianSTT(),
        llm=RAGLLMPlugin(),
        tts=google_plugin.TTS(
            language="hy-AM",
            credentials_file=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        ),
        vad=silero.VAD.load(), # voice activity detection
    )

    await session.start(
        room=ctx.room,
        agent=BankAssistant(),
    )

    # Send initial greeting when user connects
    await session.generate_reply(
        instructions="Բարև ձեզ, ես բանկային աջակցության գործակալ եմ։ Ինչպե՞ս կարող եմ օգնել։"
    )


if __name__ == "__main__":
    agents.cli.run_app(server)