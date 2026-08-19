import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from mcp_agent.agent import stream_chat  # noqa: E402 — must load env before importing agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Flowio AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str
    images: list[str] | None = None  # base64-encoded images


class ChatRequest(BaseModel):
    messages: list[Message]


@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [
        {"role": m.role, "content": m.content, **({"images": m.images} if m.images else {})}
        for m in request.messages
    ]

    async def generate() -> AsyncGenerator[str, None]:
        async for chunk in stream_chat(messages):
            safe = chunk.encode("utf-8", errors="replace").decode("utf-8")
            yield f"data: {json.dumps({'text': safe})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
