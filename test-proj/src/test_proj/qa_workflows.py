from __future__ import annotations
import logging
import os
import tempfile
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from llama_cloud import ChatMessage
from llama_cloud.types import RetrievalMode
from llama_index.core import Settings
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel, Field
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_cloud_services import LlamaCloudIndex
from workflows import Workflow, step, Context
from workflows.events import (
    StartEvent,
    StopEvent,
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows.retry_policy import ConstantDelayRetryPolicy

from .clients import (
    LLAMA_CLOUD_API_KEY,
    LLAMA_CLOUD_BASE_URL,
    get_custom_client,
    get_llama_cloud_client,
    get_llama_parse_client,
    LLAMA_CLOUD_PROJECT_ID,
)

load_dotenv()


logger = logging.getLogger(__name__)


class FileEvent(StartEvent):
    file_id: str
    index_name: str


class DownloadFileEvent(Event):
    file_id: str


class FileDownloadedEvent(Event):
    file_id: str
    file_path: str
    filename: str


class ChatEvent(StartEvent):
    index_name: str
    session_id: str


# Configure LLM and embedding model
Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

custom_client = get_custom_client()


class DocumentUploadWorkflow(Workflow):
    """Workflow to upload and index documents using LlamaParse and LlamaCloud Index"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get API key with validation

        # Initialize LlamaParse with recommended settings
        self.parser = get_llama_parse_client()

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=3, delay=10))
    async def run_file(self, event: FileEvent, ctx: Context) -> DownloadFileEvent:
        logger.info(f"Running file {event.file_id}")
        await ctx.store.set("index_name", event.index_name)
        return DownloadFileEvent(file_id=event.file_id)

    @step(retry_policy=ConstantDelayRetryPolicy(maximum_attempts=3, delay=10))
    async def download_file(
        self, event: DownloadFileEvent, ctx: Context
    ) -> FileDownloadedEvent:
        """Download the file reference from the cloud storage"""
        logger.info(f"Downloading file {event.file_id}")
        try:
            file_metadata = await get_llama_cloud_client().files.get_file(
                id=event.file_id
            )
            file_url = await get_llama_cloud_client().files.read_file_content(
                event.file_id
            )

            temp_dir = tempfile.gettempdir()
            filename = file_metadata.name
            file_path = os.path.join(temp_dir, filename)
            client = httpx.AsyncClient()
            # Report progress to the UI
            logger.info(f"Downloading file {file_url.url} to {file_path}")

            async with client.stream("GET", file_url.url) as response:
                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
            logger.info(f"Downloaded file {file_url.url} to {file_path}")
            return FileDownloadedEvent(
                file_id=event.file_id, file_path=file_path, filename=filename
            )
        except Exception as e:
            logger.error(f"Error downloading file {event.file_id}: {e}", exc_info=True)
            raise e

    @step
    async def parse_document(self, ev: FileDownloadedEvent, ctx: Context) -> StopEvent:
        """Parse document and index it to LlamaCloud"""
        try:
            logger.info(f"Parsing document {ev.file_id}")
            # Get file path or content from event
            file_path = ev.file_path
            file_name = file_path.split("/")[-1]
            index_name = await ctx.store.get("index_name")

            # Parse the document
            if file_path:
                # Parse from file path
                result = await self.parser.aparse(file_path)

            # Get parsed documents
            documents = result.get_text_documents()

            # Create or connect to LlamaCloud Index
            index = LlamaCloudIndex.create_index(
                documents=documents,
                name=index_name,
                project_id=LLAMA_CLOUD_PROJECT_ID,
                api_key=LLAMA_CLOUD_API_KEY,
                base_url=LLAMA_CLOUD_BASE_URL,
                show_progress=True,
                custom_client=custom_client,
            )

            # Insert documents to index
            logger.info(f"Inserting {len(documents)} documents to {index_name}")
            for document in documents:
                index.insert(document)

            return StopEvent(
                result={
                    "success": True,
                    "index_name": index_name,
                    "document_count": len(documents),
                    "index_url": f"https://cloud.llamaindex.ai/projects/{LLAMA_CLOUD_PROJECT_ID}/indexes/{index.id}",
                    "file_name": file_name,
                    "message": f"Successfully indexed {len(documents)} documents to '{index_name}'",
                }
            )

        except Exception as e:
            logger.error(e.stack_trace)
            return StopEvent(
                result={"success": False, "error": str(e), "stack_trace": e.stack_trace}
            )


class AppendChatMessage(Event):
    """Event emitted when chat engine appends a message to the conversation history"""

    message: ConversationMessage


class ChatDeltaEvent(Event):
    """Streaming delta for incremental response output"""

    delta: str


class ChatWorkflowState(BaseModel):
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = None
    index_name: str | None = None


class SourceMessage(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    sources: list[SourceMessage] = Field(default_factory=list)


class ChatWorkflow(Workflow[ChatWorkflowState]):
    """Workflow to handle continuous chat queries against indexed documents"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_engines: dict[
            str, BaseChatEngine
        ] = {}  # Cache chat engines per index

    @step
    async def initialize_chat(
        self, ev: ChatEvent, ctx: Context[ChatWorkflowState]
    ) -> InputRequiredEvent:
        """Initialize the chat session and request first input"""
        try:
            logger.info(f"Initializing chat {ev.index_name}")
            index_name = ev.index_name
            session_id = ev.session_id

            # Store session info in context
            await ctx.store.set("index_name", index_name)
            await ctx.store.set("session_id", session_id)
            if ctx.store.get("conversation_history", None) is None:
                await ctx.store.set("conversation_history", [])

            # Create cache key for chat engine
            cache_key = f"{index_name}_{session_id}"

            # Initialize chat engine if not exists
            if cache_key not in self.chat_engines:
                logger.info(f"Initializing chat engine {cache_key}")
                # Connect to LlamaCloud Index
                index = LlamaCloudIndex(
                    name=index_name,
                    project_id=LLAMA_CLOUD_PROJECT_ID,
                    api_key=LLAMA_CLOUD_API_KEY,
                    base_url=LLAMA_CLOUD_BASE_URL,
                    async_httpx_client=custom_client,
                )

                # Create chat engine with memory
                memory = ChatMemoryBuffer.from_defaults(
                    token_limit=3900,
                    chat_history=await ctx.store.get(
                        "conversation_history", default=[]
                    ),
                )
                self.chat_engines[cache_key] = index.as_chat_engine(
                    chat_mode=ChatMode.CONTEXT,
                    memory=memory,
                    llm=Settings.llm,
                    context_prompt=(
                        "You are a helpful assistant that answers questions based on the provided documents. "
                        "Always cite specific information from the documents when answering. "
                        "If you cannot find the answer in the documents, say so clearly."
                    ),
                    verbose=False,
                    retriever_mode=RetrievalMode.CHUNKS,
                )

            history = await ctx.store.get("conversation_history", default=[])
            if len(history) == 0:
                ctx.write_event_to_stream(
                    ConversationMessage(
                        role="assistant",
                        text="Chat initialized. Ask a question (or type 'exit' to quit): ",
                    )
                )
            else:
                for item in history:
                    item: ConversationMessage = item
                    ctx.write_event_to_stream(item)
            # Request first user input
            return InputRequiredEvent(prefix="[waiting for user message]")

        except Exception as e:
            return StopEvent(
                result={
                    "success": False,
                    "error": f"Failed to initialize chat: {str(e)}",
                }
            )

    @step
    async def process_user_response(
        self, ev: HumanResponseEvent, ctx: Context
    ) -> InputRequiredEvent | HumanResponseEvent | StopEvent | None:
        """Process user input and generate response"""
        try:
            logger.info(f"Processing user response {ev.response}")
            user_input = ev.response.strip()
            with ctx.store.edit_state() as state:
                messages = state.get("conversation_history", default=[])
                messages.append(ConversationMessage(role="user", text=user_input))
                state.set("conversation_history", messages)

            logger.info(f"User input: {user_input}")

            # Check for exit command
            if user_input.lower() == "exit":
                logger.info("User input is exit")
                conversation_history = await ctx.store.get(
                    "conversation_history", default=[]
                )
                return StopEvent(
                    result={
                        "success": True,
                        "message": "Chat session ended.",
                        "conversation_history": conversation_history,
                    }
                )

            # Get session info from context
            index_name = await ctx.store.get("index_name")
            session_id = await ctx.store.get("session_id")
            cache_key = f"{index_name}_{session_id}"

            # Get chat engine
            chat_engine = self.chat_engines[cache_key]

            # Process query with chat engine (streaming)
            stream_response = await chat_engine.astream_chat(user_input)
            full_text = ""

            # Emit streaming deltas to the event stream
            async for token in stream_response.async_response_gen():
                full_text += token
                ctx.write_event_to_stream(ChatDeltaEvent(delta=token))

            # Extract source nodes for citations
            sources = []
            if stream_response.source_nodes:
                for node in stream_response.source_nodes:
                    sources.append(
                        SourceMessage(
                            text=node.text[:197] + "..."
                            if len(node.text) >= 200
                            else node.text,
                            score=node.score,
                            metadata=node.metadata,
                        )
                    )

            # Update conversation history
            response = ConversationMessage(
                role="assistant", text=full_text.strip(), sources=sources
            )
            with ctx.store.edit_state() as state:
                messages = state.get("conversation_history", default=[])
                messages.append(response)
                state.set("conversation_history", messages)

            # After streaming completes, emit a summary response event to stream for frontend/main printing
            ctx.write_event_to_stream(AppendChatMessage(message=response))
            return InputRequiredEvent(prefix="[waiting for user message]")

        except Exception as e:
            return StopEvent(
                result={"success": False, "error": f"Error processing query: {str(e)}"}
            )


upload = DocumentUploadWorkflow(timeout=None)
chat = ChatWorkflow(timeout=None)
