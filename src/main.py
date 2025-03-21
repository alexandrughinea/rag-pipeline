import os
import tempfile
import time

from api.request_types import QueryBaseRequest, QueryConversationRequest
from dotenv import load_dotenv
from embeddings_generator import EmbeddingsGenerator
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llm.llm_handler import LLMHandler
from processors.file_processor import FileProcessor
from query.query_engine import QueryEngine
from query.query_history import QueryHistory
from starlette.responses import JSONResponse
from utils import clamp
from vector_store.vector_store import VectorStore

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable must be set")

# API key header
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=401,
            detail="x-api-key header is missing"
        )
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key_header

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm_handler = LLMHandler()
doc_processor = FileProcessor()
embeddings_generator = EmbeddingsGenerator()

query_history = QueryHistory()

vector_store = VectorStore(embeddings_generator)
query_engine = QueryEngine(embeddings_generator, vector_store)


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """Upload and process a document."""
    print(f"Original filename: {file.filename}")

    suffix = os.path.splitext(file.filename)[1].lower()
    print(f"Detected extension: {suffix}")

    if not suffix.startswith('.'):
        suffix = f'.{suffix}'

    print(f"Using suffix: {suffix}")


    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)

    try:
        content = await file.read()

        with os.fdopen(temp_fd, 'wb') as temp_file:
            temp_file.write(content)

        print(f"Temp path: {temp_path}")

        # Process file
        text = doc_processor.process_file(temp_path)

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text extracted from file"}
            )

        embeddings, chunks = embeddings_generator.generate_embeddings(text)

        print(f"Embeddings: {embeddings}")
        print(f"Chunks: {chunks}")

        source_info = {
            "filename": file.filename,
            "file_type": suffix[1:],
        }

        vector_store.add_embeddings(embeddings, chunks, source_info)

        return JSONResponse(
            status_code=200,
            content={"message": f"Original file ({file.filename}) was processed successfully."}
        )

    except Exception as e:
        print(f"Error while processing file: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Something went wrong while uploading file ({file.filename})!"}
        )

    finally:
        os.unlink(temp_path)

@app.get("/search")
async def search(query: str, limit: int = 5):
    """Query the RAG system."""
    start_time = time.time()
    num_results = clamp(limit)

    results = query_engine.query(query, num_results)
    query_response_time = time.time() - start_time

    query_history.add_query(query, limit, query_response_time)

    return JSONResponse(
        status_code=200,
        content={
            "data": results,
            "query": query,
        }
    )


@app.post("/query/stream")
async def query_stream(
    request: QueryBaseRequest,
    api_key: str = Depends(get_api_key)
):
    """Stream query results."""
    start_time = time.time()
    media_type = "text/event-stream"

    try:
        embeddings, _ = embeddings_generator.generate_embeddings(request.query)
        results = vector_store.query(embeddings[0], request.n_results)

        if not results["documents"]:
            return StreamingResponse(
                iter(["No relevant documents found for your query."]),
                media_type=media_type,
            )

        async def generate():
            async for token in llm_handler.generate_stream(
                    query=request.query,
                    context=results["documents"]
            ):
                yield f"data: {token}\n\n"

            duration = time.time() - start_time

            query_history.add_query(
                query=request.query,
                num_results=len(results["documents"]),
                query_response_time=duration
            )

        return StreamingResponse(
            generate(),
            media_type=media_type
        )

    except Exception as e:
        return StreamingResponse(
            iter([f"Error: {str(e)}"]),
            media_type=media_type
        )

@app.post("/query/conversation-stream")
async def query_conversation_stream(
    request: QueryConversationRequest,
    api_key: str = Depends(get_api_key)
):
    """Stream query results as a conversation stream."""
    start_time = time.time()
    media_type = "text/event-stream"

    try:
        embeddings, _ = embeddings_generator.generate_embeddings(request.query)

        results = vector_store.query(embeddings[0], request.n_results)

        if not results["documents"]:
            return StreamingResponse(
                iter(["No relevant documents found for your query."]),
                media_type=media_type
            )

        async def generate():
            prepend_conversation_id = True
            async for token in llm_handler.generate_conversation_stream(
                    query=request.query,
                    context=results["documents"],
                    conversation_id=request.conversation_id
            ):
                if prepend_conversation_id:
                    yield f"id: {request.conversation_id}\n\n"
                    prepend_conversation_id = False
                yield f"data: {token}\n\n"

            duration = time.time() - start_time

            query_history.add_query(
                query=request.query,
                num_results=len(results["documents"]),
                query_response_time=duration
            )

        return StreamingResponse(
            generate(),
            media_type=media_type
        )

    except Exception as e:
        return StreamingResponse(
            iter([f"Error: {str(e)}"]),
            media_type=media_type
        )

@app.get("/history")
async def get_history(
    limit: int = 10,
    api_key: str = Depends(get_api_key)
):
    """Get recent query history."""
    num_results = clamp(limit)
    results = query_history.get_recent_queries(limit=num_results)

    return JSONResponse(
        status_code=200,
        content={ "data": results }
    )

@app.get("/")
async def get_index_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

if __name__ == "__main__":

    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(app, host=host, port=port)

    app.mount("/static", StaticFiles(directory="static"), name="static")
