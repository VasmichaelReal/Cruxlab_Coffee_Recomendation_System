import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import server.state as state
from server.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle: initialization
    await state.init_data()
    yield

app = FastAPI(title="Coffee AI Coach API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular routes
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)