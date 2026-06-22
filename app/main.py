from fastapi import FastAPI

from app.api import endpoints

app = FastAPI(title="mavan API")
app.include_router(endpoints.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
