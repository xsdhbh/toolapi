from fastapi import FastAPI

from routers import image

app = FastAPI()

app.include_router(image.router)
