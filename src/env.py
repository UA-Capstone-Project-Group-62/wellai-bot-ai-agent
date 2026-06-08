import os

from dotenv import load_dotenv

load_dotenv()

env = {
    "PORT": os.getenv("PORT", "50051"),
    "BOT_SERVICE_ADDR": os.getenv("BOT_SERVICE_ADDR", "localhost:50052"),
    "SCHEDULING_SERVICE_ADDR": os.getenv("SCHEDULING_SERVICE_ADDR", "localhost:50051"),
}
