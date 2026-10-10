import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
import uvicorn.loops.asyncio

if sys.platform == "win32":
    uvicorn.loops.asyncio.asyncio_loop_factory = lambda use_subprocess=False: asyncio.SelectorEventLoop

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, loop="asyncio")
