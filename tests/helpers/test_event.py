from ghoshell_moss.helpers.event import ThreadSafeEvent
import asyncio
from threading import Thread


def test_event_set_and_wait():
    event = ThreadSafeEvent()

    done = []

    def set_thread():
        event.set()
        done.append(1)

    def wait_sync_thread():
        event.wait_sync()
        done.append(1)

    async def wait_async():
        await event.wait()
        done.append(1)

    def wait_async_thread():
        asyncio.run(wait_async())

    t = [Thread(target=set_thread)]
    for i in range(5):
        t.append(Thread(target=wait_sync_thread))
        t.append(Thread(target=wait_async_thread))

    for i in t:
        i.start()
    for i in t:
        i.join()

    assert len(done) == 11
