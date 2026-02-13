import asyncio
import threading
import time

from ghoshell_moss.core.helpers.stream import (
    create_thread_safe_stream,
)


def test_thread_send_async_receive():
    content = "hello world"
    done = []
    sender, receiver = create_thread_safe_stream()

    def sending():
        with sender:
            for char in content:
                sender.append(char)

    async def receiving():
        try:
            buffer = ""
            async with receiver:
                async for char in receiver:
                    buffer += char
            done.append(buffer)
        except Exception as e:
            done.append(str(e))

    def sync_receiving():
        asyncio.run(receiving())

    t1 = threading.Thread(target=sending)
    t2 = threading.Thread(target=sync_receiving)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert content == done[0]


def test_thread_send_and_receive():
    content = "hello world"
    done = []
    sender, receiver = create_thread_safe_stream()

    def sending():
        with sender:
            for char in content:
                sender.append(char)

    def sync_receiving():
        buffer = ""
        with receiver:
            for char in receiver:
                buffer += char
        done.append(buffer)

    t1 = threading.Thread(target=sending)
    t2 = threading.Thread(target=sync_receiving)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert content == done[0]


def test_receiver_waits_after_queue_empty_until_new_item_sync():
    sender, receiver = create_thread_safe_stream(timeout=1.0)
    consumed: list[str] = []

    def producer():
        with sender:
            sender.append("A")  # queue has one item; not completed yet
            time.sleep(0.1)  # ensure consumer attempts the next() on empty queue
            sender.append("B")
            sender.commit()

    def consumer():
        with receiver:
            a = next(receiver)
            consumed.append(a)
            b = next(receiver)
            consumed.append(b)

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert consumed == ["A", "B"]
