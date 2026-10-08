import threading
import concurrent.futures
from typing import Iterable, Callable, Iterator, Any


def map_concurrent(
    iterable: Iterable,
    fn: Callable[[Any], Any],
    *,
    max_workers: int = 4,
    max_prefetch: int = 16,
    ordered: bool = True,
    use_process: bool = False,
    discard_null: bool = False,
) -> Iterator:
    """
    Apply fn to items from iterable concurrently, with bounded prefetch and optional ordering.

    - max_workers=1 will ensure at-most-one fn running at a time (good for expensive resources).
    - max_prefetch bounds number of in-flight tasks (backpressure).
    - ordered=True preserves input order; ordered=False yields results as they complete.

    Yields: fn(item) results (or raises exceptions from fn).
    """
    executor_cls = (
        concurrent.futures.ProcessPoolExecutor
        if use_process
        else concurrent.futures.ThreadPoolExecutor
    )

    semaphore = threading.Semaphore(max_prefetch)
    with executor_cls(max_workers=max_workers) as executor:
        futures_by_index = {}  # for ordered mode: index -> future
        in_flight = []  # for unordered mode: futures list
        submitted = 0
        finished_index = 0

        # submit loop
        for idx, item in enumerate(iterable):
            semaphore.acquire()  # block if too many in-flight
            fut = executor.submit(fn, item)

            # when future completes, release one slot in prefetch
            fut.add_done_callback(lambda _f: semaphore.release())

            if ordered:
                futures_by_index[idx] = fut
                # yield while next index ready
                while (
                    finished_index in futures_by_index
                    and futures_by_index[finished_index].done()
                ):
                    res = futures_by_index.pop(finished_index).result()
                    if not discard_null or res is not None:
                        yield res
                    finished_index += 1
            else:
                in_flight.append(fut)
                # yield any completed futures right away (non-blocking)
                done_now = [f for f in in_flight if f.done()]
                for f in done_now:
                    in_flight.remove(f)
                    if not discard_null or f.result() is not None:
                        yield f.result()

            submitted += 1

        # all items submitted -> drain remaining futures
        if ordered:
            # wait for remaining indices in order
            for idx in range(finished_index, submitted):
                res = futures_by_index.pop(idx).result()
                if not discard_null or res is not None:
                    yield res
        else:
            for f in concurrent.futures.as_completed(in_flight):
                res = f.result()
                if not discard_null or res is not None:
                    yield res
