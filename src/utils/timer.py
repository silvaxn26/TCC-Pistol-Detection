from contextlib import contextmanager
from time import perf_counter


@contextmanager
def elapsed_timer():
    start = perf_counter()
    result = {"seconds": 0.0}
    try:
        yield result
    finally:
        result["seconds"] = perf_counter() - start


def mean_seconds(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

