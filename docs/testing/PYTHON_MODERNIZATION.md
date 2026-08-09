# Python Modernization

This document records the dependency and runtime modernization completed on 2026-08-09.

## Goals

- Keep Python 3.11 compatibility while supporting Python 3.14.
- Replace the stale, non-reproducible dependency ranges with tested runtime and development locks.
- Upgrade the FastAPI, Pydantic, Celery, Redis, and ML runtime stack together.
- Avoid changing model or tracking behavior while modernizing infrastructure.

## Supported Runtimes

- Python 3.11 with NumPy 2.4.6
- Python 3.14 with NumPy 2.5.2
- Docker services use `python:3.14-slim`.
- CI tests both Python versions.

NumPy uses an interpreter marker because NumPy 2.5 requires Python 3.12 or newer.

## Direct Dependencies

| Package | Previous | Modernized |
| --- | ---: | ---: |
| FastAPI | 0.95.2 | 0.141.x |
| Pydantic | 1.10.8 | 2.13.x |
| Uvicorn | 0.22.0 | 0.52.x |
| python-multipart | 0.0.6 | 0.0.32.x |
| Celery | 5.2.7 | 5.6.x |
| Redis client | 4.5.4 | 8.1.x |
| PyTorch | 2.6-2.8 declared | 2.13.x |
| torchvision | 0.21.x declared | 0.28.x |
| Ultralytics | 8.3+ | 8.4.x |
| OpenCV | 4.10+ | 5.0.x |
| ONNX Runtime | 1.18+ | 1.28.x |

Norfair remains pinned at 2.1.1 so the modernization cannot alter tracking output. Unused direct dependencies for authentication and data processing were removed; packages still required transitively remain in the lock.

## Reproducibility

- `requirements.txt` contains supported direct runtime ranges.
- `requirements-dev.txt` adds testing and lint dependencies.
- `requirements.lock` pins the complete runtime graph.
- `requirements-dev.lock` pins the complete development graph.
- macOS installs standard PyTorch wheels with MPS support.
- Linux installs CPU-only PyTorch wheels instead of the multi-gigabyte CUDA dependency graph.

Regenerate the locks with the commands recorded in their headers. Install application dependencies with:

```bash
pip install -r requirements.lock
```

For development:

```bash
pip install -r requirements-dev.lock
```

## Source Compatibility

The existing Pydantic models use only `BaseModel`, typed fields, and attribute access, which are already valid in Pydantic 2. The only runtime source migration was replacing deprecated event-loop lookup with `asyncio.get_running_loop()` inside async functions.

The upgrade also exposed and fixed:

- A deliberate analysis `HTTPException(404)` being caught and returned as 500.
- NumPy 2 tests that mocked scalar tensors with one-dimensional arrays.
- TestClient cleanup by using its context manager fixture.

## Validation

- Backend imports successfully on Python 3.11 and 3.14.
- Fatal Flake8 checks pass on both Python versions.
- Python 3.14 backend and AI-core Docker images build successfully.
- Both Docker images import their application modules successfully.
- The locked Python 3.11 stack produced identical 200-frame analysis output to the pre-modernization baseline after removing only `analysis_time`.

This branch does not claim a processing-speed improvement. Video throughput and output parity are benchmarked separately from dependency modernization so performance changes remain attributable.
