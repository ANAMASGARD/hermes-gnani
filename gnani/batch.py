"""Single-file Prisma Batch jobs with bounded synchronous waiting."""

import json
from collections.abc import Iterator
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from uuid import UUID

from .client import Client, GnaniError

JOBS = "/stt/v3/batch/jobs"
DIRECT_LIMIT = 10_000_000
UPLOAD_LIMIT = 5 * 1024**3
MAX_SECONDS = 4 * 3600


def _job_path(job_id: Any) -> str:
    try:
        return f"{JOBS}/{UUID(job_id)}"
    except AttributeError, TypeError, ValueError:
        raise GnaniError("Gnani returned an invalid Batch job ID.") from None


def cancel(client: Client, job_id: str) -> None:
    client.json("POST", f"{_job_path(job_id)}/cancel", deadline=monotonic() + 20, json={})


def _retrieve(client: Client, job: str, deadline: float) -> dict[str, Any]:
    files = client.json("GET", f"{job}/files", deadline=deadline).get("data")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        raise GnaniError("Gnani returned an unexpected Batch file list.")
    if files[0].get("status") != "COMPLETED":
        raise GnaniError("Gnani did not transcribe the Batch audio file.")
    link = files[0].get("transcript_url")
    if not isinstance(link, str):
        raise GnaniError("Gnani did not provide a signed transcript link.")
    return client.json("GET", link, deadline=deadline, signed=True)


def wait(client: Client, job_id: str, deadline: float) -> dict[str, Any]:
    job = _job_path(job_id)
    while monotonic() < deadline:
        status = client.json("GET", job, deadline=deadline).get("status")
        if status in ("COMPLETED", "PARTIAL_FAILURE"):
            return _retrieve(client, job, deadline)
        if status in ("FAILED", "START_FAILED", "CANCELLED"):
            raise GnaniError(f"Gnani Batch ended with status {status}.")
        if status not in ("CREATED", "STARTING", "QUEUED", "IN_PROGRESS", "CANCELLING"):
            raise GnaniError("Gnani returned an unexpected Batch status.")
        if monotonic() + 10 >= deadline:
            break
        sleep(10)
    raise GnaniError("Gnani Batch is pending.", code="batch_pending")


def _upload_chunks(path: Path, deadline: float) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while chunk := handle.read(256 * 1024):
            if monotonic() >= deadline:
                raise GnaniError("Gnani Batch upload exceeded its time limit.")
            yield chunk


def transcribe(
    client: Client, path: Path, filename: str, size: int, config: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    deadline = monotonic() + 90
    if size > UPLOAD_LIMIT:
        raise GnaniError("Split the recording before sending it to Gnani Batch (5 GiB limit).")
    if size <= DIRECT_LIMIT:
        with path.open("rb") as handle:
            response = client.json(
                "POST",
                JOBS,
                deadline=deadline,
                files={
                    "files": (filename, handle, "application/octet-stream"),
                    "config": (None, json.dumps(config), "application/json"),
                },
            )
    else:
        response = client.json(
            "POST",
            "/stt/v3/batch/v2/jobs",
            deadline=deadline,
            json={
                "config": config,
                "source": {"type": "direct_upload_url", "files": [{"filename": filename}]},
            },
        )
    job = _job_path(response.get("job_id"))
    job_id = job.rsplit("/", 1)[1]
    try:
        if size > DIRECT_LIMIT:
            uploads = response.get("uploads")
            if not isinstance(uploads, list) or len(uploads) != 1 or not isinstance(uploads[0], dict):
                raise GnaniError("Gnani returned an unexpected Batch upload list.")
            link = uploads[0].get("url")
            if not isinstance(link, str):
                raise GnaniError("Gnani did not provide a signed upload link.")
            with client.response(
                "PUT",
                link,
                deadline=deadline,
                signed=True,
                upload=lambda: _upload_chunks(path, deadline),
                content_length=size,
            ):
                pass
        client.json("POST", f"{job}/start", deadline=deadline)
        return wait(client, job_id, deadline), job_id
    except (GnaniError, OSError) as exc:
        message = str(exc) if isinstance(exc, GnaniError) else "Could not read the Batch audio file."
        code = exc.code if isinstance(exc, GnaniError) else "batch_error"
        raise GnaniError(
            f"{message.rstrip('.')} (job {job_id}). Check status or cancel through Gnani Batch before retrying.",
            code=code,
        ) from None
