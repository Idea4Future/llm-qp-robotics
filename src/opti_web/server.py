"""Loopback-only HTTP/SSE server for the real Isaac command worker.

The worker must append JSONL stage/spec/result events. The server never makes a
task specification or assumes that an exit code alone means mission success.
An accepted result is published only after both a worker result and exit 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

STATIC = Path(__file__).resolve().parent / "static"
TERMINAL = {"completed", "failed", "cancelled"}
# The backend owns independent LLM/physics sessions and needs time to clean them.
CANCEL_GRACE_S = 20


class JobError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class Job:
    id: str
    command: str
    directory: Path
    show_sim: bool = False
    moving_person: bool = False
    layout: str = "rack-v1"
    people_count: int = 3
    state: str = "starting"
    process: subprocess.Popen | None = None
    events: list = field(default_factory=list)
    spec: dict | None = None
    result: dict | None = None
    cancel_requested: bool = False
    cancel_started: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    condition: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, event):
        with self.condition:
            event = dict(event, seq=len(self.events) + 1,
                         received_at=datetime.now(timezone.utc).isoformat())
            self.events.append(event)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return {"id": self.id, "command": self.command, "state": self.state, "show_sim": self.show_sim,
                    "moving_person": self.moving_person, "layout": self.layout,
                    "people_count": self.people_count,
                    "effective_people_count": self.people_count if self.moving_person else 0,
                    "created_at": self.created_at, "last_seq": len(self.events),
                    "spec": self.spec, "result": self.result,
                    "output_directory": str(self.directory)}


class JobManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.worker = self.root / "scripts/isaac/run_command.sh"
        self.jobs = {}
        self.last_id = None
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            job = self.jobs.get(self.last_id)
            return {"backend_ready": self.worker.is_file(), "single_job": True,
                    "job": job.snapshot() if job else None}

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise JobError("작업을 찾을 수 없습니다.", 404)
        return job

    def start(self, command, show_sim=False, moving_person=False, layout="rack-v1", people_count=3):
        if not isinstance(command, str) or not command.strip() or len(command) > 3000:
            raise JobError("1~3000자의 작업 지시를 입력해 주세요.")
        if not isinstance(show_sim, bool):
            raise JobError("Isaac Sim 창 보기 옵션은 true 또는 false여야 합니다.")
        if not isinstance(moving_person, bool):
            raise JobError("이동하는 사람 옵션은 true 또는 false여야 합니다.")
        if not isinstance(layout, str) or layout not in {"rack-v1", "legacy"}:
            raise JobError("지원되는 환경은 rack-v1 또는 legacy입니다.")
        if type(people_count) is not int or people_count not in {1, 3}:
            raise JobError("보행자 수는 정수 1 또는 3이어야 합니다.")
        with self.lock:
            previous = self.jobs.get(self.last_id)
            if previous and previous.state not in TERMINAL:
                raise JobError("현재 작업이 끝나거나 중지된 뒤 새 지시를 보낼 수 있습니다.", 409)
            if not self.worker.is_file():
                raise JobError("실행 백엔드가 아직 준비되지 않았습니다. 작업은 실행하지 않았습니다.", 503)
            job_id = uuid.uuid4().hex
            directory = self.root / "runs/chat" / job_id
            directory.mkdir(parents=True, exist_ok=False)
            job = Job(job_id, command.strip(), directory, show_sim=show_sim,
                      moving_person=moving_person, layout=layout, people_count=people_count)
            self.jobs[job_id] = job
            self.last_id = job_id
            job.publish({"type": "stage", "stage": "starting", "detail": "작업 실행 프로세스를 시작합니다."})
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
            return job.snapshot()

    def _read_event(self, job, raw):
        try:
            def invalid_constant(value):
                raise ValueError(f"Non-finite JSON constant: {value}")
            event = json.loads(raw, parse_constant=invalid_constant)
            if not isinstance(event, dict) or event.get("type") not in ("stage", "spec", "result", "preview", "message"):
                raise ValueError("Unsupported event")
            if event["type"] == "result":
                if not isinstance(event.get("accepted"), bool):
                    raise ValueError("Result accepted must be boolean")
                return event
            if event["type"] == "spec":
                if not isinstance(event.get("spec"), dict) or event.get("validated") is not True:
                    raise ValueError("Only validated task-spec events are displayed")
                with job.condition:
                    job.spec = event
            job.publish(event)
        except (ValueError, TypeError, UnicodeDecodeError):
            job.publish({"type": "message", "level": "warning", "detail": "일부 진행 메시지를 읽지 못했습니다."})
        return None

    def _run(self, job):
        event_file = job.directory / "events.jsonl"
        offset, pending = 0, b""
        backend_result = None

        def read_events(final=False):
            nonlocal offset, pending, backend_result
            if event_file.is_file():
                with event_file.open("rb") as stream:
                    stream.seek(offset)
                    pending += stream.read()
                    offset = stream.tell()
                lines = pending.split(b"\n")
                pending = lines.pop()
                for line in lines:
                    if line.strip():
                        backend_result = self._read_event(job, line) or backend_result
            if final and pending.strip():
                backend_result = self._read_event(job, pending) or backend_result
                pending = b""

        try:
            with (job.directory / "worker.log").open("wb") as log:
                command = ["bash", str(self.worker), "--command", job.command, "--output", str(job.directory),
                           "--events", str(event_file), "--layout", job.layout]
                if job.show_sim:
                    command.append("--show-sim")
                if job.moving_person:
                    command.extend(["--moving-person", "--people-count", str(job.people_count)])
                process = subprocess.Popen(
                    command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with job.condition:
                    job.process = process
                    job.state = "cancelling" if job.cancel_requested else "running"
                if job.cancel_requested:
                    self._signal_group(job, signal.SIGTERM)
                while process.poll() is None:
                    read_events()
                    if job.cancel_started is not None and time.monotonic() - job.cancel_started >= CANCEL_GRACE_S:
                        self._signal_group(job, signal.SIGKILL)
                    time.sleep(.15)
                exit_code = process.wait()
                read_events(final=True)
            result = dict(backend_result or {})
            accepted = bool(backend_result and backend_result.get("accepted") is True and exit_code == 0
                            and not job.cancel_requested)
            if job.cancel_requested:
                detail = "작업 중지 요청에 따라 실행을 종료했습니다."
            elif backend_result is None:
                detail = "실행이 종료됐지만 최종 검증 결과를 받지 못했습니다."
            elif exit_code != 0 and backend_result.get("accepted") is True:
                # A worker claiming success and then failing its process contract
                # must be rejected. An ordinary rejected task may intentionally
                # exit nonzero; retain its useful, specific correction feedback.
                detail = f"백엔드는 통과 결과를 보냈지만 프로세스 종료 코드가 {exit_code}여서 통과로 인정하지 않았습니다."
            else:
                detail = result.get("detail") or result.get("message") or ("작업 검증을 통과했습니다." if accepted else "작업 검증을 통과하지 못했습니다.")
            if exit_code != 0:
                result["process_diagnostic"] = {
                    "exit_code": exit_code,
                    "reason": "accepted_result_nonzero_exit_mismatch" if backend_result and backend_result.get("accepted") is True
                    else "rejected_result_nonzero_exit" if backend_result else "missing_result_nonzero_exit",
                }
            result.update(type="result", accepted=accepted, exit_code=exit_code,
                          cancelled=job.cancel_requested, detail=detail)
            (job.directory / "web_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            with job.condition:
                job.state = "cancelled" if job.cancel_requested else "completed" if accepted else "failed"
                job.result = result
                job.publish(result)
        except Exception as error:
            if job.process is not None and job.process.poll() is None:
                self._signal_group(job, signal.SIGTERM)
                try:
                    job.process.wait(timeout=CANCEL_GRACE_S)
                except subprocess.TimeoutExpired:
                    self._signal_group(job, signal.SIGKILL)
                    job.process.wait()
            result = {"type": "result", "accepted": False, "detail": "작업 실행 연결에서 오류가 발생했습니다.",
                      "error": f"{type(error).__name__}: {error}", "cancelled": job.cancel_requested}
            with job.condition:
                job.state = "cancelled" if job.cancel_requested else "failed"
                job.result = result
                job.publish(result)
        finally:
            if job.process is not None and job.process.poll() is None:
                self._signal_group(job, signal.SIGTERM)
                try:
                    job.process.wait(timeout=CANCEL_GRACE_S)
                except subprocess.TimeoutExpired:
                    self._signal_group(job, signal.SIGKILL)
                    job.process.wait()

    @staticmethod
    def _signal_group(job, sig):
        try:
            if job.process is not None:
                os.killpg(job.process.pid, sig)
        except ProcessLookupError:
            pass

    def cancel(self, job_id):
        job = self.get(job_id)
        with job.condition:
            if job.state in TERMINAL:
                return job.snapshot()
            job.cancel_requested = True
            job.cancel_started = time.monotonic()
            job.state = "cancelling"
        job.publish({"type": "stage", "stage": "cancelling", "detail": "실행 중지를 요청했습니다. 자식 프로세스를 정리하고 있습니다."})
        self._signal_group(job, signal.SIGTERM)

        def force_stop():
            time.sleep(CANCEL_GRACE_S)
            if job.process is not None and job.process.poll() is None:
                self._signal_group(job, signal.SIGKILL)
        threading.Thread(target=force_stop, daemon=True).start()
        return job.snapshot()

    def close(self):
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            if job.state not in TERMINAL:
                self.cancel(job.id)
        for job in jobs:
            if job.process is not None and job.process.poll() is None:
                try:
                    job.process.wait(timeout=CANCEL_GRACE_S + 1)
                except subprocess.TimeoutExpired:
                    self._signal_group(job, signal.SIGKILL)


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, root):
        self.manager = JobManager(root)
        super().__init__(("127.0.0.1", port), Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # Job stdout/stderr has its own persistent worker.log.
        pass

    def _local(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            raise JobError("로컬 주소로 접속해 주세요.", 403)
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            raise JobError("다른 사이트에서는 작업을 시작할 수 없습니다.", 403)

    def _headers(self, status, content_type, length=None, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; font-src 'self'; connect-src 'self'; media-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def _json(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self._headers(status, "application/json; charset=utf-8", len(content))
        self.wfile.write(content)

    def _file(self, path):
        if not path.is_file():
            raise JobError("파일을 찾을 수 없습니다.", 404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        headers = {"Accept-Ranges": "bytes"}
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if not match or not any(match.groups()) or size == 0:
                self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
            if start >= size or start > end:
                self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
                return
            status = 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
            mime += "; charset=utf-8"
        length = max(0, end - start + 1)
        self._headers(status, mime, length, headers)
        with path.open("rb") as content:
            content.seek(start)
            while length:
                block = content.read(min(length, 65536))
                if not block:
                    break
                self.wfile.write(block)
                length -= len(block)

    @staticmethod
    def _within(root, relative):
        root = root.resolve()
        path = (root / unquote(relative)).resolve()
        if not path.is_relative_to(root):
            raise JobError("허용되지 않은 파일 경로입니다.", 403)
        return path

    def do_GET(self):
        try:
            self._local()
            url = urlsplit(self.path)
            parts = url.path.strip("/").split("/")
            if url.path == "/api/status":
                return self._json(self.server.manager.status())
            if len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
                job = self.server.manager.get(parts[2])
                if len(parts) == 3:
                    return self._json(job.snapshot())
                if len(parts) == 4 and parts[3] == "events":
                    after = max(int(parse_qs(url.query).get("after", [0])[0]), int(self.headers.get("Last-Event-ID", 0)))
                    return self._events(job, after)
            if len(parts) >= 3 and parts[0] == "artifacts":
                job = self.server.manager.get(parts[1])
                return self._file(self._within(job.directory, "/".join(parts[2:])))
            if url.path == "/":
                return self._file(STATIC / "index.html")
            if url.path.startswith("/static/"):
                return self._file(self._within(STATIC, url.path.removeprefix("/static/")))
            raise JobError("페이지를 찾을 수 없습니다.", 404)
        except JobError as error:
            self._json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, OSError) as error:
            self._json({"error": str(error)}, 400)

    def _events(self, job, after):
        self._headers(200, "text/event-stream; charset=utf-8")
        self.wfile.write(b": connected\n\n")
        self.wfile.flush()
        heartbeat = time.monotonic()
        try:
            while True:
                with job.condition:
                    events = list(job.events[after:])
                    terminal = job.state in TERMINAL
                    if not events and not terminal:
                        job.condition.wait(timeout=1)
                for event in events:
                    encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
                    self.wfile.write(f"id: {event['seq']}\ndata: {encoded}\n\n".encode())
                    after = event["seq"]
                if events or time.monotonic() - heartbeat > 10:
                    if not events:
                        self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    heartbeat = time.monotonic()
                if terminal:
                    self.close_connection = True
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            self._local()
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise JobError("JSON 요청이 필요합니다.", 415)
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 16384:
                raise JobError("요청 크기가 허용 범위를 벗어났습니다.", 413)
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/jobs":
                if not isinstance(body, dict) or "command" not in body or set(body) - {"command", "show_sim", "moving_person", "layout", "people_count"}:
                    raise JobError("command와 선택적 show_sim, moving_person, layout, people_count 필드만 보낼 수 있습니다.")
                return self._json(self.server.manager.start(body["command"], body.get("show_sim", False),
                                                           body.get("moving_person", False), body.get("layout", "rack-v1"),
                                                           body.get("people_count", 3)), 202)
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})/cancel", self.path)
            if match:
                return self._json(self.server.manager.cancel(match[1]))
            raise JobError("요청 경로를 찾을 수 없습니다.", 404)
        except JobError as error:
            self._json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, OSError) as error:
            self._json({"error": str(error)}, 400)
