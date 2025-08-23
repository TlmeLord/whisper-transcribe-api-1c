import os
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import CreateTaskRequest, TaskStatusResponse, UploadResponse, TaskStatus, TaskResult, TaskSummary, TaskDetail
from .queue import queue_manager, Task
from .whisper_service import transcribe, reset_model
from .config import get_settings as get_app_settings, reload_settings

# Корневая директория приложения и каталог для загрузок
APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", APP_ROOT / "uploads"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", APP_ROOT / "app" / "static"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Whisper Transcribe API", version="1.2.0")

# Разрешаем CORS для локальной разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Сжатие ответов (GZip)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Раздача статики и простая страница UI
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index_page():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    # Fallback, если файл не создан
    return HTMLResponse("<h1>Whisper UI</h1><p>Статический index.html не найден.</p>")


async def process_task(task: Task):
    return await transcribe(task)


@app.on_event("startup")
async def on_startup():
    # Фоновая обработка: транскрибация аудио
    await queue_manager.start_worker(process_task)


@app.on_event("shutdown")
async def on_shutdown():
    # Корректное завершение воркера очереди
    await queue_manager.shutdown()


@app.get("/health")
async def health():
    # Простой индикатор живости/готовности сервиса
    return {"status": "ok"}


@app.get("/api/v1/uploads")
async def list_uploads():
    """Возвращает список файлов из каталога uploads для выбора без повторной загрузки."""
    files = []
    try:
        for p in sorted(UPLOAD_DIR.glob("*")):
            if p.is_file():
                st = p.stat()
                files.append({
                    "name": p.name,
                    "path": str(p.resolve()),
                    "size": st.st_size,
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось прочитать uploads: {e}")
    return {"files": files}


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    # Принимаем multipart и сохраняем файл в каталог uploads
    filename = Path(file.filename).name
    if not filename:
        raise HTTPException(status_code=400, detail="Требуется имя файла")
    dest_path = UPLOAD_DIR / filename
    # Обеспечиваем уникальность имени
    i = 1
    stem = dest_path.stem
    suffix = dest_path.suffix
    while dest_path.exists():
        dest_path = UPLOAD_DIR / f"{stem}_{i}{suffix}"
        i += 1
    try:
        with dest_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()
    return UploadResponse(stored_path=str(dest_path.resolve()))


@app.post("/api/v1/create_task", response_model=TaskStatusResponse)
async def create_task(req: CreateTaskRequest):
    file_path = Path(req.file_path)
    if not file_path.is_absolute():
        raise HTTPException(status_code=422, detail="file_path должен быть абсолютным путём")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    try:
        task = queue_manager.create_task(str(file_path))
    except ValueError as e:
        # Например: неподдерживаемое расширение файла
        raise HTTPException(status_code=415, detail=str(e))
    return TaskStatusResponse(task_id=task.id, status=TaskStatus.PENDING, queue_position=queue_manager.queue_position(task.id))


@app.get("/api/v1/status/{task_id}", response_model=TaskStatusResponse)
async def status(task_id: str):
    task = queue_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    resp = TaskStatusResponse(
        task_id=task.id,
        status=task.status,
    )
    if task.status == TaskStatus.PENDING:
        pos = queue_manager.queue_position(task.id)
        resp.queue_position = pos if pos is not None else 0
    elif task.status == TaskStatus.IN_PROGRESS:
        resp.progress = task.progress
        resp.eta_seconds = task.eta_seconds
    elif task.status == TaskStatus.COMPLETED:
        resp.results = TaskResult.model_validate(task.result) if task.result else None
        resp.progress = 1.0
        resp.eta_seconds = 0.0
    elif task.status == TaskStatus.FAILED:
        resp.error = task.error or "Неизвестная ошибка"
    return resp


# Экспорт результатов: txt/srt/json
@app.get("/export/{task_id}.{fmt}")
async def export_result(task_id: str, fmt: str):
    task = queue_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if task.status != TaskStatus.COMPLETED or not task.result:
        raise HTTPException(status_code=409, detail="Задача не завершена или нет результата")

    phrases = task.result.get("phrases") if isinstance(task.result, dict) else task.result.phrases

    def to_srt(phs):
        def fmt_ts(t: float) -> str:
            # Гарантируем корректное округление миллисекунд и перенос через секунду/минуту/час
            if t is None:
                t = 0.0
            t = max(0.0, float(t))
            total_ms = int(round(t * 1000))
            ms = total_ms % 1000
            total_sec = total_ms // 1000
            s = total_sec % 60
            total_min = total_sec // 60
            m = total_min % 60
            h = total_min // 60
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        lines = []
        for i, p in enumerate(phs, 1):
            start = fmt_ts(float(p["start"]) if isinstance(p, dict) else p.start)
            end = fmt_ts(float(p["end"]) if isinstance(p, dict) else p.end)
            text = (p["text"] if isinstance(p, dict) else p.text).strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def to_txt(phs):
        return "\n".join((p["text"] if isinstance(p, dict) else p.text).strip() for p in phs)

    def to_dict(p):
        if isinstance(p, dict):
            return {
                "start": float(p.get("start", 0.0)),
                "end": float(p.get("end", 0.0)),
                "text": str(p.get("text", "")).strip(),
            }
        return {
            "start": float(getattr(p, "start", 0.0)),
            "end": float(getattr(p, "end", 0.0)),
            "text": str(getattr(p, "text", "")).strip(),
        }

    fmt = fmt.lower()
    if fmt == "txt":
        content = to_txt(phrases)
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")
    if fmt == "srt":
        content = to_srt(phrases)
        return PlainTextResponse(content, media_type="application/x-subrip; charset=utf-8")
    if fmt == "json":
        safe_phrases = [to_dict(p) for p in (phrases or [])]
        return JSONResponse({"phrases": safe_phrases})
    raise HTTPException(status_code=400, detail="Неподдерживаемый формат. Используйте txt|srt|json")


# Типизированные настройки (runtime)
@app.get("/api/v1/settings")
async def get_settings_handler():
    return get_app_settings().model_dump()


@app.post("/api/v1/settings")
async def set_settings(payload: dict):
    # Применение настроек
    changed: dict[str, Optional[str]] = {}
    for k, v in payload.items():
        if v is None or v == "":
            if k in os.environ:
                del os.environ[k]
            changed[k] = None
        else:
            os.environ[k] = str(v)
            changed[k] = os.environ[k]
    # Перечитывает настройки и сбрасываем модель
    reload_settings()
    reset_model()
    return {"updated": changed, "effective": get_app_settings().model_dump()}


@app.post("/api/v1/reset")
async def hard_reset():
    # Хард‑ресет: отменяет текущую и ожидающие, очищает очередь, останавливает воркер и запускает заново, сбрасывает модель.
    queue_manager.cancel_all()
    removed = queue_manager.clear_queue()
    # Останавливаем текущий воркер
    await queue_manager.shutdown()
    # Сбрасываем модель
    reset_model()
    # Запускаем воркер заново
    await queue_manager.start_worker(process_task)
    return {"removed_pending": removed, "model_reset": True, "worker_restarted": True}


# Отмена задачи
@app.post("/api/v1/cancel/{task_id}")
async def cancel_task(task_id: str):
    t = queue_manager.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    ok = queue_manager.request_cancel(task_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Невозможно отменить задачу")
    return {"cancelled": True}


# Эндпоинты просмотра задач 
@app.get("/api/v1/tasks")
async def list_tasks():
    # Возвращает краткую информацию по всем задачам
    items = []
    for t in queue_manager.tasks.values():
        items.append(TaskSummary(
            task_id=t.id,
            file_path=t.file_path,
            status=t.status,
            progress=t.progress,
            eta_seconds=t.eta_seconds,
            enqueued_at=t.enqueued_at,
            started_at=t.started_at,
            finished_at=t.finished_at,
            has_result=bool(t.result),
            error=t.error,
        ).model_dump())
    # Можно сортировать по времени постановки (свежие сверху)
    items.sort(key=lambda x: x.get("enqueued_at", 0), reverse=True)
    return {"tasks": items}


@app.get("/api/v1/tasks/{task_id}", response_model=TaskDetail)
async def get_task_detail(task_id: str):
    t = queue_manager.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    detail = TaskDetail(
        task_id=t.id,
        file_path=t.file_path,
        status=t.status,
        progress=t.progress,
        eta_seconds=t.eta_seconds,
        enqueued_at=t.enqueued_at,
        started_at=t.started_at,
        finished_at=t.finished_at,
        results=TaskResult.model_validate(t.result) if t.result else None,
        error=t.error,
    )
    return detail


# WebSocket поток статуса задачи
@app.websocket("/ws/status/{task_id}")
async def ws_status(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        while True:
            t = queue_manager.get_task(task_id)
            if not t:
                await websocket.send_json({"error": "Задача не найдена", "task_id": task_id})
                await asyncio.sleep(1.0)
                continue

            resp = {
                "task_id": t.id,
                "status": t.status,
            }
            if t.status == TaskStatus.PENDING:
                pos = queue_manager.queue_position(t.id)
                resp["queue_position"] = pos if pos is not None else 0
            elif t.status == TaskStatus.IN_PROGRESS:
                resp["progress"] = t.progress
                resp["eta_seconds"] = t.eta_seconds
            elif t.status == TaskStatus.COMPLETED:
                try:
                    resp["results"] = TaskResult.model_validate(t.result).model_dump() if t.result else None
                except Exception:
                    resp["results"] = t.result
                resp["progress"] = 1.0
                resp["eta_seconds"] = 0.0
            elif t.status == TaskStatus.FAILED:
                resp["error"] = t.error or "Неизвестная ошибка"

            await websocket.send_json(resp)
            if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                break
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
