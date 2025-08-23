import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schemas import TaskStatus


# Поддерживаемые расширения аудиофайлов
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def _ext_ok(path: str) -> bool:
    import os
    _, ext = os.path.splitext(path.lower())
    return ext in SUPPORTED_EXTENSIONS


@dataclass
class Task:
    id: str
    file_path: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[dict] = None
    error: Optional[str] = None
    enqueued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Прогресс выполнения 0..1 и ожидаемое время до завершения
    progress: Optional[float] = None
    eta_seconds: Optional[float] = None


class QueueManager:
    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}
        self._pending: List[str] = []  # простая очередь на базе списка (task_id)
        self._cv = asyncio.Condition()
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._cancelled: set[str] = set()
        self._current_task_id: Optional[str] = None

    async def start_worker(self, process_func):
        # process_func: async callable(task: Task) -> dict — выполняет транскрибацию
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._worker_loop(process_func))
        asyncio.create_task(self._notify())

    async def shutdown(self):
        self._shutdown = True
        async with self._cv:
            self._cv.notify_all()
        if self._worker_task:
            await self._worker_task
            self._worker_task = None
            
    def create_task(self, file_path: str) -> Task:
        if not _ext_ok(file_path):
            raise ValueError("Неподдерживаемый тип файла")
        task_id = str(uuid.uuid4())
        task = Task(id=task_id, file_path=file_path)
        self.tasks[task_id] = task
        # Постановка в очередь
        self._pending.append(task_id)
        asyncio.create_task(self._notify())
        return task

    async def _notify(self):
        async with self._cv:
            self._cv.notify(1)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def queue_position(self, task_id: str) -> Optional[int]:
        try:
            return self._pending.index(task_id)
        except ValueError:
            return None

    def update_progress(self, task_id: str, progress: Optional[float], eta_seconds: Optional[float]):
        t = self.tasks.get(task_id)
        if not t:
            return
        t.progress = None if progress is None else max(0.0, min(1.0, float(progress)))
        t.eta_seconds = None if eta_seconds is None else max(0.0, float(eta_seconds))

    # ===== Отмена задач =====
    def request_cancel(self, task_id: str) -> bool:
        """Помечает задачу на отмену. Возвращает True, если задача известна."""
        if task_id in self.tasks:
            self._cancelled.add(task_id)
            return True
        return False

    def cancel_all(self) -> int:
        """Отменяет текущую и все ожидающие задачи. Возвращает количество помеченных к отмене."""
        to_cancel = set(self._pending)
        if self._current_task_id:
            to_cancel.add(self._current_task_id)
        before = len(self._cancelled)
        self._cancelled.update(to_cancel)
        return len(self._cancelled) - before

    def is_cancelled(self, task_id: str) -> bool:
        return task_id in self._cancelled or self._shutdown

    def clear_queue(self) -> int:
        # Удаляет все невыполненные задачи из очереди. Возвращает количество удалённых.
        n = len(self._pending)
        self._pending.clear()
        # Статусы ожидающих меняем на FAILED с текстом
        for task_id, t in list(self.tasks.items()):
            if t.status == TaskStatus.PENDING:
                t.status = TaskStatus.FAILED
                t.error = "Отклонено: хард‑ресет"
                t.finished_at = time.time()
                # И помечаем их отменёнными, чтобы воркер не взял повторно
                self._cancelled.add(task_id)
        return n

    async def _worker_loop(self, process_func):
        while not self._shutdown:
            task_id: Optional[str] = None
            async with self._cv:
                while not self._pending and not self._shutdown:
                    await self._cv.wait()
                if self._shutdown:
                    break
                task_id = self._pending.pop(0)
            if not task_id:
                continue
            task = self.tasks.get(task_id)
            if not task:
                continue
            if self.is_cancelled(task_id):
                # Если успели отменить до старта — помечаем как FAILED и пропускаем
                task.status = TaskStatus.FAILED
                task.error = "Отменено"
                task.finished_at = time.time()
                continue
            # Обработка задачи
            task.status = TaskStatus.IN_PROGRESS
            self._current_task_id = task_id
            task.started_at = time.time()
            task.progress = 0.0
            task.eta_seconds = None
            try:
                result = await process_func(task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.progress = 1.0
                task.eta_seconds = 0.0
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            finally:
                task.finished_at = time.time()
                # Сбрасываем текущую
                if self._current_task_id == task_id:
                    self._current_task_id = None
                # Снимаем флаг отмены для завершённых
                self._cancelled.discard(task_id)


# Синглтон менеджера очереди для приложения
queue_manager = QueueManager()
