from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CreateTaskRequest(BaseModel):
    file_path: str = Field(..., description="Абсолютный путь к сохранённому аудиофайлу")


class Word(BaseModel):
    start: float
    end: float
    word: str
    probability: Optional[float] = None


class Phrase(BaseModel):
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None


class TaskResult(BaseModel):
    phrases: List[Phrase]


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    queue_position: Optional[int] = None
    results: Optional[TaskResult] = None
    error: Optional[str] = None
    # Поля для индикации прогресса
    progress: Optional[float] = None        # 0..1
    eta_seconds: Optional[float] = None     # ожидаемое время до завершения в секундах


# Модели для просмотра задач
class TaskSummary(BaseModel):
    task_id: str
    file_path: str
    status: TaskStatus
    progress: Optional[float] = None
    eta_seconds: Optional[float] = None
    enqueued_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    has_result: bool = False
    error: Optional[str] = None


class TaskDetail(BaseModel):
    task_id: str
    file_path: str
    status: TaskStatus
    progress: Optional[float] = None
    eta_seconds: Optional[float] = None
    enqueued_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    results: Optional[TaskResult] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    stored_path: str


class Preset(BaseModel):
    name: str = Field(..., min_length=1, description="Уникальное имя пресета")
    description: Optional[str] = Field(default=None, description="Краткое описание")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Параметры распознавания в виде словаря переменных окружения")


class PresetsList(BaseModel):
    presets: List[Preset]


class PresetUpsertRequest(BaseModel):
    preset: Preset
    overwrite: bool = False


class PresetDeleteResponse(BaseModel):
    deleted: bool
    remaining: int


class PresetImportRequest(BaseModel):
    presets: List[Preset]
    overwrite: bool = False
