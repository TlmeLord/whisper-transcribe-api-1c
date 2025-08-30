# Язык определяется автоматически (autodetect).
# При наличии GPU можно заменить device="cpu" на device="cuda".

import os
import asyncio
import time
import inspect
from typing import List, Dict, Optional
from faster_whisper import WhisperModel, BatchedInferencePipeline

from .queue import queue_manager, Task  # для обновления прогресса
from .config import get_settings as get_app_settings

_model: Optional[WhisperModel] = None
_device_mode: Optional[str] = None  # 'cuda' | 'cpu'


def _get_env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _init_model(prefer: str) -> WhisperModel:
    """
    Инициализация модели с учётом параметров качества/производительности из переменных окружения.
    - WHISPER_MODEL: tiny|base|small|medium|large-v3 ... (по умолчанию 'small')
    - WHISPER_COMPUTE_TYPE: int8|int8_float16|float16|int8_float32 ... (по умолчанию зависит от устройства)
    """
    prefer = (prefer or "auto").lower()
    settings = get_app_settings()
    model_name = settings.WHISPER_MODEL or "small"
    compute_type_cfg = settings.WHISPER_COMPUTE_TYPE

    def build(device: str) -> WhisperModel:
        # Выбираем compute_type: для CUDA по умолчанию float16, для CPU — int8
        if compute_type_cfg:
            ct = compute_type_cfg
        else:
            ct = "float16" if device == "cuda" else "int8"
        return WhisperModel(model_name, device=device, compute_type=ct)

    if prefer == "cpu":
        return build("cpu")
    if prefer == "cuda":
        return build("cuda")
    # auto
    try:
        return build("cuda")
    except Exception:
        return build("cpu")


def get_model() -> WhisperModel:
    """
    Возвращает экземпляр модели Whisper с учётом настроек (auto|cuda|cpu).
    Дополнительно читаются WHISPER_MODEL, WHISPER_COMPUTE_TYPE.
    """
    global _model, _device_mode
    if _model is None:
        prefer = (get_app_settings().WHISPER_DEVICE or "auto").lower()
        try:
            _model = _init_model(prefer)
        except Exception:
            s = get_app_settings()
            _model = WhisperModel(s.WHISPER_MODEL or "small", device="cpu", compute_type=(s.WHISPER_COMPUTE_TYPE or "int8"))
        try:
            _device_mode = "cuda" if prefer == "cuda" else prefer
        except Exception:
            _device_mode = "cpu"
    return _model


def _force_cpu_model() -> WhisperModel:
    global _model, _device_mode
    s = get_app_settings()
    _model = WhisperModel(s.WHISPER_MODEL or "small", device="cpu", compute_type=(s.WHISPER_COMPUTE_TYPE or "int8"))
    _device_mode = "cpu"
    return _model


def reset_model() -> None:
    """Сбрасывает кэшированную модель. Новые настройки применятся при следующем get_model()."""
    global _model
    _model = None


async def transcribe(task: Task) -> Dict:
    """Асинхронная обёртка над блокирующей транскрибацией с обновлением прогресса."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transcribe_blocking, task)


def _estimate_eta(start_time: float, progress: float) -> Optional[float]:
    if progress <= 0:
        return None
    elapsed = time.time() - start_time
    remaining = elapsed * (1 - progress) / max(progress, 1e-6)
    return max(0.0, remaining)


def _filter_kwargs(func, kwargs: Dict) -> Dict:
    """Фильтрует kwargs по сигнатуре функции, чтобы избежать несовместимых аргументов между версиями faster-whisper."""
    try:
        params = set(inspect.signature(func).parameters.keys())
    except Exception:
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if (k in params and v is not None)}


def _transcribe_blocking(task: Task) -> Dict:
    """
    Блокирующая транскрибация файла.
    При ошибке CUDA в рантайме выполняет одно повторение на CPU.
    Обновляет прогресс задачи по мере обработки сегментов.
    """
    model = get_model()
    try:
        return _do_transcribe(model, task)
    except Exception as e:
        # Если ошибка могла быть связана с CUDA/библиотеками — пробуем один раз на CPU
        try:
            cpu_model = _force_cpu_model()
            return _do_transcribe(cpu_model, task)
        except Exception:
            raise e


def _do_transcribe(model: WhisperModel, task: Task) -> Dict:
    file_path = task.file_path
    # Параметры распознавания из типизированных настроек
    s = get_app_settings()
    language = s.WHISPER_LANGUAGE  # например, "ru" для фиксированного русского (None => autodetect)
    task_mode = (s.WHISPER_TASK or "transcribe")
    beam_size = int(s.WHISPER_BEAM or 8)
    patience = float(s.WHISPER_PATIENCE or 1.0)
    length_penalty = float(s.WHISPER_LENGTH_PENALTY or 1.0)
    condition_prev = bool(s.WHISPER_CONDITION_PREV)
    initial_prompt = s.WHISPER_PROMPT  # подсказка стилю/лексике, можно на русском
    vad_enabled = bool(getattr(s, "WHISPER_VAD_ENABLED", True))
    vad_min_sil = int(getattr(s, "WHISPER_VAD_MIN_SIL_MS", 300) or 300)
    log_progress = bool(getattr(s, "WHISPER_LOG_PROGRESS", False))
    word_timestamps = bool(getattr(s, "WHISPER_WORD_TIMESTAMPS", False))
    multilingual = bool(getattr(s, "WHISPER_MULTILINGUAL", False))
    hotwords = getattr(s, "WHISPER_HOTWORDS", None)
    max_new_tokens = getattr(s, "WHISPER_MAX_NEW_TOKENS", None)
    batched = bool(getattr(s, "WHISPER_BATCHED", False))
    batch_size = int(getattr(s, "WHISPER_BATCH_SIZE", 8) or 8)

    common_kwargs = dict(
        language=language if language else None,
        task=task_mode,
        beam_size=beam_size,
        patience=patience,
        length_penalty=length_penalty,
        vad_filter=vad_enabled,
        vad_parameters={"min_silence_duration_ms": vad_min_sil},
        condition_on_previous_text=condition_prev,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
        multilingual=multilingual,
        hotwords=hotwords,
        max_new_tokens=max_new_tokens,
        log_progress=log_progress,
    )

    if batched:
        pipeline = BatchedInferencePipeline(model)
        call_kwargs = {**common_kwargs, "batch_size": batch_size}
        call_kwargs = _filter_kwargs(pipeline.transcribe, call_kwargs)
        segments, info = pipeline.transcribe(file_path, **call_kwargs)
    else:
        call_kwargs = _filter_kwargs(model.transcribe, common_kwargs)
        segments, info = model.transcribe(file_path, **call_kwargs)

    total_duration = getattr(info, "duration", None)
    processed_time = 0.0
    phrases: List[Dict] = []

    for seg in segments:
        # Кооперативная отмена: если задача помечена как отменённая — прерываем обработку
        if queue_manager.is_cancelled(task.id):
            raise Exception("Отменено")
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", start))
        text = (getattr(seg, "text", "") or "").strip()
        phrase: Dict = {"start": start, "end": end, "text": text}
        if word_timestamps and getattr(seg, "words", None):
            words_out = []
            for w in getattr(seg, "words", []) or []:
                words_out.append({
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0)),
                    "word": str(getattr(w, "word", "")),
                    "probability": getattr(w, "probability", None),
                })
            if words_out:
                phrase["words"] = words_out
        phrases.append(phrase)
        processed_time = max(processed_time, end)
        # Обновляем прогресс/ETA
        if total_duration and total_duration > 0:
            progress = min(1.0, max(0.0, processed_time / total_duration))
        else:
            progress = None
        eta = _estimate_eta(task.started_at or time.time(), progress) if progress is not None else None
        queue_manager.update_progress(task.id, progress, eta)

    return {"phrases": phrases}