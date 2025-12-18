"""
FastAPI сервис для обработки видео и детекции нарушений
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from pathlib import Path
from datetime import datetime, date
import json
import threading
from detector import DefecationDetector, dog_detect_model, pose_model, SEQ_LENGTH
import mysql.connector
from mysql.connector import Error

app = FastAPI(title="DogsNeuro API", version="1.0.0")

# CORS для работы с Laravel фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для хранения результатов и прогресса
# Используем threading.Lock для безопасного доступа из разных потоков
import threading
progress_lock = threading.Lock()
results_storage = {}
progress_storage = {}  # Хранилище прогресса обработки по video_id
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)

# Инициализация детектора (кэшируется)
detector_instance = None

def get_detector():
    """Получение экземпляра детектора (singleton)"""
    global detector_instance
    if detector_instance is None:
        detector_instance = DefecationDetector(
            lstm_path="models/structured_lstm_model_final.pth",
            dog_detect_model=dog_detect_model,
            pose_model=pose_model,
            window_size=SEQ_LENGTH,
            threshold=0.8,
            smooth=5
        )
    return detector_instance


class ViolationResponse(BaseModel):
    """Модель ответа с нарушением"""
    time: str
    type: str
    description: str
    source: str
    date: str
    video_url: Optional[str] = None
    breed: Optional[str] = None
    muzzle: Optional[bool] = None


class ProcessingResponse(BaseModel):
    """Модель ответа обработки видео"""
    success: bool
    video_id: str
    violations: List[ViolationResponse] = []
    processing_time: Optional[str] = None
    message: Optional[str] = None


class ViolationData(BaseModel):
    """Модель данных нарушения для сохранения"""
    time: str
    type: str
    description: str
    source: str
    date: str
    video_url: Optional[str] = None
    breed: Optional[str] = None
    muzzle: Optional[bool] = None


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {"message": "DogsNeuro API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def process_video_sync(video_id: str, input_path: str, output_path: Path, filename: str):
    """Синхронная обработка видео в отдельном потоке"""
    try:
        # Callback для обновления прогресса
        def update_progress(percent, total_frames, status):
            try:
                progress_data = {
                    "percent": min(percent, 100),
                    "current_frame": int(total_frames * percent / 100) if total_frames > 0 else 0,
                    "total_frames": total_frames,
                    "status": status or "Обработка..."
                }
                # Безопасное обновление из потока
                with progress_lock:
                    progress_storage[video_id] = progress_data
                # Логируем для отладки (выводим в stdout, чтобы было видно в docker logs)
                import sys
                sys.stdout.write(f"[PROGRESS] {video_id}: {percent}% ({progress_data['current_frame']}/{total_frames}) - {status}\n")
                sys.stdout.flush()
            except Exception as e:
                import sys
                sys.stdout.write(f"[ERROR] Ошибка обновления прогресса для {video_id}: {e}\n")
                sys.stdout.flush()
        
        # Обработка видео с callback прогресса
        import sys
        sys.stdout.write(f"[PROCESS] Создаю детектор с callback для {video_id}\n")
        sys.stdout.flush()
        
        from detector import DefecationDetector
        # frame_skip=2 означает обрабатывать каждый 2-й кадр (ускоряет в ~2 раза)
        # Можно увеличить до 3-4 для еще большего ускорения, но точность снизится
        detector = DefecationDetector(
            lstm_path="models/structured_lstm_model_final.pth",
            dog_detect_model=dog_detect_model,
            pose_model=pose_model,
            window_size=SEQ_LENGTH,
            threshold=0.8,
            smooth=5,
            progress_callback=update_progress,
            frame_skip=2  # Обрабатываем каждый 2-й кадр для ускорения (~2x быстрее)
        )
        
        sys.stdout.write(f"[PROCESS] Детектор создан, callback установлен: {detector.progress_callback is not None}\n")
        sys.stdout.flush()
        
        start_time = datetime.now()
        detector.run_video(str(input_path), str(output_path))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Извлекаем информацию о нарушениях из детектора
        violations = []
        
        fps = detector.fps if hasattr(detector, 'fps') else 30
        violation_interval_seconds = 120  # 2 минуты для группировки нарушений
        violation_interval_frames = int(violation_interval_seconds * fps)
        
        # Используем сохраненные периоды нарушений из детектора
        # Нарушения фиксируются только после истечения 1 минуты без уборки
        violation_periods = []
        
        if hasattr(detector, 'violation_periods') and detector.violation_periods:
            # Используем реальные периоды нарушений (когда прошла 1 минута без уборки)
            violation_periods = detector.violation_periods.copy()
        else:
            # Fallback: если violation_periods не заполнен, проверяем violation_active
            # Это может быть, если обработка завершилась во время активного нарушения
            if hasattr(detector, 'violation_active') and detector.violation_active:
                if hasattr(detector, 'violation_start_frame') and detector.violation_start_frame is not None:
                    # Получаем общее количество кадров из hist
                    total_frames = len(detector.hist) + SEQ_LENGTH if hasattr(detector, 'hist') and detector.hist else 0
                    if total_frames > 0:
                        violation_periods = [(detector.violation_start_frame, total_frames)]
        
        # Группируем периоды нарушений: одна запись на каждые 2 минуты
        grouped_periods = []
        for start_frame, end_frame in violation_periods:
            period_start = start_frame
            while period_start < end_frame:
                period_end = min(period_start + violation_interval_frames, end_frame)
                grouped_periods.append((period_start, period_end))
                period_start = period_end
        
        # Создаем одну запись на каждый период (группируем по 2 минутам)
        for start_frame, end_frame in grouped_periods:
            time_seconds = start_frame / fps
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = int(time_seconds % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            duration_seconds = int((end_frame - start_frame) / fps)
            
            violation = {
                "time": time_str,
                "type": "Неубранные экскременты",
                "description": f"Экскременты не были убраны в течение 1 минуты после дефекации. Длительность нарушения: {duration_seconds} сек.",
                "source": filename,
                "date": datetime.now().strftime('%Y-%m-%d'),
                "video_url": f"/api/v1/video/{video_id}",
                "breed": None,
                "muzzle": None
            }
            violations.append(violation)
        
        # Сохраняем результаты
        results_storage[video_id] = {
            "violations": violations,
            "processing_time": f"{processing_time:.1f}s",
            "output_path": str(output_path),
            "input_filename": filename,
            "created_at": datetime.now().isoformat()
        }
        
        # Удаляем временный входной файл
        try:
            os.unlink(input_path)
        except:
            pass
        
        # Обновляем прогресс на 100%
        with progress_lock:
            progress_storage[video_id] = {
                "percent": 100,
                "status": "Обработка завершена!",
                "completed": True
            }
        import sys
        sys.stdout.write(f"[PROGRESS] {video_id}: 100% - Обработка завершена!\n")
        sys.stdout.flush()
    except Exception as e:
        with progress_lock:
            progress_storage[video_id] = {
                "percent": 0,
                "status": f"Ошибка: {str(e)}",
                "completed": False,
                "error": str(e)
            }
        import sys
        sys.stdout.write(f"[ERROR] Ошибка обработки {video_id}: {e}\n")
        sys.stdout.flush()
        # Удаляем временный файл при ошибке
        try:
            os.unlink(input_path)
        except:
            pass


@app.post("/api/v1/process-video", response_model=ProcessingResponse)
async def process_video(file: UploadFile = File(...)):
    """
    Обработка загруженного видео (запускается асинхронно)
    """
    try:
        # Проверка формата файла
        if not file.filename:
            raise HTTPException(status_code=400, detail="Файл не указан")
        
        allowed_extensions = ['.mp4', '.avi', '.mov']
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Неподдерживаемый формат. Разрешены: {', '.join(allowed_extensions)}"
            )
        
        # Создаем временный файл для входного видео
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_input:
            content = await file.read()
            temp_input.write(content)
            input_path = temp_input.name
        
        # Генерируем уникальный ID для видео
        video_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(file.filename).stem}"
        output_path = output_dir / f"processed_{video_id}.mp4"
        
        # Инициализируем прогресс
        with progress_lock:
            progress_storage[video_id] = {
                "percent": 0,
                "current_frame": 0,
                "total_frames": 0,
                "status": "Инициализация..."
            }
        import sys
        sys.stdout.write(f"[API] Инициализирован прогресс для {video_id}\n")
        sys.stdout.flush()
        
        # Запускаем обработку в отдельном потоке (асинхронно)
        thread = threading.Thread(
            target=process_video_sync,
            args=(video_id, input_path, output_path, file.filename),
            name=f"VideoProcessing-{video_id}"
        )
        thread.daemon = True
        thread.start()
        import sys
        sys.stdout.write(f"[API] Запущен поток обработки для {video_id}\n")
        sys.stdout.flush()
        
        # Сразу возвращаем video_id, обработка идет в фоне
        return ProcessingResponse(
            success=True,
            video_id=video_id,
            violations=[],  # Нарушения будут доступны после завершения обработки
            processing_time=None,
            message="Обработка запущена. Используйте /api/v1/progress/{video_id} для отслеживания прогресса."
        )
        
    except Exception as e:
        # Удаляем временные файлы при ошибке
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Ошибка обработки видео: {str(e)}")


@app.get("/api/v1/video/{video_id}")
async def get_processed_video(video_id: str):
    """Получение обработанного видео"""
    if video_id not in results_storage:
        raise HTTPException(status_code=404, detail="Видео не найдено")
    
    output_path = Path(results_storage[video_id]["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Файл видео не найден")
    
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"processed_{video_id}.mp4"
    )


@app.get("/api/v1/violations", response_model=List[ViolationResponse])
async def get_violations(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Получение списка всех нарушений
    Можно фильтровать по датам
    """
    all_violations = []
    
    for video_id, data in results_storage.items():
        violations = data.get("violations", [])
        for violation in violations:
            violation_date = violation.get("date", "")
            
            # Фильтрация по датам
            if start_date and violation_date < start_date:
                continue
            if end_date and violation_date > end_date:
                continue
            
            all_violations.append(ViolationResponse(**violation))
    
    # Сортируем по дате и времени
    all_violations.sort(key=lambda x: (x.date, x.time))
    
    return all_violations


def get_db_connection():
    """Получение подключения к MySQL"""
    try:
        conn = mysql.connector.connect(
            host=os.environ.get('MYSQL_HOST', 'neurodog-mysql'),
            port=int(os.environ.get('MYSQL_PORT', 3306)),
            database=os.environ.get('MYSQL_DATABASE', 'laravel'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', 'secret'),
            autocommit=True
        )
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


@app.post("/api/v1/violations/save")
async def save_violations(violations: List[ViolationData], video_id: Optional[str] = None):
    """Сохранение нарушений в базу данных"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        cursor = conn.cursor()
        saved_count = 0
        
        for violation in violations:
            cursor.execute("""
                INSERT INTO violations 
                (time, type, description, source, date, video_id, video_url, breed, muzzle)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                violation.time,
                violation.type,
                violation.description,
                violation.source,
                violation.date,
                video_id,
                violation.video_url,
                violation.breed,
                violation.muzzle
            ))
            saved_count += 1
        
        conn.commit()
        return {"success": True, "saved": saved_count, "message": f"Saved {saved_count} violations"}
    except Error as e:
        print(f"Error saving violations: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


# ВАЖНО: Более специфичные маршруты должны быть ПЕРЕД параметризованными
# Иначе /api/v1/violations/db будет перехватываться /api/v1/violations/{video_id}
@app.get("/api/v1/violations/db")
async def get_violations_from_db(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    video_id: Optional[str] = None
):
    """Получение нарушений из базы данных"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        query = "SELECT * FROM violations WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        if video_id:
            query += " AND video_id = %s"
            params.append(video_id)
        
        query += " ORDER BY date, time"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        violations = []
        for row in rows:
            violations.append({
                "id": row.get("id"),
                "time": row.get("time"),
                "type": row.get("type"),
                "description": row.get("description"),
                "source": row.get("source"),
                "date": row.get("date").strftime('%Y-%m-%d') if isinstance(row.get("date"), date) else str(row.get("date")),
                "video_id": row.get("video_id"),
                "video_url": row.get("video_url"),
                "breed": row.get("breed"),
                "muzzle": bool(row.get("muzzle")) if row.get("muzzle") is not None else None,
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None
            })
        
        return {"success": True, "violations": violations, "count": len(violations)}
    except Error as e:
        print(f"Error getting violations from DB: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


@app.get("/api/v1/violations/db/{video_id}")
async def get_video_violations_from_db(video_id: str):
    """Получение нарушений для конкретного видео из базы данных"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM violations 
            WHERE video_id = %s 
            ORDER BY date, time
        """, (video_id,))
        
        rows = cursor.fetchall()
        
        violations = []
        for row in rows:
            violations.append({
                "id": row.get("id"),
                "time": row.get("time"),
                "type": row.get("type"),
                "description": row.get("description"),
                "source": row.get("source"),
                "date": row.get("date").strftime('%Y-%m-%d') if isinstance(row.get("date"), date) else str(row.get("date")),
                "video_id": row.get("video_id"),
                "video_url": row.get("video_url"),
                "breed": row.get("breed"),
                "muzzle": bool(row.get("muzzle")) if row.get("muzzle") is not None else None
            })
        
        return {"success": True, "violations": violations, "count": len(violations)}
    except Error as e:
        print(f"Error getting violations from DB: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


@app.get("/api/v1/violations/{video_id}", response_model=ProcessingResponse)
async def get_video_violations(video_id: str):
    """Получение нарушений для конкретного видео"""
    if video_id not in results_storage:
        raise HTTPException(status_code=404, detail="Видео не найдено")
    
    data = results_storage[video_id]
    violations = [ViolationResponse(**v) for v in data.get("violations", [])]
    
    return ProcessingResponse(
        success=True,
        video_id=video_id,
        violations=violations,
        processing_time=data.get("processing_time"),
        message=f"Найдено нарушений: {len(violations)}"
    )




@app.get("/api/v1/progress/{video_id}")
async def get_progress(video_id: str):
    """Получение прогресса обработки видео"""
    # Логируем запрос прогресса
    with progress_lock:
        available_ids = list(progress_storage.keys())
        has_progress = video_id in progress_storage
    
    import sys
    sys.stdout.write(f"[API] Запрос прогресса для {video_id}, доступные: {available_ids}\n")
    sys.stdout.flush()
    
    if not has_progress:
        # Если прогресс не найден, но есть результат - обработка завершена
        if video_id in results_storage:
            return {
                "percent": 100,
                "status": "Завершено",
                "completed": True
            }
        # Если прогресс еще не инициализирован, возвращаем начальное состояние
        return {
            "percent": 0,
            "current_frame": 0,
            "total_frames": 0,
            "status": "Ожидание начала обработки...",
            "completed": False
        }
    
    with progress_lock:
        progress = progress_storage.get(video_id, {})
    
    response_data = {
        "percent": progress.get("percent", 0),
        "current_frame": progress.get("current_frame", 0),
        "total_frames": progress.get("total_frames", 0),
        "status": progress.get("status", "Обработка..."),
        "completed": progress.get("percent", 0) >= 100
    }
    import sys
    sys.stdout.write(f"[API] Возвращаем прогресс для {video_id}: {response_data}\n")
    sys.stdout.flush()
    return response_data


@app.delete("/api/v1/video/{video_id}")
async def delete_video(video_id: str):
    """Удаление видео и результатов"""
    if video_id not in results_storage:
        raise HTTPException(status_code=404, detail="Видео не найдено")
    
    # Удаляем файл
    output_path = Path(results_storage[video_id]["output_path"])
    if output_path.exists():
        try:
            output_path.unlink()
        except:
            pass
    
    # Удаляем из хранилища
    del results_storage[video_id]
    
    return {"success": True, "message": "Видео удалено"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

