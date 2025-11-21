<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>Система контроля содержания собак</title>
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .hero-section {
            background-color: #3498db;
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
        }
        .upload-container {
            max-width: 800px;
            margin: 0 auto;
        }
        .file-upload {
            border: 2px dashed #dee2e6;
            border-radius: 5px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .file-upload:hover {
            border-color: #3498db;
            background-color: #f8f9fa;
        }
        .file-upload input {
            display: none;
        }
        #progressContainer {
            display: none;
            margin-top: 15px;
        }
        #uploadStatus {
            margin-top: 5px;
            font-size: 0.9rem;
        }
        #resultsContainer {
            display: none;
            margin-top: 30px;
        }
        .violation-item {
            border-left: 4px solid #dc3545;
            padding: 10px 15px;
            margin-bottom: 10px;
            background-color: #f8f9fa;
        }
        .violation-time {
            font-weight: bold;
            color: #dc3545;
        }
        .violation-source {
            font-size: 0.8rem;
            color: #6c757d;
            margin-top: 5px;
        }
        .file-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
            padding: 5px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .file-item-name {
            flex-grow: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .file-item-remove {
            color: #dc3545;
            cursor: pointer;
            margin-left: 10px;
        }
        #selectedFiles {
            max-height: 200px;
            overflow-y: auto;
            margin-bottom: 15px;
        }
        .report-period {
            margin-top: 30px;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <!-- Шапка -->
    <header class="hero-section text-center">
        <div class="container">
            <h1 class="display-4">Система контроля содержания собак</h1>
            <p class="lead">Загрузите видео с нарушением правил выгула собак</p>
        </div>
    </header>

    <!-- Основной контент -->
    <main class="container mb-5">
        <div class="upload-container">
            <div class="card shadow">
                <div class="card-header bg-primary text-white">
                    <h2 class="h5 mb-0">Форма загрузки видео</h2>
                </div>
                <div class="card-body">
                    <form id="videoUploadForm">
                        <!-- Поле загрузки видео -->
                        <div class="mb-3">
                            <label class="form-label">Видео нарушения:</label>
                            <div class="file-upload mb-2" id="fileUploadArea">
                                <input type="file" id="videoFile" name="videoFile" accept="video/*" multiple required>
                                <label for="videoFile" class="d-block">
                                    <i class="bi bi-cloud-arrow-up fs-1"></i>
                                    <p class="mt-2">Нажмите для загрузки видео или перетащите файлы сюда</p>
                                    <p class="small">Можно выбрать несколько файлов</p>
                                </label>
                            </div>
                            
                            <div id="selectedFiles">
                                <!-- Сюда будут добавляться выбранные файлы -->
                            </div>
                            
                            <div class="form-text">Допустимые форматы: MP4, AVI, MOV</div>
                            
                            <!-- Прогресс-бар -->
                            <div id="progressContainer" class="mt-3">
                                <div class="progress">
                                    <div id="uploadProgress" class="progress-bar progress-bar-striped progress-bar-animated" 
                                         role="progressbar" style="width: 0%"></div>
                                </div>
                                <div id="uploadStatus" class="text-center small mt-1">0%</div>
                            </div>
                        </div>

                        <!-- Кнопка отправки -->
                        <div class="d-grid">
                            <button type="submit" class="btn btn-success btn-lg" id="submitBtn">
                                Отправить видео
                            </button>
                        </div>
                    </form>

                    <!-- Блок результатов -->
                    <div id="resultsContainer" class="mt-4">
                        <h4 class="mb-3">Результаты анализа:</h4>
                        
                        <!-- Сообщение о результатах -->
                        <div id="analysisResult" class="alert mb-3" style="display: none;">
                            <!-- Будет заполнено через JavaScript -->
                        </div>
                        
                        <!-- Кнопка скачивания обработанного видео -->
                        <div id="downloadVideoContainer" class="mb-4" style="display: none;">
                            <a id="downloadVideoBtn" href="#" class="btn btn-primary btn-lg" download>
                                <i class="bi bi-download"></i> Скачать обработанное видео
                            </a>
                        </div>
                        
                        <!-- Список нарушений -->
                        <div id="violationsList">
                            <!-- Сюда будут добавляться нарушения -->
                        </div>
                    </div>
                    
                    <!-- Блок формирования отчета -->
<div class="report-period mt-5">
    <h4 class="mb-3">Формирование отчета за период</h4>
    <form id="reportForm" action="{{ route('report') }}" method="POST">
        @csrf
        <div class="row">
            <div class="col-md-6 mb-3">
                <label for="startDate" class="form-label">Дата начала</label>
                <input type="date" class="form-control" id="startDate" name="start_date" required>
            </div>
            <div class="col-md-6 mb-3">
                <label for="endDate" class="form-label">Дата окончания</label>
                <input type="date" class="form-control" id="endDate" name="end_date" required>
            </div>
        </div>
        <div class="d-grid">
            <button type="submit" class="btn btn-primary">
                Сформировать отчет за период
            </button>
        </div>
    </form>
</div>
                </div>
            </div>
        </div>
    </main>

    <!-- Подвал -->
    <footer class="bg-dark text-white py-3">
        <div class="container text-center">
            <p class="mb-0">&copy; 2025 Система контроля содержания собак в общественных местах</p>
        </div>
    </footer>

    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('videoFile');
            const selectedFilesContainer = document.getElementById('selectedFiles');
            const fileUploadArea = document.getElementById('fileUploadArea');
            const form = document.getElementById('videoUploadForm');
            const progressContainer = document.getElementById('progressContainer');
            const uploadProgress = document.getElementById('uploadProgress');
            const uploadStatus = document.getElementById('uploadStatus');
            const submitBtn = document.getElementById('submitBtn');
            const resultsContainer = document.getElementById('resultsContainer');
            const violationsList = document.getElementById('violationsList');
            const reportForm = document.getElementById('reportForm');
            const analysisResult = document.getElementById('analysisResult');
            const downloadVideoContainer = document.getElementById('downloadVideoContainer');
            const downloadVideoBtn = document.getElementById('downloadVideoBtn');
            
            let filesToUpload = [];

            // Обработка выбора файлов
            fileInput.addEventListener('change', function() {
                updateSelectedFilesList(this.files);
            });

            // Обновление списка выбранных файлов
            function updateSelectedFilesList(files) {
                selectedFilesContainer.innerHTML = '';
                filesToUpload = Array.from(files);
                
                if (filesToUpload.length === 0) {
                    selectedFilesContainer.innerHTML = '<div class="text-muted small">Файлы не выбраны</div>';
                    return;
                }
                
                filesToUpload.forEach((file, index) => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.innerHTML = `
                        <span class="file-item-name">${file.name}</span>
                        <span class="file-item-remove" data-index="${index}">
                            <i class="bi bi-x-circle"></i>
                        </span>
                    `;
                    selectedFilesContainer.appendChild(fileItem);
                });
                
                // Добавляем обработчики для кнопок удаления
                document.querySelectorAll('.file-item-remove').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const index = parseInt(this.getAttribute('data-index'));
                        filesToUpload.splice(index, 1);
                        
                        // Обновляем input files
                        const dataTransfer = new DataTransfer();
                        filesToUpload.forEach(file => dataTransfer.items.add(file));
                        fileInput.files = dataTransfer.files;
                        
                        updateSelectedFilesList(fileInput.files);
                    });
                });
            }

            // Обработка drag and drop
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                fileUploadArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            ['dragenter', 'dragover'].forEach(eventName => {
                fileUploadArea.addEventListener(eventName, highlight, false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                fileUploadArea.addEventListener(eventName, unhighlight, false);
            });

            function highlight() {
                fileUploadArea.classList.add('border-primary');
                fileUploadArea.style.backgroundColor = '#f8f9fa';
            }

            function unhighlight() {
                fileUploadArea.classList.remove('border-primary');
                fileUploadArea.style.backgroundColor = '';
            }

            fileUploadArea.addEventListener('drop', handleDrop, false);

            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                // Обновляем input files
                const dataTransfer = new DataTransfer();
                for (let i = 0; i < files.length; i++) {
                    dataTransfer.items.add(files[i]);
                }
                fileInput.files = dataTransfer.files;
                
                updateSelectedFilesList(fileInput.files);
            }

            // Обработка формы загрузки видео
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                if (filesToUpload.length === 0) {
                    alert('Пожалуйста, выберите хотя бы один видео файл');
                    return;
                }
                
                // Показываем прогресс-бар
                progressContainer.style.display = 'block';
                submitBtn.disabled = true;
                resultsContainer.style.display = 'none';
                violationsList.innerHTML = '';
                if (analysisResult) {
                    analysisResult.style.display = 'none';
                }
                if (downloadVideoContainer) {
                    downloadVideoContainer.style.display = 'none';
                }
                
                // Инициализируем прогресс на 0%
                updateProgress(0);
                uploadStatus.textContent = '0% - Загрузка видео...';
                
                // Загружаем и обрабатываем видео
                uploadAndProcessVideo();
            });

            // Функция для загрузки и обработки видео через API
            async function uploadAndProcessVideo() {
                const formData = new FormData();
                
                // Добавляем все файлы
                filesToUpload.forEach((file, index) => {
                    formData.append('videoFile', file);
                });
                
                let videoId = null;
                
                try {
                    // Отправляем первый файл (можно расширить для множественной загрузки)
                    const response = await fetch('/api/video/upload', {
                        method: 'POST',
                        body: formData,
                        headers: {
                            'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '',
                            'Accept': 'application/json'
                        }
                    });
                    
                    // Получаем текст ответа
                    const text = await response.text();
                    
                    // Пытаемся найти JSON в ответе (на случай если есть HTML warnings перед ним)
                    let jsonText = text;
                    const jsonMatch = text.match(/\{[\s\S]*\}/);
                    if (jsonMatch) {
                        jsonText = jsonMatch[0];
                    }
                    
                    let result;
                    try {
                        result = JSON.parse(jsonText);
                    } catch (e) {
                        throw new Error('Не удалось распарсить ответ сервера.');
                    }
                    
                    if (!response.ok) {
                        throw new Error(result.message || 'Ошибка обработки видео');
                    }
                    
                    if (result.success) {
                        videoId = result.data.video_id;
                        
                        // Начинаем отслеживание прогресса
                        await trackProgress(videoId);
                        
                        // Ждем немного, чтобы обработка точно завершилась
                        await new Promise(resolve => setTimeout(resolve, 1000));
                        
                        // Получаем финальные результаты через Laravel API
                        let finalData = { violations: [], processing_time: 'N/A' };
                        try {
                            const finalResponse = await fetch(`/api/violations/${encodeURIComponent(videoId)}`);
                            if (finalResponse.ok) {
                                const responseData = await finalResponse.json();
                                if (responseData.success && responseData.data) {
                                    finalData = responseData.data;
                                }
                            }
                        } catch (e) {
                            // Используем данные из начального ответа, если не удалось получить финальные
                        }
                        
                        updateProgress(100);
                        
                        // Отображаем результаты
                        displayResults({
                            success: true,
                            violations: finalData.violations || result.data.violations || [],
                            processingTime: finalData.processing_time || result.data.processing_time || 'N/A',
                            videoId: videoId
                        });
                    } else {
                        throw new Error(result.message || 'Ошибка обработки видео');
                    }
                } catch (error) {
                    let errorMessage = 'Ошибка при обработке видео: ';
                    if (error.message) {
                        errorMessage += error.message;
                    } else {
                        errorMessage += 'Неизвестная ошибка';
                    }
                    alert(errorMessage);
                    updateProgress(0);
                } finally {
                    form.reset();
                    filesToUpload = [];
                    updateSelectedFilesList([]);
                    progressContainer.style.display = 'none';
                    submitBtn.disabled = false;
                }
            }
            
            // Функция для отслеживания прогресса обработки
            async function trackProgress(videoId) {
                const maxAttempts = 600; // Максимум 5 минут (600 * 0.5 сек)
                let attempts = 0;
                
                return new Promise((resolve, reject) => {
                    const checkProgress = async () => {
                        try {
                            const response = await fetch(`/api/video/progress/${videoId}`);
                            if (!response.ok) {
                                // Если прогресс не найден, возможно обработка уже завершена
                                resolve();
                                return;
                            }
                            
                            const progressData = await response.json();
                            
                            // Обновляем прогресс-бар с полным статусом
                            const percent = progressData.percent || 0;
                            const statusText = progressData.status || 'Обработка...';
                            const frameInfo = progressData.total_frames > 0 
                                ? ` (${progressData.current_frame}/${progressData.total_frames} кадров)`
                                : '';
                            const fullStatusText = `${Math.round(percent)}% - ${statusText}${frameInfo}`;
                            
                            updateProgress(percent, fullStatusText);
                            
                            // Если обработка завершена
                            if (progressData.completed || percent >= 100) {
                                updateProgress(100);
                                resolve();
                                return;
                            }
                            
                            // Продолжаем проверку
                            attempts++;
                            if (attempts >= maxAttempts) {
                                resolve();
                                return;
                            }
                            
                            // Проверяем снова через 500мс
                            setTimeout(checkProgress, 500);
                        } catch (error) {
                            // Продолжаем попытки
                            attempts++;
                            if (attempts < maxAttempts) {
                                setTimeout(checkProgress, 1000);
                            } else {
                                resolve(); // Разрешаем промис даже при ошибках
                            }
                        }
                    };
                    
                    // Начинаем проверку прогресса
                    checkProgress();
                });
            }
            
            // Функция для симуляции прогресса (пока видео обрабатывается)
            function simulateProgress() {
                let progress = 0;
                const interval = setInterval(() => {
                    progress += Math.random() * 5;
                    if (progress >= 90) {
                        progress = 90; // Останавливаем на 90%, ждем реального ответа
                        clearInterval(interval);
                    }
                    updateProgress(progress);
                }, 500);
                
                return interval;
            }

            // Обновление прогресс-бара
            function updateProgress(percent, statusText = null) {
                uploadProgress.style.width = percent + '%';
                
                // Обновляем статус только если не передан явный текст
                if (statusText === null) {
                    uploadStatus.textContent = Math.round(percent) + '%';
                } else {
                    uploadStatus.textContent = statusText;
                }
                
                if (percent >= 100) {
                    uploadProgress.classList.remove('progress-bar-animated');
                    uploadProgress.classList.remove('progress-bar-striped');
                }
            }

            // Отображение результатов
            function displayResults(data) {
                resultsContainer.style.display = 'block';
                
                const violations = data.violations || [];
                const violationsCount = violations.length;
                
                // Отображаем сообщение о результатах анализа
                if (analysisResult) {
                    if (violationsCount > 0) {
                        analysisResult.className = 'alert alert-danger mb-3';
                        analysisResult.innerHTML = `
                            <h5 class="alert-heading">⚠️ Обнаружены нарушения!</h5>
                            <p class="mb-0">В ходе анализа видео обнаружено <strong>${violationsCount}</strong> нарушений правил выгула собак.</p>
                        `;
                    } else {
                        analysisResult.className = 'alert alert-success mb-3';
                        analysisResult.innerHTML = `
                            <h5 class="alert-heading">✅ Нарушений не обнаружено</h5>
                            <p class="mb-0">Видео проанализировано. Нарушений правил выгула собак не выявлено.</p>
                        `;
                    }
                    analysisResult.style.display = 'block';
                }
                
                // Отображаем кнопку скачивания обработанного видео
                if (data.videoId && downloadVideoContainer && downloadVideoBtn) {
                    const videoUrl = `/api/video/${encodeURIComponent(data.videoId)}`;
                    downloadVideoBtn.href = videoUrl;
                    downloadVideoBtn.download = `processed_${data.videoId}.mp4`;
                    downloadVideoContainer.style.display = 'block';
                } else {
                    if (downloadVideoContainer) {
                        downloadVideoContainer.style.display = 'none';
                    }
                }
                
                // Отображаем список нарушений
                if (violationsCount > 0) {
                    violationsList.innerHTML = '<h5 class="mb-3">Детали нарушений:</h5>';
                    violations.forEach(violation => {
                        const violationElement = document.createElement('div');
                        violationElement.className = 'violation-item';
                        violationElement.innerHTML = `
                            <div class="violation-time">Время: ${violation.time}</div>
                            <div class="violation-type"><strong>${violation.type}</strong></div>
                            <div class="violation-description">${violation.description}</div>
                            <div class="violation-source">Источник: ${violation.source} (дата: ${violation.date})</div>
                        `;
                        violationsList.appendChild(violationElement);
                    });
                } else {
                    violationsList.innerHTML = '';
                }
            }
            
            // Устанавливаем даты по умолчанию для отчета (текущий месяц)
            const today = new Date();
            const firstDay = new Date(today.getFullYear(), today.getMonth(), 1);
            
            document.getElementById('startDate').valueAsDate = firstDay;
            document.getElementById('endDate').valueAsDate = today;
        });
    </script>
</body>
</html>