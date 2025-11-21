<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчет о нарушениях</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .hero-section {
            background-color: #3498db;
            color: white;
            padding: 3rem 0;
            margin-bottom: 2rem;
        }
        .video-btn {
            padding: 5px 10px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
 
    
    <!-- Шапка -->
    <header class="hero-section text-center">
        <div class="container">
            <h1 class="display-4">Система контроля содержания собак</h1>
        </div>
    </header>
    
    <div class="container mt-5">
       <h1 class="text-center mb-4">{{ $reportTitle }}</h1>
        
        <div class="d-grid mb-4">
            <a href="{{ route('export.excel') }}" class="btn btn-success btn-lg">
                Выгрузить отчет в Excel
            </a>
        </div>

        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Дата нарушения</th> <!-- Изменил название колонки -->
                    <th>Тип нарушения</th>
                    <th>Таймкод</th>
                    <th>Описание</th>
                    <th>Порода</th>
                    <th>Наличие намордника</th>
                    <th>Видео</th>
                </tr>
            </thead>
            <tbody>
               @foreach($violations as $violation)
            <tr>
                <td>{{ $violation['date'] }}</td> <!-- Теперь берем дату из нарушения -->
                    <td>{{ $violation['type'] }}</td>
                    <td>{{ $violation['time'] }}</td>
                    <td>{{ $violation['description'] }}</td>
                    <td>{{ $violation['breed'] }}</td>
                    <td>{{ $violation['muzzle'] ? 'Да' : 'Нет' }}</td>
                    <td>
                        @if(isset($violation['video_url']))
                        <a href="{{ $violation['video_url'] }}" 
                           class="btn btn-primary btn-sm video-btn" 
                           download="нарушение_{{ $violation['time'] }}.mp4">
                            <i class="bi bi-download"></i> Скачать
                        </a>
                        @else
                        <span class="text-muted">Нет видео</span>
                        @endif
                    </td>
                </tr>
                @endforeach
            </tbody>
        </table>
    </div>

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
</body>
</html>