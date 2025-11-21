# Инструкции по сборке и запуску

## Структура проекта

- **Python бэкенд** (FastAPI) - в корне проекта, обрабатывает видео через нейросети
- **Laravel фронтенд** - в папке `vkr/`, веб-интерфейс для загрузки видео и просмотра отчетов

## Быстрый старт

### Запуск всех сервисов через Docker Compose

```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

После запуска:
- **Python API (FastAPI)**: http://localhost:8000
- **Laravel фронтенд**: http://localhost:3000
- **API документация**: http://localhost:8000/docs

### Сборка отдельных сервисов

#### Python бэкенд

```bash
docker build -t dogsneuro-backend .
docker run -p 8000:8000 dogsneuro-backend
```

#### Laravel фронтенд

```bash
cd vkr
docker build -t dogsneuro-frontend .
```

## Решение проблем

### Ошибка с node_modules

Если возникает ошибка `archive/tar: unknown file mode` при сборке Laravel:

1. Убедитесь, что файл `vkr/.dockerignore` существует и содержит `node_modules`
2. Очистите контекст сборки:
   ```bash
   cd vkr
   rm -rf node_modules
   docker build -t dogsneuro-frontend .
   ```

### Проблемы с правами доступа

Для Laravel могут потребоваться дополнительные права:

```bash
cd vkr
sudo chown -R $USER:$USER storage bootstrap/cache
chmod -R 775 storage bootstrap/cache
```

## Переменные окружения

### Laravel

Создайте файл `vkr/.env` на основе `vkr/.env.example`:

```env
DB_HOST=mysql
DB_DATABASE=laravel
DB_USERNAME=root
DB_PASSWORD=secret

# URL Python API (в Docker используйте имя сервиса)
PYTHON_API_URL=http://python-backend:8000
```

### Python

Переменные окружения для Python бэкенда можно задать в `docker-compose.yml`.

## Миграции базы данных

После первого запуска Laravel выполните миграции:

```bash
docker-compose exec laravel-app php artisan migrate
```

## Использование

1. Откройте браузер и перейдите на http://localhost:3000
2. Загрузите видео через drag-and-drop или кнопку выбора файла
3. Дождитесь обработки видео (может занять время в зависимости от длины)
4. Просмотрите обнаруженные нарушения
5. Сформируйте отчет за период через форму внизу страницы

## API Endpoints

### Python API (FastAPI)

- `GET /` - Информация об API
- `GET /health` - Проверка здоровья сервиса
- `POST /api/v1/process-video` - Обработка видео
- `GET /api/v1/violations` - Список всех нарушений
- `GET /api/v1/violations/{video_id}` - Нарушения для конкретного видео
- `GET /api/v1/video/{video_id}` - Получение обработанного видео
- `GET /docs` - Swagger документация

### Laravel API

- `POST /api/video/upload` - Загрузка и обработка видео
- `GET /api/violations` - Получение списка нарушений
- `GET /api/video/{videoId}` - Получение обработанного видео

