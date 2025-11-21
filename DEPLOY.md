# Инструкция по развертыванию на бесплатном хостинге

## Варианты бесплатного хостинга

### 1. Railway.app (Рекомендуется) ⭐
- **Бесплатный план**: $5 кредитов в месяц
- **Поддержка Docker**: ✅ Отличная
- **Простота**: ⭐⭐⭐⭐⭐
- **База данных**: Встроенная MySQL
- **Ссылка**: https://railway.app

### 2. Render.com
- **Бесплатный план**: Ограниченный, но достаточный
- **Поддержка Docker**: ✅ Хорошая
- **Простота**: ⭐⭐⭐⭐
- **База данных**: Отдельный сервис
- **Ссылка**: https://render.com

### 3. Fly.io
- **Бесплатный план**: 3 shared-cpu-1x VMs
- **Поддержка Docker**: ✅ Отличная
- **Простота**: ⭐⭐⭐
- **База данных**: Отдельный сервис
- **Ссылка**: https://fly.io

---

## Вариант 1: Развертывание на Railway.app

### Шаг 1: Подготовка репозитория

1. Убедитесь, что ваш код загружен в GitHub/GitLab/Bitbucket
2. Проверьте, что все файлы на месте:
   - `Dockerfile` (для Python бэкенда)
   - `vkr/Dockerfile` (для Laravel фронтенда)
   - `docker-compose.yml`
   - Все необходимые файлы проекта

### Шаг 2: Регистрация на Railway

1. Перейдите на https://railway.app
2. Нажмите "Login" и войдите через GitHub
3. Подтвердите доступ к репозиторию

### Шаг 3: Создание проекта

1. Нажмите "New Project"
2. Выберите "Deploy from GitHub repo"
3. Выберите ваш репозиторий
4. Railway автоматически обнаружит `docker-compose.yml`

### Шаг 4: Настройка сервисов

Railway автоматически создаст сервисы из `docker-compose.yml`. Вам нужно:

#### 4.1. Настроить Python Backend

1. Откройте сервис `python-backend`
2. В разделе "Settings" → "Variables" добавьте:
   ```
   PYTHONUNBUFFERED=1
   ```
3. В разделе "Settings" → "Deploy" убедитесь, что:
   - Build Command: (пусто, используется Dockerfile)
   - Start Command: (пусто, используется CMD из Dockerfile)

#### 4.2. Настроить MySQL

1. Railway автоматически создаст MySQL сервис
2. В разделе "Variables" будут автоматически созданы:
   - `MYSQLDATABASE`
   - `MYSQLUSER`
   - `MYSQLPASSWORD`
   - `MYSQLHOST`
   - `MYSQLPORT`

#### 4.3. Настроить Laravel Frontend

1. Откройте сервис `laravel-app`
2. В разделе "Settings" → "Variables" добавьте:
   ```
   DB_HOST=${{MySQL.MYSQLHOST}}
   DB_DATABASE=${{MySQL.MYSQLDATABASE}}
   DB_USERNAME=${{MySQL.MYSQLUSER}}
   DB_PASSWORD=${{MySQL.MYSQLPASSWORD}}
   DB_PORT=${{MySQL.MYSQLPORT}}
   PYTHON_API_URL=${{python-backend.RAILWAY_PUBLIC_DOMAIN}}
   PHP_UPLOAD_MAX_FILESIZE=500M
   PHP_POST_MAX_SIZE=500M
   PHP_MAX_EXECUTION_TIME=600
   PHP_MEMORY_LIMIT=512M
   APP_ENV=production
   APP_DEBUG=false
   APP_KEY=base64:YOUR_APP_KEY_HERE
   ```
   **Важно**: Сгенерируйте `APP_KEY` локально:
   ```bash
   cd vkr
   php artisan key:generate --show
   ```
   Скопируйте ключ и вставьте в переменную `APP_KEY`

#### 4.4. Настроить Nginx

1. Railway может не поддерживать несколько сервисов на одном порту
2. **Альтернатива**: Используйте встроенный веб-сервер Laravel или настройте Nginx как отдельный сервис

### Шаг 5: Настройка домена

1. В каждом сервисе перейдите в "Settings" → "Networking"
2. Нажмите "Generate Domain" для получения публичного URL
3. Для Laravel: используйте этот URL как `APP_URL` в переменных окружения

### Шаг 6: Запуск миграций

После первого деплоя выполните миграции:

1. Откройте сервис `laravel-app`
2. Перейдите в "Deployments" → выберите последний деплой
3. Откройте "View Logs"
4. Или используйте Railway CLI:
   ```bash
   railway run --service laravel-app php artisan migrate
   ```

### Шаг 7: Обновление PYTHON_API_URL

В Laravel сервисе обновите переменную:
```
PYTHON_API_URL=https://your-python-backend.railway.app
```
(Используйте публичный домен Python бэкенда)

---

## Вариант 2: Развертывание на Render.com

### Шаг 1: Подготовка

1. Зарегистрируйтесь на https://render.com
2. Подключите ваш GitHub репозиторий

### Шаг 2: Создание сервисов

#### 2.1. Python Backend

1. "New" → "Web Service"
2. Выберите репозиторий
3. Настройки:
   - **Name**: `dogsneuro-backend`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `Dockerfile`
   - **Docker Context**: `.`
   - **Start Command**: (пусто)
4. Добавьте переменные окружения:
   ```
   PYTHONUNBUFFERED=1
   ```

#### 2.2. MySQL Database

1. "New" → "PostgreSQL" (или MySQL если доступен)
2. Настройки:
   - **Name**: `dogsneuro-db`
   - **Database**: `laravel`
   - **User**: (автоматически)
   - **Password**: (автоматически)
3. Сохраните credentials

#### 2.3. Laravel Frontend

1. "New" → "Web Service"
2. Выберите репозиторий
3. Настройки:
   - **Name**: `dogsneuro-frontend`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `vkr/Dockerfile`
   - **Docker Context**: `vkr`
   - **Start Command**: (пусто)
4. Добавьте переменные окружения:
   ```
   DB_HOST=<из MySQL сервиса>
   DB_DATABASE=laravel
   DB_USERNAME=<из MySQL сервиса>
   DB_PASSWORD=<из MySQL сервиса>
   DB_PORT=3306
   PYTHON_API_URL=https://your-backend.onrender.com
   APP_ENV=production
   APP_DEBUG=false
   APP_KEY=<сгенерируйте локально>
   ```

---

## Вариант 3: Развертывание на Fly.io

### Шаг 1: Установка Fly CLI

```bash
# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex

# Mac/Linux
curl -L https://fly.io/install.sh | sh
```

### Шаг 2: Регистрация

```bash
fly auth signup
```

### Шаг 3: Создание приложений

#### 3.1. Python Backend

```bash
cd /path/to/your/project
fly launch --name dogsneuro-backend --dockerfile Dockerfile
```

#### 3.2. Laravel Frontend

```bash
cd vkr
fly launch --name dogsneuro-frontend --dockerfile Dockerfile
```

#### 3.3. MySQL

```bash
fly postgres create --name dogsneuro-db
```

### Шаг 4: Настройка переменных окружения

```bash
# Для Python Backend
fly secrets set -a dogsneuro-backend PYTHONUNBUFFERED=1

# Для Laravel Frontend
fly secrets set -a dogsneuro-frontend \
  DB_HOST=<из postgres> \
  DB_DATABASE=laravel \
  DB_USERNAME=<из postgres> \
  DB_PASSWORD=<из postgres> \
  PYTHON_API_URL=https://dogsneuro-backend.fly.dev
```

---

## Общие рекомендации

### 1. Оптимизация Docker образов

Создайте `.dockerignore` файлы для уменьшения размера:

**Корневой `.dockerignore`**:
```
__pycache__/
*.pyc
.git/
.vscode/
output/
*.mp4
*.avi
*.mov
.venv/
venv/
.env
```

**`vkr/.dockerignore`**:
```
node_modules/
vendor/
.git/
.env
storage/
bootstrap/cache/
```

### 2. Переменные окружения

Создайте файл `.env.example` для документации:

**`.env.example`**:
```env
# Python Backend
PYTHONUNBUFFERED=1

# Laravel
DB_HOST=mysql
DB_DATABASE=laravel
DB_USERNAME=root
DB_PASSWORD=secret
PYTHON_API_URL=http://python-backend:8000
APP_ENV=production
APP_DEBUG=false
APP_KEY=
```

### 3. Ограничения бесплатных планов

- **Railway**: 500 часов в месяц, $5 кредитов
- **Render**: Медленный старт после бездействия (spin down)
- **Fly.io**: 3 shared VMs, ограниченный трафик

### 4. Мониторинг

Используйте встроенные логи платформ для отслеживания ошибок.

### 5. Резервное копирование

Настройте автоматическое резервное копирование базы данных через встроенные инструменты платформ.

---

## Решение проблем

### Проблема: Сервисы не могут найти друг друга

**Решение**: Используйте внутренние домены платформы:
- Railway: `${{service-name.RAILWAY_PRIVATE_DOMAIN}}`
- Render: Используйте внутренние DNS имена
- Fly.io: Используйте `.internal` домены

### Проблема: Ошибки миграций

**Решение**: Выполните миграции вручную через CLI или веб-интерфейс платформы.

### Проблема: Большие файлы не загружаются

**Решение**: Увеличьте лимиты в настройках платформы или используйте внешнее хранилище (S3, Cloudflare R2).

---

## Полезные ссылки

- [Railway Documentation](https://docs.railway.app)
- [Render Documentation](https://render.com/docs)
- [Fly.io Documentation](https://fly.io/docs)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

