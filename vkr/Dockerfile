# Используем официальный образ PHP с FPM
FROM php:8.2-fpm

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libpng-dev \
    libonig-dev \
    libxml2-dev \
    zip \
    unzip \
    libzip-dev \
    && docker-php-ext-install pdo_mysql mbstring exif pcntl bcmath gd zip

# Настройка PHP для загрузки больших файлов
# Создаем файл конфигурации с правильными настройками
RUN echo "upload_max_filesize = 500M" > /usr/local/etc/php/conf.d/uploads.ini && \
    echo "post_max_size = 500M" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "max_execution_time = 600" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "max_input_time = 600" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "memory_limit = 512M" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "max_input_vars = 3000" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "display_errors = Off" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "display_startup_errors = Off" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "log_errors = On" >> /usr/local/etc/php/conf.d/uploads.ini && \
    echo "error_reporting = E_ALL & ~E_DEPRECATED & ~E_STRICT" >> /usr/local/etc/php/conf.d/uploads.ini && \
    cat /usr/local/etc/php/conf.d/uploads.ini

# Также создаем скрипт для применения настроек при старте контейнера
RUN echo '#!/bin/sh' > /usr/local/bin/php-config.sh && \
    echo 'echo "upload_max_filesize = 500M" > /usr/local/etc/php/conf.d/uploads.ini' >> /usr/local/bin/php-config.sh && \
    echo 'echo "post_max_size = 500M" >> /usr/local/etc/php/conf.d/uploads.ini' >> /usr/local/bin/php-config.sh && \
    echo 'echo "max_execution_time = 600" >> /usr/local/etc/php/conf.d/uploads.ini' >> /usr/local/bin/php-config.sh && \
    echo 'echo "max_input_time = 600" >> /usr/local/etc/php/conf.d/uploads.ini' >> /usr/local/bin/php-config.sh && \
    echo 'echo "memory_limit = 512M" >> /usr/local/etc/php/conf.d/uploads.ini' >> /usr/local/bin/php-config.sh && \
    chmod +x /usr/local/bin/php-config.sh

# Установка Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Рабочая директория
WORKDIR /var/www

# Сначала копируем только composer файлы для кэширования слоя
COPY composer.json composer.lock ./

# Установка зависимостей Laravel (кэшируется если composer.json не изменился)
RUN composer install --optimize-autoloader --no-dev --no-scripts

# Копируем остальные файлы проекта
COPY . .

# Запускаем скрипты composer (post-install-cmd и т.д.)
RUN composer dump-autoload --optimize

# Настройка прав (Laravel требует запись в storage и bootstrap/cache)
RUN chown -R www-data:www-data /var/www/storage \
    && chown -R www-data:www-data /var/www/bootstrap/cache \
    && chmod -R 775 /var/www/storage \
    && chmod -R 775 /var/www/bootstrap/cache