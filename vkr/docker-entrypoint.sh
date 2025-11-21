#!/bin/sh

# Применяем настройки PHP для больших файлов при каждом запуске
{
    echo "upload_max_filesize = 500M"
    echo "post_max_size = 500M"
    echo "max_execution_time = 600"
    echo "max_input_time = 600"
    echo "memory_limit = 512M"
    echo "max_input_vars = 3000"
    echo "display_errors = Off"
    echo "display_startup_errors = Off"
    echo "log_errors = On"
} > /usr/local/etc/php/conf.d/uploads.ini

# Настройка PHP-FPM для прослушивания на TCP порту 9000
echo "Настройка PHP-FPM для прослушивания на порту 9000..."

# Обновляем все конфигурационные файлы PHP-FPM
for conf_file in /usr/local/etc/php-fpm.d/*.conf; do
    if [ -f "$conf_file" ]; then
        # Заменяем listen на TCP порт 9000
        sed -i 's/^listen = .*/listen = 9000/' "$conf_file" 2>/dev/null || true
        sed -i 's/^listen = \/run\/php\/php.*\.sock/listen = 9000/' "$conf_file" 2>/dev/null || true
        sed -i 's/^listen = 127\.0\.0\.1:9000/listen = 9000/' "$conf_file" 2>/dev/null || true
        
        # Если в секции [www] нет listen = 9000, добавляем
        if grep -q "^\[www\]" "$conf_file" && ! grep -q "^listen = 9000" "$conf_file"; then
            sed -i '/^\[www\]/a listen = 9000' "$conf_file" 2>/dev/null || \
            echo "listen = 9000" >> "$conf_file"
        fi
        
        # Настраиваем allowed_clients
        sed -i 's/^;listen.allowed_clients = .*/listen.allowed_clients = 0.0.0.0/' "$conf_file" 2>/dev/null || true
        sed -i 's/^listen.allowed_clients = .*/listen.allowed_clients = 0.0.0.0/' "$conf_file" 2>/dev/null || true
        if ! grep -q "^listen.allowed_clients" "$conf_file"; then
            echo "listen.allowed_clients = 0.0.0.0" >> "$conf_file"
        fi
    fi
done

# Выводим финальную конфигурацию для отладки
echo "Проверка конфигурации PHP-FPM:"
grep -h "^listen" /usr/local/etc/php-fpm.d/*.conf 2>/dev/null | head -5 || true

# Если команда не передана, используем php-fpm по умолчанию
if [ $# -eq 0 ]; then
    set -- php-fpm
fi

# Выполняем команду
exec "$@"

