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

# Если команда не передана, используем php-fpm по умолчанию
if [ $# -eq 0 ]; then
    set -- php-fpm
fi

# Выполняем команду
exec "$@"

