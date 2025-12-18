#!/usr/bin/env python3
"""
Скрипт для инициализации базы данных MySQL
Проверяет подключение и создаёт необходимые таблицы, если их нет
"""
import os
import sys
import time
import mysql.connector
from mysql.connector import Error

def get_db_config():
    """Получение конфигурации БД из переменных окружения"""
    return {
        'host': os.environ.get('MYSQL_HOST', 'neurodog-mysql'),
        'port': int(os.environ.get('MYSQL_PORT', 3306)),
        'database': os.environ.get('MYSQL_DATABASE', 'laravel'),
        'user': os.environ.get('MYSQL_USER', 'root'),
        'password': os.environ.get('MYSQL_PASSWORD', 'secret'),
    }

def wait_for_mysql(max_retries=30, delay=2):
    """Ожидание доступности MySQL"""
    config = get_db_config()
    print(f"Waiting for MySQL at {config['host']}:{config['port']}...")
    
    for i in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host=config['host'],
                port=config['port'],
                user=config['user'],
                password=config['password'],
                connection_timeout=5
            )
            conn.close()
            print(f"✓ MySQL is available")
            return True
        except Error as e:
            if i < max_retries - 1:
                print(f"  Attempt {i+1}/{max_retries}: MySQL not ready yet ({e}), waiting {delay}s...")
                time.sleep(delay)
            else:
                print(f"✗ MySQL is not available after {max_retries} attempts")
                return False
    
    return False

def check_table_exists(cursor, table_name):
    """Проверка существования таблицы"""
    cursor.execute("""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = %s
    """, (get_db_config()['database'], table_name))
    return cursor.fetchone()[0] > 0

def create_tables(cursor):
    """Создание необходимых таблиц, если их нет"""
    config = get_db_config()
    database = config['database']
    
    # Используем базу данных
    cursor.execute(f"USE `{database}`")
    
    # Таблица для хранения результатов обработки видео (если нужна)
    # Можно добавить другие таблицы по необходимости
    tables_created = []
    
    # Пример: таблица для хранения информации о видео
    if not check_table_exists(cursor, 'video_processing'):
        print("Creating table 'video_processing'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `video_processing` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `video_id` VARCHAR(255) NOT NULL UNIQUE,
                `filename` VARCHAR(255),
                `status` VARCHAR(50) DEFAULT 'pending',
                `progress` INT DEFAULT 0,
                `violations_count` INT DEFAULT 0,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                `completed_at` TIMESTAMP NULL,
                INDEX `idx_video_id` (`video_id`),
                INDEX `idx_status` (`status`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        tables_created.append('video_processing')
        print("  ✓ Table 'video_processing' created")
    
    # Создаем таблицу violations для фронтенда (Django)
    # Структура соответствует модели Violation в Django
    if not check_table_exists(cursor, 'violations'):
        print("Creating table 'violations'...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `violations` (
                `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
                `time` VARCHAR(20) NOT NULL,
                `type` VARCHAR(255) NOT NULL,
                `description` LONGTEXT NOT NULL,
                `source` VARCHAR(255) NOT NULL,
                `date` DATE NOT NULL,
                `video_id` VARCHAR(255) NULL,
                `video_url` VARCHAR(200) NULL,
                `breed` VARCHAR(255) NULL,
                `muzzle` TINYINT(1) NULL,
                `created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                `updated_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
                INDEX `violations_date_ti_idx` (`date`, `time`),
                INDEX `violations_video_i_idx` (`video_id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        tables_created.append('violations')
        print("  ✓ Table 'violations' created")
    else:
        # Проверяем наличие поля updated_at, если его нет - добавляем
        cursor.execute("""
            SELECT COUNT(*) 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = %s 
            AND TABLE_NAME = 'violations' 
            AND COLUMN_NAME = 'updated_at'
        """, (database,))
        has_updated_at = cursor.fetchone()[0] > 0
        
        if not has_updated_at:
            print("Adding missing 'updated_at' column to 'violations' table...")
            cursor.execute("""
                ALTER TABLE `violations` 
                ADD COLUMN `updated_at` DATETIME(6) NOT NULL 
                DEFAULT CURRENT_TIMESTAMP(6) 
                ON UPDATE CURRENT_TIMESTAMP(6)
            """)
            print("  ✓ Column 'updated_at' added to 'violations' table")
        else:
            print("  ✓ Table 'violations' exists with all required columns")
    
    if tables_created:
        print(f"\n✓ Created {len(tables_created)} table(s): {', '.join(tables_created)}")
    else:
        print("\n✓ All required tables already exist")
    
    return len(tables_created)

def main():
    """Основная функция"""
    print("=" * 60)
    print("MySQL Database Initialization")
    print("=" * 60)
    
    # Ожидаем доступности MySQL
    if not wait_for_mysql():
        print("\n⚠️  Warning: MySQL is not available. Skipping database initialization.")
        print("Backend will continue to run without database.")
        return 0
    
    config = get_db_config()
    print(f"\nConnecting to MySQL database '{config['database']}' at {config['host']}:{config['port']}...")
    
    try:
        # Подключаемся к MySQL (без указания базы данных сначала)
        conn = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            connection_timeout=10
        )
        
        cursor = conn.cursor()
        
        # Проверяем существование базы данных
        cursor.execute(f"SHOW DATABASES LIKE '{config['database']}'")
        if not cursor.fetchone():
            print(f"Creating database '{config['database']}'...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{config['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            print(f"  ✓ Database '{config['database']}' created")
        else:
            print(f"  ✓ Database '{config['database']}' exists")
        
        # Создаём таблицы
        create_tables(cursor)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("✓ Database initialization completed successfully")
        print("=" * 60)
        return 0
        
    except Error as e:
        print(f"\n✗ Error during database initialization: {e}")
        print("Backend will continue to run, but database features may be limited.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

