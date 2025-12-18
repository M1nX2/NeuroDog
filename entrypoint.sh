#!/bin/sh
# Не прерываем выполнение при ошибках - бэкэнд должен запуститься даже если VPN недоступен
set +e

VPN_GATEWAY=${VPN_GATEWAY:-openvpn-client}
VPN_ROUTES="${VPN_ROUTES:-}"
# Используем переменные окружения из docker-compose напрямую
# Значения по умолчанию задаются в docker-compose.yml
DB_HOST="${DB_HOST}"
MYSQL_HOST="${MYSQL_HOST}"

# Проверяем, нужно ли настраивать VPN маршруты
# Если VPN_ROUTES задан или DB_HOST/MYSQL_HOST содержат IP из VPN подсетей
NEED_VPN_ROUTES=false

# Если VPN_ROUTES задан явно, используем его
if [ -n "$VPN_ROUTES" ]; then
  NEED_VPN_ROUTES=true
else
  # Иначе проверяем, содержат ли адреса VPN IP (извлекаем подсети из VPN_ROUTES или используем дефолтные)
  # Извлекаем подсети из VPN_ROUTES для проверки
  VPN_SUBNETS=$(echo "$VPN_ROUTES" | tr ',' '\n' | sed 's|/.*||' | sed 's|\.[0-9]*$|\.|' | sort -u | tr '\n' '|' | sed 's/|$//')
  
  if [ -z "$VPN_SUBNETS" ]; then
    # Если VPN_ROUTES не задан, используем дефолтные паттерны для обратной совместимости
    VPN_SUBNETS="10\.0\.70\.|10\.0\.60\."
  fi
  
  if [ -n "$DB_HOST" ] && echo "$DB_HOST" | grep -qE "^($VPN_SUBNETS)"; then
    NEED_VPN_ROUTES=true
  fi
  
  if [ -n "$MYSQL_HOST" ] && echo "$MYSQL_HOST" | grep -qE "^($VPN_SUBNETS)"; then
    NEED_VPN_ROUTES=true
  fi
fi

if [ "$NEED_VPN_ROUTES" = "true" ]; then
  # Резолвим IP шлюза через DNS
  echo "VPN IP addresses detected, checking VPN gateway availability..."
  VPN_GATEWAY_IP=$(getent hosts $VPN_GATEWAY 2>/dev/null | awk '{ print $1 }')

  if [ -z "$VPN_GATEWAY_IP" ]; then
    echo "Warning: VPN gateway ($VPN_GATEWAY) is not available."
    echo "This might be because ./config/client.ovpn is missing or VPN client is not running."
    echo "Skipping VPN routes. Backend will work without VPN."
  else
    # Добавляем маршруты для VPN-сетей через OpenVPN gateway
    echo "Configuring VPN routes through $VPN_GATEWAY ($VPN_GATEWAY_IP)..."
    
    # Получаем маршруты из переменной окружения VPN_ROUTES
    VPN_ROUTES="${VPN_ROUTES:-}"
    
    if [ -n "$VPN_ROUTES" ]; then
      # Разбиваем строку по запятым и добавляем каждый маршрут
      echo "$VPN_ROUTES" | tr ',' '\n' | while read -r route; do
        route=$(echo "$route" | xargs)  # Убираем пробелы
        if [ -n "$route" ]; then
          echo "Adding route: $route via $VPN_GATEWAY_IP"
          ip route add "$route" via "$VPN_GATEWAY_IP" 2>/dev/null || echo "Route $route already exists or failed"
        fi
      done
    else
      echo "Warning: VPN_ROUTES not set, skipping route configuration"
    fi
    
    echo "Current routes:"
    ip route
    
    # Проверяем VPN интерфейс
    echo "Checking VPN interface..."
    if ip addr show tun0 >/dev/null 2>&1; then
      echo "VPN interface (tun0) is available"
      ip addr show tun0 | head -3
    else
      echo "Warning: VPN interface (tun0) is not available in this container"
      echo "Note: VPN interface is in openvpn-client container, not here"
    fi
  fi
else
  echo "Local environment detected (no VPN IPs in MYSQL_HOST/DB_HOST), skipping VPN routes"
fi

# Показываем информацию о сетевых интерфейсах
echo "Network interfaces:"
ip addr show | grep -E "^[0-9]+:|inet " | head -10

# Показываем, на каких адресах будет слушать бэкенд
echo "Backend will listen on 0.0.0.0:8000 (all interfaces)"

echo "Starting FastAPI backend..."
echo "Backend will listen on 0.0.0.0:8000 (all interfaces)"

# Показываем информацию о сетевых интерфейсах
echo "Network interfaces in container:"
ip addr show | grep -E "^[0-9]+:|inet " | head -10

# Показываем, на каких адресах будет доступен бэкенд
CONTAINER_IP=$(hostname -i | awk '{print $1}')
echo "Backend will be accessible on:"
echo "  - Localhost (from host): http://localhost:8000"
echo "  - Docker network: http://neurodog-backend:8000"
if [ -n "$CONTAINER_IP" ]; then
  echo "  - Container IP: http://$CONTAINER_IP:8000"
fi

# Проверяем VPN маршруты и доступность VPN IP
if [ -n "$VPN_ROUTES" ]; then
  echo "VPN routes configured: $VPN_ROUTES"
  
  # Пытаемся определить VPN IP адрес из маршрутов
  # Извлекаем первую подсеть и проверяем доступность
  FIRST_SUBNET=$(echo "$VPN_ROUTES" | cut -d',' -f1 | xargs)
  if [ -n "$FIRST_SUBNET" ]; then
    echo "Checking VPN connectivity for subnet: $FIRST_SUBNET"
    
    # Пытаемся пинговать первый IP в подсети (обычно это gateway)
    SUBNET_BASE=$(echo "$FIRST_SUBNET" | cut -d'/' -f1 | sed 's/\.[0-9]*$/.1/')
    if ping -c 1 -W 1 "$SUBNET_BASE" >/dev/null 2>&1; then
      echo "  ✓ VPN subnet $FIRST_SUBNET is reachable"
    else
      echo "  ✗ VPN subnet $FIRST_SUBNET is not reachable"
    fi
  fi
fi

# Важное предупреждение
echo ""
echo "⚠️  IMPORTANT: If backend should be accessible via VPN IP (e.g., 10.0.70.61:8000):"
echo "   1. Backend must be running on the remote server with that IP address"
echo "   2. OR configure port forwarding through VPN"
echo "   3. Current setup: Backend is accessible on localhost:8000 and Docker network only"
echo ""

# Инициализация базы данных (создание таблиц, если их нет)
if [ -f "/app/init_db.py" ]; then
  echo "Initializing database..."
  python /app/init_db.py || echo "Warning: Database initialization failed, continuing anyway..."
  echo ""
fi

# Запускаем FastAPI
exec python api.py

