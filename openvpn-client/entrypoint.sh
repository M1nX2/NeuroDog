#!/bin/sh
# Не прерываем выполнение при ошибках - если конфига нет, просто выходим
set +e

OPENVPN_CONFIG=${OPENVPN_CONFIG:-/config/client.ovpn}
OPENVPN_ARGS=${OPENVPN_ARGS:-}

if [ ! -f "$OPENVPN_CONFIG" ]; then
  echo "OpenVPN config $OPENVPN_CONFIG not found. VPN client will not start."
  echo "Backend will work without VPN."
  exit 0
fi

echo "Starting OpenVPN with $OPENVPN_CONFIG"

# Запускаем OpenVPN в фоне
openvpn --config "$OPENVPN_CONFIG" $OPENVPN_ARGS --daemon

# Ждём, пока поднимется tun0
echo "Waiting for tun0 interface..."
for i in $(seq 1 30); do
  if ip addr show tun0 >/dev/null 2>&1; then
    echo "tun0 is up"
    break
  fi
  sleep 1
done

# Настраиваем NAT для пересылки трафика из Docker-сети в VPN
echo "Configuring NAT for VPN routing..."
iptables -t nat -A POSTROUTING -o tun0 -j MASQUERADE
iptables -A FORWARD -i eth0 -o tun0 -j ACCEPT
iptables -A FORWARD -i tun0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# Настраиваем проброс порта бэкенда через VPN, если указан BACKEND_VPN_IP
BACKEND_VPN_IP="${BACKEND_VPN_IP:-}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_CONTAINER="${BACKEND_CONTAINER:-neurodog-backend}"

if [ -n "$BACKEND_VPN_IP" ]; then
  echo "Configuring port forwarding for backend: $BACKEND_VPN_IP:$BACKEND_PORT -> $BACKEND_CONTAINER:$BACKEND_PORT"
  
  # Получаем IP адрес контейнера бэкенда
  BACKEND_CONTAINER_IP=$(getent hosts $BACKEND_CONTAINER 2>/dev/null | awk '{ print $1 }')
  
  if [ -z "$BACKEND_CONTAINER_IP" ]; then
    echo "Warning: Could not resolve $BACKEND_CONTAINER, skipping port forwarding"
  else
    echo "Backend container IP: $BACKEND_CONTAINER_IP"
    
    # Настраиваем DNAT для проброса порта через VPN интерфейс
    # Запросы к BACKEND_VPN_IP:PORT перенаправляются на BACKEND_CONTAINER_IP:PORT
    iptables -t nat -A PREROUTING -i tun0 -d "$BACKEND_VPN_IP" -p tcp --dport "$BACKEND_PORT" -j DNAT --to-destination "$BACKEND_CONTAINER_IP:$BACKEND_PORT"
    iptables -A FORWARD -i tun0 -d "$BACKEND_CONTAINER_IP" -p tcp --dport "$BACKEND_PORT" -j ACCEPT
    
    # Настраиваем SNAT для ответов, чтобы они возвращались через VPN интерфейс
    iptables -t nat -A POSTROUTING -s "$BACKEND_CONTAINER_IP" -p tcp --sport "$BACKEND_PORT" -o tun0 -j SNAT --to-source "$BACKEND_VPN_IP"
    iptables -A FORWARD -s "$BACKEND_CONTAINER_IP" -p tcp --sport "$BACKEND_PORT" -o tun0 -j ACCEPT
    
    echo "Port forwarding configured: $BACKEND_VPN_IP:$BACKEND_PORT -> $BACKEND_CONTAINER_IP:$BACKEND_PORT"
    echo "SNAT configured for responses from backend"
  fi
else
  echo "BACKEND_VPN_IP not set, skipping port forwarding"
fi

echo "VPN gateway is ready"

# Держим контейнер живым
tail -f /dev/null

