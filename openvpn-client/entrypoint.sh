#!/bin/sh
set -e

OPENVPN_CONFIG=${OPENVPN_CONFIG:-/config/client.ovpn}

BACKEND_CONTAINER=${BACKEND_CONTAINER:-neurodog-backend}
BACKEND_PORT=${BACKEND_PORT:-8000}
BACKEND_VPN_IP=${BACKEND_VPN_IP:-}

MYSQL_CONTAINER=${MYSQL_CONTAINER:-neurodog-mysql}
MYSQL_PORT=${MYSQL_PORT:-3306}
MYSQL_VPN_IP=${MYSQL_VPN_IP:-}

echo "Starting OpenVPN client..."
openvpn --config "$OPENVPN_CONFIG" --daemon

echo "Waiting for tun0..."
for i in $(seq 1 30); do
  ip addr show tun0 >/dev/null 2>&1 && break
  sleep 1
done

if ! ip addr show tun0 >/dev/null 2>&1; then
  echo "ERROR: tun0 not created"
  exit 1
fi

TUN0_IP=$(ip addr show tun0 | awk '/inet / {print $2}' | cut -d/ -f1)
echo "VPN connected, tun0 IP: $TUN0_IP"

# -------------------------
# NAT Docker -> VPN
# -------------------------
iptables -t nat -A POSTROUTING -o tun0 -j MASQUERADE
iptables -A FORWARD -i eth0 -o tun0 -j ACCEPT
iptables -A FORWARD -i tun0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# TCP MSS fix (CRITICAL for MySQL)
iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu

# -------------------------
# Backend port forwarding
# -------------------------
if [ -n "$BACKEND_VPN_IP" ]; then
  BACKEND_IP=$(getent hosts "$BACKEND_CONTAINER" | awk '{print $1}')

  iptables -t nat -A PREROUTING -i tun0 -d "$BACKEND_VPN_IP" -p tcp --dport "$BACKEND_PORT" \
    -j DNAT --to-destination "$BACKEND_IP:$BACKEND_PORT"

  iptables -A FORWARD -d "$BACKEND_IP" -p tcp --dport "$BACKEND_PORT" -j ACCEPT

  iptables -t nat -A POSTROUTING -s "$BACKEND_IP" -p tcp --sport "$BACKEND_PORT" -o tun0 \
    -j SNAT --to-source "$BACKEND_VPN_IP"

  echo "Backend exposed: $BACKEND_VPN_IP:$BACKEND_PORT"
fi

# -------------------------
# MySQL port forwarding
# -------------------------
if [ -n "$MYSQL_VPN_IP" ]; then
  MYSQL_IP=$(getent hosts "$MYSQL_CONTAINER" | awk '{print $1}')

  iptables -t nat -A PREROUTING -i tun0 -d "$MYSQL_VPN_IP" -p tcp --dport "$MYSQL_PORT" \
    -j DNAT --to-destination "$MYSQL_IP:$MYSQL_PORT"

  iptables -A FORWARD -d "$MYSQL_IP" -p tcp --dport "$MYSQL_PORT" -j ACCEPT

  iptables -t nat -A POSTROUTING -s "$MYSQL_IP" -p tcp --sport "$MYSQL_PORT" -o tun0 \
    -j SNAT --to-source "$MYSQL_VPN_IP"

  echo "MySQL exposed: $MYSQL_VPN_IP:$MYSQL_PORT"
fi

echo "VPN gateway ready"
tail -f /dev/null
