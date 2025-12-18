FROM alpine:3.20

RUN apk add --no-cache openvpn bash iproute2 iptables netcat-openbsd iputils-ping

WORKDIR /config

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

