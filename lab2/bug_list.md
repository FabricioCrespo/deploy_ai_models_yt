```
NoBrokersAvailable: NoBrokersAvailable
```
Not connected to the server.

```
running:  kcat -b localhost:9092 -t recitation-c -C -o earliest
%3|1725369171.734|FAIL|rdkafka#consumer-1| [thrd:localhost:9092/bootstrap]: localhost:9092/bootstrap: Connect to ipv6#[::1]:9092 failed: Connection refused (after 1ms in state CONNECT)
% ERROR: Failed to query metadata for topic recitation-c: Local: Broker transport failure
```
When you use kcat command, you still need to connect to server first by running: ` ssh -o ServerAliveInterval=60 -L 9092:localhost:9092 tunnel@128.2.204.215 -NTf`
