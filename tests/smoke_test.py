"""Quick smoke test — start server, hit endpoints, shut down."""
import subprocess, time, urllib.request, sys

proc = subprocess.Popen(
    ["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8005"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

for i in range(20):
    time.sleep(1)
    try:
        r = urllib.request.urlopen("http://127.0.0.1:8005/")
        print("GET /       →", r.status, r.read().decode())
        r = urllib.request.urlopen("http://127.0.0.1:8005/api/health")
        print("GET /health →", r.status, r.read().decode())
        r = urllib.request.urlopen("http://127.0.0.1:8005/api/exercises")
        print("GET /exercises →", r.status, r.read().decode()[:200])
        print("\n✅ ALL ENDPOINTS OK")
        break
    except Exception:
        if i == 19:
            print("Server never came up")
            out = proc.stdout.read().decode()[:2000]
            print(out)
else:
    print("\nDone")
proc.terminate()
proc.wait()
