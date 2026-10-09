#!/usr/bin/env python3
import json
import urllib.request

payload = {
    "model": "muse-spark-1.3-contributor-free",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "temperature": 0,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
