import requests

url = "http://127.0.0.1:8000/analyze"
data = {"url": "https://www.bbc.com/news/articles/c4gzl5p8zpvo"}
try:
    response = requests.post(url, json=data)
    print("Status:", response.status_code)
    print("Response:", response.text[:200])
except Exception as e:
    print("Error:", e)
