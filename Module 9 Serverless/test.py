import requests

url = 'http://localhost:8081/2015-03-31/functions/function/invocations'

data = {'url': 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'}

response = requests.post(url, json=data)

print("Response status code:", response.status_code)
print("Response text:", response.text)
print(response)

try:
    result = response.json()
    print(result)
except requests.exceptions.JSONDecodeError as e:
    print("Error decoding JSON:", e)