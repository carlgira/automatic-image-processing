import requests

fileobj = open('test.zip', 'rb')
r = requests.post('http://localhost:6000/submit', data={"test": "go"}, files={"images": ("test.zip", fileobj)})

print(r.json())
