import requests

fileobj = open('test.zip', 'rb')
r = requests.post('http://localhost:6000/submit', data={"gender": "male"}, files={"images": ("test.zip", fileobj)})

print(r.json())
