import requests

fileobj = open('test.zip', 'rb')
r = requests.post('http://localhost:6000/', data={"test": "go"}, files={"archive": ("test.zip", fileobj)})

print(r.text)
