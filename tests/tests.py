#coding:utf-8
import requests

url = 'http://127.0.0.1:9090/upload'
files = {'ufile': open('tiger.jpeg', 'rb')}

r = requests.post(url, files=files)

print(r)
print(r.text)