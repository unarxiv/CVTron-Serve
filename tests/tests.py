#coding:utf-8
import requests

url = 'http://192.168.1.4:9090/classify'
files = {'ufile': open('tiger.jpeg', 'rb')}

r = requests.post(url, files=files)

print(r)
print(r.text)