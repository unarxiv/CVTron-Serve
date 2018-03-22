#coding:utf-8
import requests

url = [
    'http://127.0.0.1:9090/segmentor/segment'
    ]

for each in url:
    files = {'ufile': open('tests/tiger.jpeg', 'rb')}
    r = requests.post(each, files=files)
    print(r)
    print(r.text)
