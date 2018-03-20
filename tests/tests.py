#coding:utf-8
import requests

url = [
    'http://127.0.0.1:8080/classifier/classify',
    'http://127.0.0.1:8080/detector/detect',
    'http://127.0.0.1:8080/segmentor/segment'
    ]

for each in url:
    files = {'ufile': open('tests/tiger.jpeg', 'rb')}
    r = requests.post(each, files=files)
    print(r)
    print(r.text)
