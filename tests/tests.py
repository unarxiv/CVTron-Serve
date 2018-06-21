#coding:utf-8
import requests

def test_inference():
    url = [
        'http://127.0.0.1:9090/segmentor/segment'
        ]

    for each in url:
        files = {'ufile': open('tests/tiger.jpeg', 'rb')}
        r = requests.post(each, files=files)
        print(r)
        print(r.text)

def test_train():
    config_url = 'http://127.0.0.1:9090/segmentor/get_train_config'
    r = requests.get(config_url)
    print(r.text)

def test_misc():
    hardware_url = 'http://127.0.0.1:9090/resource/device'
    r = requests.get(hardware_url)
    print(r.text)

if __name__ == '__main__':
    test_misc()