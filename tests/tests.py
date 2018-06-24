#coding:utf-8
import requests

def test_inference():
    url = [
        'http://127.0.0.1:9090/segmentor/segment'
        ]

    for each in url:
        files = {'ufile': open('tests/tiger.jpeg', 'rb')}
        r = requests.post(each, files=files)
        print(r.text)

def test_train():
    config_url = 'http://127.0.0.1:9090/segmentor/get_train_config'
    r = requests.get(config_url)
    print(r.text)

def test_misc():
    hardware_url = 'http://127.0.0.1:9090/resource/device'
    r = requests.get(hardware_url)
    print(r.text)

def test_upload_zip():
    upload_url = 'http://127.0.0.1:9090/segmentor/upload_train_file'
    files = {
        'dataset': open('tests/classification.zip', 'rb')
    }
    r = requests.post(upload_url, files = files)
    print(r.text)

if __name__ == '__main__':
    test_upload_zip()