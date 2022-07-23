import shutil
import requests

def download_img(url, filename):
    # url='https://datamall.lta.gov.sg/trafficsmart/images/8701_1744_20220722174533_b36a80.jpg'
    # filename = 'img.png'
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response