import shutil
import requests
import time
SAVEPATH = "./data/manual_label/"
url = 'https://npm.cpami.gov.tw/CheckImageCode.aspx'
for i in range(3000):
    try:
        response = requests.get(url, stream=True)
        with open(SAVEPATH + str(i) + '.gif', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
    except:
        i = i - 1
    time.sleep(0.5)
