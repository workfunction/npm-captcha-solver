from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import datetime
import io

model = load_model("data/model/imitate_5_model.h5")
LETTERSTR = "02468BDFHJLNPRTVXZ"

prev_element = 0
send = 0

### 設定送出時間 ###
date_time_str = '2021-09-12 08:22:00'
### 設定送出時間 ###

date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get('https://npm.cpami.gov.tw/apply_2_1.aspx')

while True:
    if datetime.datetime.now() >= date_time_obj:
        date_time_obj = datetime.datetime.strptime('2099-09-12 07:00:00', '%Y-%m-%d %H:%M:%S')
        driver.refresh()
        send = 1
    try:
        captcha_element = driver.find_element_by_id('ContentPlaceHolder1_imgcode')
        if prev_element == captcha_element:
            continue
        prev_element = captcha_element
        image = captcha_element.screenshot_as_png
    except:
        prev_element = 0
        continue

    imageStream = io.BytesIO(image)
    captcha = Image.open(imageStream).convert("RGB")
    captcha.save('captcha.gif', 'GIF')

    prediction = model.predict(np.stack([np.array(captcha)[1:-1, 1:-1]/255.0]))
    answer = ""
    for predict in prediction:
        answer += LETTERSTR[np.argmax(predict[0])]

    captcha_textbox = driver.find_element_by_id('ContentPlaceHolder1_vcode')
    captcha_textbox.clear()
    captcha_textbox.send_keys(answer)

    if send == 1:
        try:
            send_button = driver.find_element_by_id('ContentPlaceHolder1_btnsave')
        except:
            continue
        send_button.click()
        send = 0
