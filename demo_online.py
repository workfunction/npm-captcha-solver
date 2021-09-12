from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
import csv
import io

model = load_model("data/model/imitate_5_model.h5")
LETTERSTR = "02468BDFHJLNPRTVXZ"
SAVEPATH = "./data/real_data/"

prev_element = 0
send = 0

driver = webdriver.Chrome(ChromeDriverManager().install())

IDNumber = "A134310936" # 填入你的身分證字號
model = load_model("./data/model/imitate_5_model.h5") # 辨識5碼的Model

LETTERSTR = "02468BDFHJLNPRTVXZ"
correct, wrong = 0, 0
letterlist = []
captchacsv = open(SAVEPATH + "captcha_real.csv", 'w', encoding = 'utf8', newline = '')

for i in range(1000):# 跑1000次
    driver.get('https://npm.cpami.gov.tw/apply_3.aspx')
    serial_textbox = driver.find_element_by_id('ContentPlaceHolder1_serial')
    serial_textbox.send_keys("FFF")
    id_textbox = driver.find_element_by_id('ContentPlaceHolder1_sid')
    id_textbox.send_keys(IDNumber)
    select = Select(driver.find_element_by_id('ContentPlaceHolder1_nation'))
    select.select_by_value('中華民國')

    time.sleep(0.5)

    captcha_element = driver.find_element_by_id('ContentPlaceHolder1_imgcode')
    image = captcha_element.screenshot_as_png
    imageStream = io.BytesIO(image)
    captcha = Image.open(imageStream).convert("RGB")

    prediction = model.predict(np.stack([np.array(captcha)[1:-1, 1:-1]/255.0]))

    answer = ""
    for predict in prediction:
        answer += LETTERSTR[np.argmax(predict[0])]
    captcha_textbox = driver.find_element_by_id('ContentPlaceHolder1_vcode')
    captcha_textbox.send_keys(answer)
    driver.find_element_by_id('ContentPlaceHolder1_btnok').click()
    WebDriverWait(driver, 3).until(EC.alert_is_present())
    alert = driver.switch_to.alert
    text = alert.text
    alert.accept()
    if "驗證碼錯誤" in text:
        wrong += 1
        i = i - 1
    else:
        correct += 1
        captcha.save(SAVEPATH + str(i) + '.gif', 'GIF')
        letterlist.append([str(i), answer])
    print("{:.4f}% [Correct: {:d}/ Wrong: {:d}]".format(correct/(correct+wrong)*100, correct, wrong))
    time.sleep(1)

writer = csv.writer(captchacsv)
writer.writerows(letterlist)
