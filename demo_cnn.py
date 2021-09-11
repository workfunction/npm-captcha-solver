from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
import os
import csv
LETTERSTR = "02468BDFHJLNPRTVXZ"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(17)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


print("Loading test data...")
testcsv = open('./data/manual_label/captcha_test.csv', 'r', encoding = 'utf8')
test_data = np.stack([np.array(Image.open("./data/manual_label/" + row[0] + ".gif").convert('RGB'))[1:-1, 1:-1]/255.0 for row in csv.reader(testcsv)])
testcsv = open('./data/manual_label/captcha_test.csv', 'r', encoding = 'utf8')
test_label = [row[1] for row in csv.reader(testcsv)]
print("Loading model...")
K.clear_session()
model5 = load_model("./data/model/imitate_5_model.h5")
print("Predicting...")
prediction5 = model5.predict(test_data) # 5碼

total5 = len(test_data)
correct5 = 0
correct5digit= [0 for _ in range(5)]
totalalpha, correctalpha = len([1 for ans in test_label for char in ans if char.isalpha()]), 0
for i in range(total5):
    allequal = True
    for char in range(5):
        if LETTERSTR[np.argmax(prediction5[char][i])] == test_label[i][char]:
            correct5digit[char] += 1
            correctalpha += 1 if LETTERSTR[np.argmax(prediction5[char][i])].isalpha() else 0
        else:
            allequal = False
    if allequal:
        correct5 += 1
    else:
        checkcorrect = False

print("5digits model acc:{:.4f}%".format(correct5/total5*100)) # 5模型acc
for i in range(5):
    print("digit{:d} acc:{:.4f}%".format(i+1, correct5digit[i]/total5*100)) # 5模型各字元acc
print("---------------------------")
