from PIL import Image, ImageDraw, ImageFont
from random import randint
import csv
import numpy as np

FONTPATH = "./data/font/Arial-Bold-Italic.ttf"
LETTERSTR = "02468BDFHJLNPRTVXZ"
COLORS = [0, 0.2, 0.4, 0.6, 0.8, 1, 1/6, 2/6, 3/6, 4/6, 5/6]

class rect:
    def __init__(self):
        self.size = (randint(5, 21), randint(5, 21))
        self.location = (randint(1, 199), randint(1, 59))
        self.luoverlay = True if randint(1, 10) > 6 else False
        self.rdoverlay = False if self.luoverlay else True if randint(1, 10) > 8 else False
        self.lucolor = 0 if randint(0, 1) else 255
        self.rdcolor = 0 if self.lucolor == 255 else 255
        self.ludrawn = False
        self.rddrawn = False
        self.pattern = randint(0, 1)


    def draw(self, image, overlay):
        if((overlay or not self.luoverlay) and not self.ludrawn):
            self.ludrawn = True
            stp = self.location
            transparent = int(255 * 0.45 if self.lucolor == 0 else 255 * 0.8)
            color = (self.lucolor, self.lucolor, self.lucolor, transparent)
            uline = Image.new("RGBA", (self.size[0], 1), color)
            lline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(uline, stp, uline)
            image.paste(lline, stp, lline)
        if((overlay or not self.rdoverlay) and not self.rddrawn):
            self.rddrawn = True
            dstp = (self.location[0], self.location[1] + self.size[1])
            rstp = (self.location[0] + self.size[0], self.location[1])
            transparent = int(255 * 0.45 if self.rdcolor == 0 else 255 * 0.8)
            color = (self.rdcolor, self.rdcolor, self.rdcolor, transparent)
            dline = Image.new("RGBA", (self.size[0], 1), color)
            rline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(dline, dstp, dline)
            image.paste(rline, rstp, rline)


class captchatext:
    def __init__(self, priority, offset, captchalen, engletter, ENGNOLIMIT):
        self.engletter = engletter
        self.letters = ''.join(LETTERSTR[randint(0, len(LETTERSTR) - 1)] for _ in range(captchalen))
        self.color = [0, 0, 255]
        self.angle = randint(-55, 55)
        self.priority = priority
        self.offset = offset
        self.next_offset = 0
        self.captchalen = captchalen


    def draw(self, image):
        gradient = Image.open('./data/gradient.png')
        font = ImageFont.truetype(FONTPATH, 21)
        text = Image.new("L", (78, 30))
        textdraw = ImageDraw.Draw(text)
        location = (6,1)
        textdraw.text(location, self.letters, font=font, fill='white')
        gradient.putalpha(text)
        gradient.save('gradient.png')
        image.paste(gradient, (0,0), gradient)

def add_salt_and_pepper(image, amount):

    output = np.copy(np.array(image))

    # add salt
    nb_salt = np.ceil(amount * output.size * 0.05)

    i = 0
    while i < nb_salt:
        x, y = randint(0, 30 - 1), randint(0, 78 - 1)
        output[(x, y)] = [COLORS[randint(0, 10)]*255, COLORS[randint(0, 10)]*255, COLORS[randint(0, 10)]*255, 0]
        i = i + 1

    return Image.fromarray(output)

def generate(GENNUM, SAVEPATH, ENGP=25, FIVEP=0, ENGNOLIMIT=False, filename="train"):
    captchacsv = open(SAVEPATH + "captcha_{:s}.csv".format(filename), 'w', encoding = 'utf8', newline = '')
    lencsv = open(SAVEPATH + "len_{:s}.csv".format(filename), 'w', encoding = 'utf8', newline = '')
    letterlist = []
    lenlist = []
    for index in range(1, GENNUM + 1, 1):
        captchastr = ""
        captchalen = 5
        captcha = Image.new('RGBA', (78, 30), (255, 255, 255, 255))
        #rectlist = [rect() for _ in range(32)]
        #for obj in rectlist:
        #    obj.draw(image=captcha, overlay=False)
        offset = 0
        #for i in range(captchalen):
        newtext = captchatext(0, offset, captchalen, True, ENGNOLIMIT)
        newtext.draw(image=captcha)
        offset = newtext.next_offset
        captchastr += newtext.letters

        mean = 0
        var = 10
        sigma = var ** 0.5
        captcha = add_salt_and_pepper(captcha, 1)


        letterlist.append([str(index).zfill(len(str(GENNUM))), captchastr])
        lenlist.append([str(index).zfill(len(str(GENNUM))), captchalen])
        #for obj in rectlist:
        #    obj.draw(image=captcha, overlay=True)
        captcha.convert("RGB").save(SAVEPATH + str(index).zfill(len(str(GENNUM))) + ".gif", "GIF")
    writer = csv.writer(captchacsv)
    writer.writerows(letterlist)
    writer = csv.writer(lencsv)
    writer.writerows(lenlist)
    captchacsv.close()
    lencsv.close()


if __name__ == "__main__":
    generate(50000, "./data/5_imitate_train_set/",  ENGP=100, FIVEP=100, ENGNOLIMIT=True, filename="train")
    generate(10240, "./data/5_imitate_vali_set/",  ENGP=100, FIVEP=100, ENGNOLIMIT=True, filename="vali")
