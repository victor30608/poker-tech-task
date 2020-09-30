import argparse
import cv2
from easyocr import Reader
import os
import numpy as np
import json

TEMPLATES_PATH = "templates"

def save_json(data,filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def find_best(result):
    max_prob = 0
    value = ''
    for (bbox, text, prob) in result:
        if prob > max_prob:
            max_prob = prob
            value = text
    return value

def find_cards(image):
    images= []
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 10000:
            roi = image[y:y + h, x:x + w]
            images.append(roi)
    return images

def find_suit(card_image):
    imgray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 3000 and cv2.contourArea(cnt) > 700:
            return card_image[y:y+h,x:x+w]

def preprocess(img,blur = True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.medianBlur(gray,3)
    return gray

def classify_suit(suit):
    tamplates = {}
    for file in os.listdir(TEMPLATES_PATH):
        image = cv2.imread(os.path.join(TEMPLATES_PATH,file))
        tamplates[file.split(".")[0]] = image
    label = ""
    min_val = 1000000000
    for name in tamplates:
        h, w, depth = suit.shape
        resized_templ = cv2.resize(tamplates[name],(w,h),cv2.INTER_CUBIC)
        diff = cv2.absdiff(preprocess(suit), preprocess(resized_templ))
        #print(name," ",np.sum(diff))
        if (np.sum(diff)<min_val):
            min_val = np.sum(diff)
            label = name
    return label

def find_min_cnt(contours):
    if len(contours) == 0:
        return None
    x, y, w, h = contours[0]
    min_sq = w*h
    out_cnt = None
    for cnt in contours:
        x, y, w, h = cnt
        if w*h<=min_sq:
            min_sq = w*h
            out_cnt = cnt
    return out_cnt

def analyze(image,th1=150,th2=450,blur = True):
    imgray = preprocess(image,blur)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    applicants = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # print(cv2.contourArea(cnt))
        if cv2.contourArea(cnt) > th1 and cv2.contourArea(cnt) < th2:
            #print( cv2.contourArea(cnt))
            # cv2.imshow("gg", image[y:y + h, x:x + w])
            # cv2.waitKey(0)
            applicants.append((x,y,w,h))
    out_cnt = find_min_cnt(applicants)
    if out_cnt is None:
        return None
    x, y, w, h = out_cnt
    # cv2.imshow("gg", image[y:y + h, x:x + w])
    # cv2.waitKey(0)
    return image[y:y+h,x:x+w]


def analyze_player_cards(img,reader):
    cards = []
    h, w, depth = img.shape
    part1 = img[0:int(h/2)+10,0:int(w/2)+1]
    part2 = img[0:int(h/2)+10,int(w/2):w]
    suit1 = analyze(part1)
    suit2 = analyze(part2)
    if suit1 is not None:
        h, w, depth = part1.shape
        #part1_c = part1[0:int(h/2)+10, 0:w]

        # cv2.imshow("gg", part1)
        # cv2.waitKey(0)

        result = reader.readtext(part1)
        print(result)
        value = find_best(result)
        label = classify_suit(suit1)
        cards.append({value:label})

    if suit2 is not None:
        h, w, depth = part2.shape
        #part2_c = part2[0:int(h/2)+10, 0:w]

        # cv2.imshow("gg2", part2)
        # cv2.waitKey(0)

        result = reader.readtext(part2)
        value = find_best(result)
        label = classify_suit(suit2)
        cards.append({value: label})
    return  cards



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())

# load the input image from disk
image = cv2.imread(args["image"])
# image_path = "image.png"
#
# image = cv2.imread(image_path)

reader = Reader(["en"])
##{name:[x,y,w,h]
OCRLocations = {
    "player1_balance":[440,1916,199,44],
    "player1_number_of_chips": [463, 1632, 165, 44],
    "player2_balance": [18,1649,189,42],
    "player2_number_of_chips": [215,1609,161,50],
    "player3_balance": [24,1310,185,37],
    "player3_number_of_chips": [211,1274,165,43],
    "pot":[442,863,192,43],
    "player1_cards":[643,1765,210,155],
    "cards_on_board":[138,905,789,197]
}
output = {}
for name in OCRLocations:
    (x, y, w, h) = OCRLocations[name]
    roi = image[y:y+h,x:x+w]

    if name == "player1_cards":
        cards = analyze_player_cards(roi,reader)
        output[name] = cards
        continue

    if name == "cards_on_board":
        cards = find_cards(roi)
        print("Len cards",len(cards))
        cards_arr =[]
        for card in cards:
            h, w, depth = card.shape
            part = card[52:52+43,0:64]
            #cv2.imshow("pp",part)
            #cv2.waitKey(0)
            #suit = analyze(card,100,1500,False)
            suit = analyze(part, 100, 1500, False)
            #suit = find_suit(card)
            # cv2.imshow("pp",suit)
            # cv2.waitKey(0)
            label = classify_suit(suit)
            text_image = card[0:59,0:122]
            result = reader.readtext(text_image)
            value = find_best(result)
            cards_arr.append({value:label})
        output[name] = cards_arr
        continue
    result = reader.readtext(roi)
    print(result)
    max_prob = 0
    value = ''
    for (bbox, text, prob) in result:
        if prob > max_prob:
            max_prob = prob
            value = text
    output[name] = value

print(output)
save_json(output,"output.json")

