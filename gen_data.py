import numpy as np 
import cv2, math
from os import mkdir, path
from random import randint, uniform, shuffle
import matplotlib.pyplot as plt
import csv

def noisy(image, noise_typ, var, amount):
    if noise_typ == "gauss":
        row,col= image.shape
        ch = 1
        mean = 0
        #var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        ch = 1
        s_vs_p = var
        out = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0

        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy

tot_imgs = []


for i in range(10000):

    radius = randint(5,10)
    color_in = randint(0,100)

    color_out = randint(200,255)

    center_x = randint(4,20)
    center_y = randint(4,20)

    var_sp = np.random.uniform(0.3,0.7)
    var_gauss = np.random.uniform(0.0,0.2)

    amount = np.random.uniform(0.001,0.01)
    image = np.ones((32,32), np.uint8)
    image *= color_out

    cv2.circle(image, (center_x, center_y), radius, color_in, -1)

    image = noisy(image, 's&p', var_sp, amount)
    image = noisy(image, 'gauss', var_gauss, amount)

    cv2.imwrite('frames/c0/' + str(i).zfill(5)  +".png", image)
    tot_imgs.append(['frames/c0/' + str(i).zfill(5)  +".png", 0])


for i in range(10000):

    color_in = randint(0,100)

    color_out = randint(200,255)

    pt1_x = randint(5,15)
    pt1_y = randint(5,15)

    pt2_x = randint(15,25)
    pt2_y = randint(15,25)


    var_sp = np.random.uniform(0.3,0.7)
    var_gauss = np.random.uniform(0.0,0.2)

    amount = np.random.uniform(0.001,0.01)
    image = np.ones((32,32), np.uint8)
    image *= color_out

    cv2.rectangle(image, (pt1_x, pt1_y), (pt2_x, pt2_y), color_in, -1)

    image = noisy(image, 's&p', var_sp, amount)
    image = noisy(image, 'gauss', var_gauss, amount)

    cv2.imwrite('frames/c1/' + str(i).zfill(5)  +".png", image)
    tot_imgs.append(['frames/c1/' + str(i).zfill(5)  +".png", 1])

for i in range(10000):

    fonts = [cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL,cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_PLAIN,
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]

    font = fonts[randint(0,len(fonts)-1)]
    color_in = randint(0,100)

    color_out = randint(200,255)

    pt1_x = randint(2,8)
    pt1_y = randint(12,18)


    var_sp = np.random.uniform(0.3,0.7)
    var_gauss = np.random.uniform(0.0,0.2)

    amount = np.random.uniform(0.001,0.01)
    image = np.ones((32,32), np.uint8)
    image *= color_out

    cv2.putText(image, 'IITD', (pt1_x, pt1_y), font, uniform(0.3,0.5), color_in, randint(1,3))

    image = noisy(image, 's&p', var_sp, amount)
    image = noisy(image, 'gauss', var_gauss, amount)

    cv2.imwrite('frames/c2/' + str(i).zfill(5)  +".png", image)
    tot_imgs.append(['frames/c2/' + str(i).zfill(5)  +".png", 2])

shuffle(tot_imgs)
shuffle(tot_imgs)
shuffle(tot_imgs)

train = tot_imgs[:15000]
test = tot_imgs[15000:]

w_train = csv.writer(open('train.txt', 'wb'), delimiter=' ')
w_test = csv.writer(open('test.txt', 'wb'), delimiter=' ')

for t in train:
    w_train.writerow(t)

for t in test:
    w_test.writerow(t)