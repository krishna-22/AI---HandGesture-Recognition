import cv2
import datetime
import argparse
import imutils
from imutils.video import VideoStream
import os
import random

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from keras.models import load_model
import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

from utils import detector_utils as detector_utils

import nltk,pandas as pd,numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, RegexpStemmer
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import FreqDist
from nltk.corpus import brown

import pyttsx3

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'Complete1':26,'Complete2':27,'delete':28}


def get_labels_for_plot(predictions):
    predictions_labels = []
    for ins in labels_dict:
        if predictions == labels_dict[ins]:
            predictions_labels.append(ins)
            return ins

            # break
    # return predictions_labels


def load_test_dataa():
    images = []
    names = []
    size = 50, 50
    temp = cv2.imread('./../images/closing.jpg')
    temp = cv2.resize(temp, size)
    # temp = cv2.cvtColor(temp,cv2.COLOR_RGB2GRAY)
    images.append(temp)
    names.append('img2')
    images = np.array(images)
    images = images.astype('float32') / 255.0
    return images, names


# -------------------------------------------------------------------------------
# Function - To find the running average over the background
# -------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


# -------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
# -------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    return thresholded

def recommend_word(text):
    if len(text)==0:
        return [None,None]
    r = []
    flag = 0
    cnt = 0

    for i in words:
        if i.startswith(text):
            r.append(i)
            cnt += 1
            if len(r) == 2:
                flag = 1
                return r

    if cnt == 1:
        return [r[0],None]
    elif flag == 0:
        return [None,None]

def recommend_sentence(text):
        if len(text)==0:
            return [None,None]
        sent = [text]
        test = [func(sent[0])]
        x1 = x.transform(test)
        vals = cosine_similarity(x1, y)
        vals = np.array(vals)
        result = np.argsort(vals)
        index1 = result[0][-1]
        index2 = result[0][-2]
        return ([data.loc[index1], data.loc[index2]])

def func(text):
        # text = ' '.join([stem.stem(i) for i in text.split()])
        # text = ' '.join([lemma.lemmatize(i) for i in text.split()])
        return text.lower()




ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60

    # model = create_model()
    # model.load_weights('Final_model_wn_weights.h5')

    Engine=pyttsx3.init()

    
    data=pd.read_excel('Book1.xlsx')
    

    nltk.download('stopwords')
    stop = stopwords.words('english')
    b=['where','how','what','who','why']
    stop=list(set(stop)-set(b))

    stem = PorterStemmer()
    nltk.download('wordnet')
    lemma = WordNetLemmatizer()


    data1=data
    #data1['data'] = data['data'].apply(func)

    x = TfidfVectorizer(stop_words=stop)
    y = x.fit_transform(data1['data'])

    nltk.download('brown')
    a = FreqDist(i.lower() for i in brown.words())
    b = a.most_common()
    words= []
    for j in b:
        if j[0] not in stop and len(j[0]) > 4:
            words.append(j[0])


    ####################################################

    model = load_model('./../models/new_model8.h5')
    # Get stream from webcam and set parameters)
    vs = VideoStream().start()

    # max number of hands we want to detect/track
    num_hands_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)

    arr = []
    final_arr = []
    Sent_arr = []
    r1 = "None"
    r2 = "None"
    r3 = "None"
    r4 = "None  "
    recommended = "None"
    counter = 0
    flag_r1 = 0
    flag_r2 = 0
    gesture_no = 1
    warn = 0
    bg = None
    aWeight = 0.5
    num_frames = 0

    print("\n\n\n\n\n\n\n")
    print("**********************************************************\n")
    print("Starting VideoStream\n")
    print("**********************************************************\n")

    try:

        while True:
            # Read Frame and process
            frame = vs.read()
            # frame = cv2.resize(frame, (320, 240))

            frame = cv2.flip(frame, 1)

            clone = frame.copy()

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                cv2.imwrite('./../images/frame_copy.jpg', frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            except:
                print("Error converting to RGB")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            change_every = 500

            if num_frames < 25 or ( (num_frames > ((int(num_frames/change_every))*change_every) + 450 and (num_frames < (int(num_frames/change_every))*change_every +525)) or (num_frames > ((int(num_frames/change_every))*change_every) and (num_frames < (int(num_frames/change_every))*change_every +25)) ):
                #print("Warning!! :Stay Still." + str(num_frames))

                cv2.putText(frame, 'Warning!! : Stay still!' ,
                        (int(im_width * 0.3), int(im_height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                warn =1


            if num_frames < 25 or (num_frames > ((int(num_frames/change_every))*change_every) and (num_frames < (int(num_frames/change_every))*change_every + 30)):
            #if num_frames < 30 or (num_frames > 500 and num_frames < 530):
                flag_bg = 1
                run_avg(gray, aWeight)
                #print("run average " + str(num_frames))
                # print("Warning!! :Stay Still.")
            else:
                # segment the hand region
                warn = 0
                # cv2.putText(frame, 'You can start entering gestures again!' ,
                #         (int(im_width * 0.3), int(im_height * 0.05)),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                flag_bg = 0
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded) = hand
                    cv2.imwrite('./../images/threshold.jpg', thresholded)

                    # draw the segmented region and display the frame
                    # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    cv2.imshow("Thesholded", thresholded)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        vs.stop()
                        break

            # print(num_frames)
            num_frames += 1

            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            kernel = np.ones((3, 3))

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # When no hand is detected
            if (max(scores) < 0.6):
                frame_copy = cv2.imread('./../images/frame_copy.jpg')
                cv2.imwrite("./../images/img_thr2.jpg", frame_copy)

            # Preparing cropped image for prediction
            tester_img, tester_name = load_test_dataa()
            pred = model.predict_classes(tester_img)
            predictions_labels_plot = get_labels_for_plot(pred)

            if (max(scores) < 0.6):
                if flag_bg == 0:
                    predictions_labels_plot = "nothing"

            if (len(arr) > 35):
                for elmts in arr:
                    if predictions_labels_plot == elmts:
                        counter += 1
                        if counter == len(arr):
                            arr = []
                            counter = 0
                            if flag_bg == 0:
                                if (predictions_labels_plot == 'nothing'):

                                    if warn != 1:
                                        if (len(final_arr) != 0):

                                            gesture_no = 1
                                            Sent_arr.append(str(''.join(final_arr)))
                                            Engine.say(str(''.join(final_arr)))
                                            Engine.runAndWait()
                                            final_arr = []
                                elif (predictions_labels_plot == 'delete'):

                                    if (len(final_arr) != 0):
                                        final_arr.pop()
                                        gesture_no += 1
                                        #print(str(''.join(final_arr)).lower())
                                        r1,r2=recommend_word(str(''.join(final_arr)).lower())

                                    else:
                                        if(len(Sent_arr)!=0):
                                            Sent_arr.pop()
                                            gesture_no = 1
                                


                                    gesture_no = 1
                                elif (predictions_labels_plot == 'Complete1'):
                                    print('c1 detected')
                                    if(r3!= None and flag_r1 == 1):
                                        Sent_arr = []
                                        Sent_arr.append(r3)
                                        final_arr = []
                                        Engine.say(r3)
                                        Engine.runAndWait()


                                        gesture_no = 1
                                        r3 = None
                                        r4 = None
                                        r1 = None
                                        r2 = None
                                        Sent_arr = []

                                        flag_r1 = 0
                                        flag_r2 = 0

                                    if(r1 != None):
                                        Sent_arr.append(r1)
                                        final_arr = []
                                        Engine.say(r1)
                                        Engine.runAndWait()

                                        word2tdf=r1
                                        print(word2tdf)
                                        gesture_no=1
                                        r1 = None
                                        r2 = None
                                        r3,r4=recommend_sentence(word2tdf)
                                        r3 = r3[0]
                                        r4 = r4[0]
                                        print(r3)
                                        print(r4)
                                        flag_r1 = 1

                                elif (predictions_labels_plot == 'Complete2'):

                                    if(r3!= None and flag_r1 == 1):
                                        Sent_arr = []
                                        Sent_arr.append(r4)
                                        final_arr = []
                                        Engine.say(r4)
                                        Engine.runAndWait()
                                        gesture_no = 1
                                        r3 = None
                                        r4 = None
                                        r1 = None
                                        r2 = None
                                        Sent_arr = []
                                        flag_r1 = 0
                                        flag_r2 = 0

                                    if(r2 != None):
                                        Sent_arr.append(r2)
                                        final_arr=[]
                                        Engine.say(r2)
                                        Engine.runAndWait()

                                        word2tdf=r2
                                        gesture_no=1
                                        r1 = None
                                        r2 = None
                                        r3,r4=recommend_sentence(word2tdf)
                                        r3 = r3[0]
                                        r4 = r4[0]
                                        print(r3)
                                        print(r4)
                                        flag_r2 = 1

        
                                else:
                                    Engine.say(predictions_labels_plot)
                                    Engine.runAndWait()
                                    final_arr.append(str(predictions_labels_plot))
                                    gesture_no += 1
                                
                                    if len(final_arr)!=0:
                                        #print(str(''.join(final_arr)).lower())
                                        r1,r2=recommend_word(str(''.join(final_arr)).lower())
     
                                    break

            arr.append(predictions_labels_plot)

            cv2.putText(frame, 'Enter gesture ' + str(gesture_no) + ': ' + str(predictions_labels_plot),
                        (int(im_width * 0.01), int(im_height * 0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, 'Word : ' + str(''.join(final_arr)),
                        (int(im_width * 0.01), int(im_height * 0.75)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, 'Word Recommendations -  1: ' + str(r1) + ' 2: ' + str(r2),
                        (int(im_width * 0.01), int(im_height * 0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, 'Sentence : ' + str(' '.join(Sent_arr)),
                        (int(im_width * 0.01), int(im_height * 0.96)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, 'Sentence Recommendation-1: ' + str(r3),
                        (int(im_width * 0.01), int(im_height * 0.82)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(frame, 'Sentence Recommendations-2: ' + str(r4),
                        (int(im_width * 0.01), int(im_height * 0.89)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                # Display FPS on frame
                # detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))