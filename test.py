import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import cv2

from decode_np import Decode


#crop
def crop(x,y,w,h,margin,img_width,img_height):
    xmin = int(x-w*margin)
    xmax = int(x+w*margin)
    ymin = int(y-h*margin)
    ymax = int(y+h*margin)
    if xmin<0:
        xmin = 0
    if ymin<0:
        ymin = 0
    if xmax>img_width:
        xmax = img_width
    if ymax>img_height:
        ymax = img_height
    return xmin,xmax,ymin,ymax


# display result
def show_results(img, yolo_img, results, img_width, img_height, model_age, model_gender, model_emotion):
    img_cp = img.copy()
    for (x, y, w, h) in results:
        w = w // 2
        h = h // 2
        x = x + w
        y = y + h
        # xmin,xmax,ymin,ymax=crop(x,y,w,h,1.0,img_width,img_height)
        # cv2.rectangle(yolo_img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        # cv2.rectangle(yolo_img,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
        # cv2.putText(yolo_img,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        #analyze detected face
        xmin2,xmax2,ymin2,ymax2=crop(x,y,w,h,1.1,img_width,img_height)
        face_image = img[ymin2:ymax2, xmin2:xmax2]

        if(face_image.shape[0]<=0 or face_image.shape[1]<=0):
            continue

        cv2.rectangle(yolo_img, (xmin2,ymin2), (xmax2,ymax2), color=(0,0,255), thickness=3)

        offset = 16

        lines_age = open('words/agegender_age_words.txt').readlines()
        lines_gender = open('words/agegender_gender_words.txt').readlines()
        lines_fer2013 = open('words/emotion_words.txt').readlines()

        if(model_age!=None):
            shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
            img_keras = cv2.resize(face_image, (shape[1],shape[2]))
            # img_keras = img_keras[::-1, :, ::-1].copy()	#BGR to RGB
            img_keras = np.expand_dims(img_keras, axis=0)
            img_keras = img_keras / 255.0

            pred_age_keras = model_age.predict(img_keras)[0]
            prob_age_keras = np.max(pred_age_keras)
            cls_age_keras = pred_age_keras.argmax()

            age=0
            for i in range(101):
                age=age+pred_age_keras[i]*i
            label=str(int(age))

            # label="%.2f" % prob_age_keras + " " + lines_age[cls_age_keras]

            cv2.putText(yolo_img, "Age : "+label, (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250))
            offset=offset+16

        if(model_gender!=None):
            shape = model_gender.layers[0].get_output_at(0).get_shape().as_list()

            img_gender = cv2.resize(face_image, (shape[1],shape[2]))
            #img_gender = img_gender[::-1, :, ::-1].copy()	#BGR to RGB
            img_gender = np.expand_dims(img_gender, axis=0)
            img_gender = img_gender / 255.0

            pred_gender_keras = model_gender.predict(img_gender)[0]
            prob_gender_keras = np.max(pred_gender_keras)
            cls_gender_keras = pred_gender_keras.argmax()
            cv2.putText(yolo_img, "Gender : %.2f" % prob_gender_keras + " " + lines_gender[cls_gender_keras], (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250))
            offset=offset+16

        if(model_emotion!=None):
            shape = model_emotion.layers[0].get_output_at(0).get_shape().as_list()

            img_fer2013 = cv2.resize(face_image, (shape[1],shape[2]))
            img_fer2013 = cv2.cvtColor(img_fer2013,cv2.COLOR_BGR2GRAY)
            img_fer2013 = np.expand_dims(img_fer2013, axis=0)
            img_fer2013 = np.expand_dims(img_fer2013, axis=3)
            img_fer2013 = img_fer2013 / 255.0 *2 -1

            pred_emotion_keras = model_emotion.predict(img_fer2013)[0]
            prob_emotion_keras = np.max(pred_emotion_keras)
            cls_emotion_keras = pred_emotion_keras.argmax()
            cv2.putText(yolo_img, "Emotion : %.2f" % prob_emotion_keras + " " + lines_fer2013[cls_emotion_keras], (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250))
            offset = offset+16
    return yolo_img


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


if __name__ == '__main__':
    print('Please visit https://github.com/miemie2013/Keras-YOLOv4 for more complete model!')

    # model_path = 'ep073-loss11.905.h5'
    model_path = 'yolo4_voc_weights.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    model_image_size = (608, 608)

    # 分数阈值和nms_iou阈值
    conf_thresh = 0.2
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    MODEL_ROOT_PATH = "./pretrain/"
    model_age = load_model(MODEL_ROOT_PATH + 'agegender_age101_squeezenet.hdf5')
    model_gender = load_model(MODEL_ROOT_PATH + 'agegender_gender_squeezenet.hdf5')
    model_emotion = load_model(MODEL_ROOT_PATH + 'fer2013_mini_XCEPTION.119-0.65.hdf5')
    # face recog
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    ##################################################################################
    # Picture
    # while True:
    #     img = input('Input image filename:')
    #     try:
    #         image = cv2.imread(img)
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     except:
    #         print('Open Error! Try again!')
    #         continue
    #     else:
    #         image, boxes, scores, classes = _decode.detect_image(image, True)
    #         # faces = face_cascade.detectMultiScale(gray_image, 1.1, 3)
    #         # for (x, y, w, h) in faces:
    #         # # 在原图像上绘制矩形
    #         #     image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #
    #         cv2.imshow('result', image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    ##################################################################################
    # Camera
    vid = cv2.VideoCapture(0)  # for camera use
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(video_size)
    # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, boxes, scores, classes = _decode.detect_image(frame, True)

        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Age and Gender Detection
        result = show_results(frame, result, faces, frame.shape[1], frame.shape[0], model_age, model_gender, model_emotion)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ##################################################################################
    # # Basler
    # from pypylon import pylon
    #
    # # conecting to the first available camera
    #
    # camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    #
    # # Grabing Continusely (video) with minimal delay
    # camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    # converter = pylon.ImageFormatConverter()
    #
    # # converting to opencv bgr format
    # converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    # converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    #
    # # vid = cv2.VideoCapture(0)  # for camera use
    # # if not vid.isOpened():
    # #     raise IOError("Couldn't open webcam or video")
    # # video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    # # video_fps = vid.get(cv2.CAP_PROP_FPS)
    # # video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    # #               int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # # print(video_size)
    # # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #
    # accum_time = 0
    # curr_fps = 0
    # fps = "FPS: ??"
    # prev_time = timer()
    # while camera.IsGrabbing():
    #     grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    #     if grabResult.GrabSucceeded():
    #         # Access the image data
    #         image = converter.Convert(grabResult)
    #         frame = image.GetArray()
    #         frame = cv2.resize(frame, (840, 480))
    #
    #     # return_value, frame = vid.read()
    #     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     result, boxes, scores, classes = _decode.detect_image(frame, True)
    #     faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     curr_time = timer()
    #     exec_time = curr_time - prev_time
    #     prev_time = curr_time
    #     accum_time = accum_time + exec_time
    #     curr_fps = curr_fps + 1
    #     if accum_time > 1:
    #         accum_time = accum_time - 1
    #         fps = "FPS: " + str(curr_fps)
    #         curr_fps = 0
    #     cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=0.50, color=(255, 0, 0), thickness=2)
    #     cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #     cv2.imshow("result", result)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    ##################################################################################
    yolo4_model.close_session()
