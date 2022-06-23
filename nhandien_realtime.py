# LOAD THU VIEN VA MODUL 
import cv2
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import Preprocess
import math
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image

ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

Min_char_area = 0.015
Max_char_area = 0.06

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

max_size_plate = 18000
min_size_plate = 5000

RESIZED_IMAGE_WIDTH = 25
RESIZED_IMAGE_HEIGHT = 55

tongframe = 0
biensotimthay = 0

#DOC HINH ANH - TACH HINH ANH NHAN DIEN
cap = cv2.VideoCapture(0)

while(True):
    # tiền xử lý ảnh
    ret, img = cap.read()
    # img=cv2.resize(img,dsize = (1920,1080))
    tongframe = tongframe + 1
    #img = cv2.resize(img, None, fx=0.5, fy=0.5) 
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate,250,255) #Tách biên bằng canny
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1) #tăng sharp cho egde (Phép nở). để biên canny chỗ nào bị đứt thì nó liền lại để vẽ contour

    # lọc vùng biển số 
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10] #Lấy 10 contours có diện tích lớn nhất
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True) #Tính chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True) # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w/h
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)
    if screenCnt is None:
        detected = 0
        print ("No plate detected")
    else:
        detected = 1

    if detected == 1:
        n=1
        for screenCnt in screenCnt:

            ################## Tính góc xoay###############
            (x1,y1) = screenCnt[0,0]
            (x2,y2) = screenCnt[1,0]
            (x3,y3) = screenCnt[2,0]
            (x4,y4) = screenCnt[3,0]
            array = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
            sorted_array = array.sort(reverse=True, key=lambda x:x[1])
            (x1,y1) = array[0]
            (x2,y2) = array[1]

            doi = abs(y1 - y2)
            ke = abs (x1 - x2)
            angle = math.atan(doi/ke) * (180.0 / math.pi) 
            #################################################

            # Masking the part other than the number plate
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
  
               # Now crop
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))

            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx)/2, (bottomy - topy)/2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx ))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx ))

            roi = cv2.resize(roi,(0,0),fx = 3, fy = 3)
            imgThresh = cv2.resize(imgThresh,(0,0),fx = 3, fy = 3)

            #Tiền xử lý biển số
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            thre_mor = cv2.morphologyEx(imgThresh,cv2.MORPH_DILATE,kerel3)
            cont,hier = cv2.findContours(thre_mor,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            #Phân đoạn kí tự
            char_x_ind = {}
            char_x = []
            height, width,_ = roi.shape
            roiarea = height*width
            #print ("roiarea",roiarea)
            for ind,cnt in enumerate(cont) :
                area = cv2.contourArea(cnt)
                (x,y,w,h) = cv2.boundingRect(cont[ind])
                ratiochar = w/h
                if (Min_char*roiarea < area < Max_char*roiarea) and ( 0.25 < ratiochar < 0.7):
                    if x in char_x: #Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)    
                    char_x_ind[x] = ind

                
            # Nhận diện kí tự và in ra số xe
            if len(char_x) in range (7,10):
                cv2.drawContours(img, [screenCnt], -1, (0,255, 0), 3)

                char_x = sorted(char_x) 
                strFinalString = ""
                hang1 = ""
                hang2 = ""
                strCurrentChar='0'
                for i in char_x:
                    (x,y,w,h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        
                    imgROI = thre_mor[y:y+h,x:x+w]     
                    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     
                    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))     
                    cv2.imwrite("Kytu.jpg",imgROIResized)

                    model=load_model('model_trained.h5')
                    path="Kytu.jpg"     
                    img_file=tf.keras.utils.load_img(path,target_size=(20,60))
                    x=tf.keras.utils.img_to_array(img_file)
                    x=np.expand_dims(x, axis=0)
                    img_data=preprocess_input(x)
                    classes=model.predict(img_data)


                    if int(classes[0][0])==1:
                        strCurrentChar='0'
                    elif int(classes[0][1])==1:
                        strCurrentChar='1'
                    elif int(classes[0][2])==1:
                        strCurrentChar='2'
                    elif int(classes[0][3])==1:
                        strCurrentChar='3'
                    elif int(classes[0][4])==1:
                        strCurrentChar='4'
                    elif int(classes[0][5])==1:
                        strCurrentChar='5'
                    elif int(classes[0][6])==1:
                        strCurrentChar='6'
                    elif int(classes[0][7])==1:
                        strCurrentChar='7'
                    elif int(classes[0][8])==1:
                        strCurrentChar='8'
                    elif int(classes[0][9])==1:
                        strCurrentChar='9'
                    elif int(classes[0][10])==1:
                        strCurrentChar='A'
                    elif int(classes[0][11])==1:
                        strCurrentChar='B'
                    elif int(classes[0][12])==1:
                        strCurrentChar='C'
                    elif int(classes[0][13])==1:
                        strCurrentChar='D'
                    elif int(classes[0][14])==1:
                        strCurrentChar='E'
                    elif int(classes[0][15])==1:
                        strCurrentChar='F'
                    elif int(classes[0][16])==1:
                        strCurrentChar='G'
                    elif int(classes[0][17])==1:
                        strCurrentChar='H'
                    elif int(classes[0][18])==1:
                        strCurrentChar='K'
                    elif int(classes[0][19])==1:
                        strCurrentChar='L'
                    elif int(classes[0][20])==1:
                        strCurrentChar='M'
                    elif int(classes[0][21])==1:
                        strCurrentChar='N'
                    elif int(classes[0][22])==1:
                        strCurrentChar='P'
                    elif int(classes[0][23])==1:
                        strCurrentChar='Q'
                    elif int(classes[0][24])==1:
                        strCurrentChar='R'
                    elif int(classes[0][25])==1:
                        strCurrentChar='S'
                    elif int(classes[0][26])==1:
                        strCurrentChar='T'
                    elif int(classes[0][27])==1:
                        strCurrentChar='U'
                    elif int(classes[0][28])==1:
                        strCurrentChar='V'
                    elif int(classes[0][29])==1:
                        strCurrentChar='X'
                    elif int(classes[0][30])==1:
                        strCurrentChar='Y'
                    elif int(classes[0][31])==1:
                        strCurrentChar='Z'

                    if (y < height/3): 
                        hang1 = hang1 + strCurrentChar
                    else:
                        hang2 = hang2 + strCurrentChar
                
                    strFinalString = hang1 + hang2   
                    print ("\n License Plate " +str(n)+ " is: " + hang1 + " - " + hang2 + "\n")
                    cv2.putText(img, strFinalString, (topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                    n = n + 1
                
                    if hang1 == "77L1":
                        print("Thành phố Quy Nhơn - Bình Định")
                    if hang1 == "77F1":
                        print("Thị xã An Nhơn - Bình Định")
                    if hang1 == "77M1":
                        print("Huyện An Lão - Bình Định")
                    if hang1 == "77K1":
                        print("Huyện Hoài Ân - Bình Định")
                    if hang1 == "77C1":
                        print("Huyện Hoài Nhơn - Bình Định")
                    if hang1 == "77E1":
                        print("Huyện Phù Cát - Bình Định")
                    if hang1 == "77D1":
                        print("Huyện Phù Mỹ - Bình Định")
                    if hang1 == "77G1":
                        print("Huyện Tuy Phước - Bình Định")
                    if hang1 == "77H1":
                        print("Huyện Tây Sơn - Bình Định")
                    if hang1 == "77B1":
                        print("Huyện Vân Canh - Bình Định")
                    if hang1 == "77N1":
                        print("Huyện Vĩnh Thạnh - Bình Định")
                    
                cv2.imshow("a",cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))

    imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('License plate', imgcopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()                        