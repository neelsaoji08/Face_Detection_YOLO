from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import cv2
from time import time 

###########################
classID = 0 # 0 is Fake :::  1 is Real
outputFolderPath='Dataset/DataCollect'
confidence=0.8
save=True
blurThreshold=35 #Larger More Focus


debug=False
offestPrecentageW=10
offestPrecentageH=20
camWidth,camHeight=640,480
floatingPoint=6

###########################

cap=cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)

detector=FaceDetector()

while True:
    success, img =cap.read()
    imgOut=img.copy()
    img, bboxs = detector.findFaces(img,draw=False)

    listBlur=[] #True False Values indicating if the faces are blur or not
    listInfo=[] #The Normalized values and the class name for the label txt file

    if bboxs:
        for bbox in bboxs:
            x, y ,w ,h=bbox["bbox"]
            score=bbox["score"][0]
            #Check The confidence score
            if score>confidence:

                #Adding offset to the face detected
                offsetW=(offestPrecentageW/100)*w
                x=int(x-offsetW)
                w=int(w+offsetW*2)

                offsetH=(offestPrecentageH/100)*h
                y= int(y-offsetH*3)
                h=int(h+offsetH*3.5)

                #To Avoid Values below Zero
                if x<0: x=0
                if y<0: y=0
                if w<0: w=0
                if h<0: h=0
                

                #Finding the Blurriness
                imgFace=img[y:y+h,x:x+w]
                blurValue=int(cv2.Laplacian(imgFace,cv2.CV_64F).var())

                if blurValue > blurThreshold:
                    listBlur.append(True)
                else :
                    listBlur.append(False)

                
                #Normalization of Values
                ih , iw, _ =img.shape

                xc, yc=x+w/2, y+h/2

                xcn=round(xc/iw,floatingPoint)
                wn=round(w/iw,floatingPoint)
                ycn=round(yc/ih,floatingPoint)
                hn=round(h/iw,floatingPoint)

                #To Avoid Values below one
                if xcn>1: xcn=1
                if ycn>1: ycn=1
                if wn>1: wn=1
                if hn>1: hn=1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                #Drawing
                cvzone.putTextRect(imgOut,f'Score:{int(score*100)}%  Blur: {blurValue}',(x,y-20),scale=2,thickness=3)
                cv2.rectangle(imgOut,(x,y,w,h),(255,0,0),3)

                if debug:
                    cvzone.putTextRect(img,f'Score:{int(score*100)}%  Blur: {blurValue}',(x,y-20),scale=2,thickness=3)
                    cv2.rectangle(img,(x,y,w,h),(255,0,0),3)

        #To Save Image
        if save:
            if all(listBlur) and listBlur!=[]:
                timeNow=str(time())
                timeNow=timeNow.split('.')
                timeNow=timeNow[0]+timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg",img)

                #Saving the labels
                for info in listInfo:
                    f=open(f"{outputFolderPath}/{timeNow}.txt","a")
                    f.write(info)
                    f.close()

    cv2.imshow("Image",imgOut)
    #cv2.imshow("Image",imgFace)
    cv2.waitKey(1)
