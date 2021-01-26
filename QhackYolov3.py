from cv2 import cv2
import numpy as np 
import time
from gtts import gTTS
import os
from playsound import playsound
from twilio.rest import Client

'''
TwiceVision
base YOLO framework learnt from https://www.youtube.com/watch?v=h56M5iUVgGs&t=311s

Using the framework for Yolo, This program is targetted at store owners as a capacity counting replacement.
Capacity counting right now due to covid mainly involves the use of staff which is a waste of resources, By using
TwiceVision, The store owner should be notified of capacity by SMS, and the entire store should also be notified
through a text to speech audio display.

'''


#currently using tiny weights to improve FPS problems, can use alternative if greater processing power
#Tiny weights are less accurate but up the frame rate by alot, real weights will be used for the demo for accuracy proof.
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#Green set to colors
colors = (0,255,0) 
#load YOLO 

#Change the capacity of the room
maxcapacity = 5

cap = cv2.VideoCapture(0)
starting_time = time.time()
frame_id = 0
#this is so that the TTS does not run on every frame, 60 is too much here but
# the program runs around 12 fps, so 60 would give a good break for TTS
#i put it at 40, TTS starts before the windows pops up, this gives it time to load
indexnum = 0

while True:
    _,frame = cap.read()
    #the frame was mirrored
    frame = cv2.flip(frame,1)
    frame_id +=1

    height, width, channel = frame.shape

    #convert image to a blob, extract
    #detecting objects
    #yolo needs the blobs
    #changed size to help processing, smaller = faster ,running on mac
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0,0,0), True, crop = False) #invert blue with red


    net.setInput(blob)
    #forward to the end
    outs = net.forward(output_layers)

    #show info on screen

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            #detect confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                
                #object detected
                center_x = int(detection[0]*width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3]*height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #how many objects detected
    #take only these indexes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    num_objects_detected = len(boxes)
    num_person_detected = 0
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            #only count the person
            if label == 'person':
                num_person_detected = num_person_detected+ 1
            #replace with colors for clarity, colors made above
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, label, (x,y + 30), font, 3, (0.255,0),3)

            #google TTS output of label
            #Every 60 frames the frame would be scanned, this was for demo purposes, the intervals between each scan can be
            #adjusted accordingly to the store owners desire. 
            if indexnum == 60:
                #Twilio API codes for sending text messages to specified "to" number
                client = Client("AC2980ffd04bb537b836e052ce09c5915e", "f5553dbc73e29be14b59802caa20527e")
                text = "There are " + str(num_person_detected) +  "Persons. Store is at" + str( int((num_person_detected / maxcapacity) * 100)) + " Percent Capacity"
                #capacity is at 100%
                if (num_person_detected / maxcapacity) * 100 == 100:
                    client.messages.create(to="+12368636162", 
                       from_="+16122947829", 
                       body="Store is at max capacity: "+str(num_person_detected))
                #Capacity is at 60%
                elif(num_person_detected / maxcapacity) * 100 >= 60:
                        client.messages.create(to="+12368636162", 
                       from_="+16122947829", 
                       body="Store is approaching max capacity: "+str(num_person_detected))
                #Someone has entered the store
                elif(num_person_detected / maxcapacity) * 100 >= 0:
                    client.messages.create(to="+12368636162", 
                    from_="+16122947829", 
                    body="Someone has entered the store: "+str(num_person_detected))
                    
                #Google Text to speech
                language = 'en'
                speech = gTTS(text = text, lang = language, slow = False)
                speech.save("text.mp3")
                #Saves the audio as text.mp3
                os.system("start text.mp3")
                #Plays the sound file
                playsound('text.mp3')
                indexnum = 0
            else:
                indexnum +=1
                

    #FPS counter
    elapsed_time = time.time() -starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(fps), (10,30),font, 3, (0,255,0), 1)
    


    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
