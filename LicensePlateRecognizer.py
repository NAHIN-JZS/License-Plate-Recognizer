# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:02:36 2021

@author: Nahin
"""


from tkinter import *
from tkinter.messagebox import askyesno, showinfo
from tkinter.filedialog import askopenfile, askopenfilename
from PIL import Image, ImageTk
import cv2
import numpy as np

#import random
import imutils
import csv
import time


global video
video=0

root = Tk()
root.geometry("800x600+300+100")
root.title("License Plate Recognizer")

def Welcomeguide():
    lab1 = Label(root, text="Welcome to License Plate Recognizer", font=("times",12, "bold")).place(x=295, y=245)
    lab2 = Label(root, text="To add image/video, Go to 'File', Click on 'Open Folder'", font=12).place(x=250, y=270)
    lab3 = Label(root, text="After choosing a file, Click on 'Analyze' to get License Plate Numbers", font=12).place(
        x=210, y=290)
    lab4 = Label(root, text="Check output.csv file", font=12).place(x=355, y=310)
    
    

def createmenu():
    menu1 = Menu()
    mymenu2 = Menu()
    mymenu2.add_command(label="Select image", command=openfiledirectory)
    mymenu2.add_command(label="Select video", command=openvideodirectory)
    mymenu2.add_command(label="Exit", command=exitwindow)
    menu1.add_cascade(label="File", menu=mymenu2)
    menu1.add_cascade(label="Analyze", command =startanalyzing)
    root.config(menu=menu1)

def exitwindow():
    me = askyesno(title="Warning", message="Are you sure you want to close ?")
    if me:
        root.destroy()





def openfiledirectory():
    global video
    video=0
    filepath = askopenfile(title="Select a file")
    global fname
    
    fname = filepath.name
    if (fname !=""):
        text="IMAGE SELECTED"
        r_label.config(text = text)
        img = cv2.imread(fname)
        cv2.imshow("Selected Image",img)
    

def openvideodirectory():
    global video
    video =1
    global fname
    fname = askopenfilename()
    if(fname != ""):
        text="VIDEO SELECTED"
        r_label.config(text = text)
    
    

def startanalyzing():
    showinfo("Info", "Analyzing....\nPress OK to continue")
    global video
    global fname
    if(video == 1):
        cap = cv2.VideoCapture(fname)
        #cv2.namedWindow("Selected Video", cv2.WINDOW_NORMAL)
        while True:
            _, img = cap.read()
            img = cv2.resize(img, None , fx=0.4, fy=0.4)
            #cv2.imshow("Selected Video", img)
            if cv2.waitKey(1)==27:
                break
            startmodel(img)
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(fname)
        startmodel(img)
        cv2.destroyAllWindows()
    text="NO IMAGE OR VIDEO IS SELECTED"
    r_label.config(text = text)
        
    
def showOutputOnPopUp(head,number):
    result = head+"\n"+number
    showinfo("Result", result+"\nPlease check output.csv file")
    
def showOutputOnInterfce(head,number):
    result = "RESULT: "+"\n"+head.upper()+"\n"+number
    r_label.config(text = result)
    

def startmodel(img):
    
    
    
    net = cv2.dnn.readNet("yolov4-obj.weights" , "yolov4-custom.cfg")
    
    char_net = cv2.dnn.readNet("yolov4-obj_char_final.weights" , "yolov4-obj_char_final.cfg")
    
    #Name custom object
    classes = ["number_plate"]
    
    
    char_classes = ["0","1","2","3","4","5","6","7","8","9","BA","CA","CHATTOGRAM","DA","DHAKA","GA","GHA","HA","JA","JASHORE","KA","KHA","KHULNA","METRO","NA","TA","VA"]
    num_class = ["0","1","2","3","4","5","6","7","8","9"]
    
    
    
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    
    char_layer_names = char_net.getLayerNames()
    char_output_layers = [char_layer_names[i[0] - 1] for i in char_net.getUnconnectedOutLayers()]
    char_colors = np.random.uniform(0, 255, size=(len(char_classes), 3))
    
    
    
    
    
    height, width, channels = img.shape

    #detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)
  
    net.setInput(blob)
    outs = net.forward(output_layers)
  
  

    #Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
  
    for out in outs:
    
      for detection in out:
      
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
        #object detection
        #print(class_id)
        
          center_x = int (detection[0] * width)
          center_y = int (detection[1] * height)
          w = int (detection[2] * width)
          h = int (detection[3] * height)

        #Rectangle coordinates
          x = int (center_x -w / 2)
          y = int (center_y - h / 2)
        
        
        
          boxes.append([x,y,w,h])
          confidences.append(float(confidence))
          class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)




    font = cv2.FONT_HERSHEY_PLAIN
    k=0
    for i in range(len(boxes)):
      if i in indexes:
        
          x,y,w,h = boxes[i]
          labe = str(classes[class_ids[i]])
          color = colors[class_ids[i]]
          plate_confi = confidences[i]
          #print(x,y,w,h)
          new_img = img[y:y+h, x:x+w]
          if new_img.size == 0:
              continue
          
        
        
          cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
          cv2.putText(img, labe +" "+str(plate_confi)[:4], (x,y), font, 0.8, (0, 255, 0), 1)
          
          cv2.imshow('Selected Video with Detected License Plate',img)
          cv2.waitKey(1)
        
        
           
          #Charecter part
          #loading image
          
          char_img = new_img
          char_img = cv2.resize(char_img, None , fx=1.0, fy=1.0)
          char_height, char_width, char_channels = char_img.shape
        
          #detecting objects
          char_blob = cv2.dnn.blobFromImage(char_img, 0.00392, (416,416), (0,0,0), True, crop = False)
          
          char_net.setInput(char_blob)
          char_outs = char_net.forward(char_output_layers)
          
          
        
          #Showing informations on the screen
          char_class_ids = []
          char_confidences = []
          char_boxes = []
          
          for char_out in char_outs:
            
            for char_detection in char_out:
              
              char_scores = char_detection[5:]
              char_class_id = np.argmax(char_scores)
              char_confidence = char_scores[char_class_id]
              if char_confidence > 0.9:
                #object detection
                
                
                char_center_x = int (char_detection[0] * char_width)
                char_center_y = int (char_detection[1] * char_height)
                char_w = int (char_detection[2] * char_width)
                char_h = int (char_detection[3] * char_height)
        
                #Rectangle coordinates
                char_x = int (char_center_x -char_w / 2)
                char_y = int (char_center_y - char_h / 2)
                
                
                
                char_boxes.append([char_x,char_y,char_w,char_h])
                char_confidences.append(float(char_confidence))
                char_class_ids.append(char_class_id)
        
          char_indexes = cv2.dnn.NMSBoxes(char_boxes, char_confidences, 0.5, 0.4)
          #print(indexes)
        
        
        
        
          char_font = cv2.FONT_HERSHEY_PLAIN
          
          char_final_obj = []
          for i in range(len(char_boxes)):
            if i in char_indexes:
                
                
                
                x,y,w,h = char_boxes[i]
              
                
                char_obj_index = (x,y,w,h,i)
                #print(obj_index)
                char_final_obj.append(char_obj_index)
                char_label = str(char_classes[char_class_ids[i]])
                char_color = char_colors[char_class_ids[i]]
              
              
                char_confi = char_confidences[i]
              
              
                char_new_img = char_img[y:y+h, x:x+w]
                if new_img.size == 0:
                    continue
                #print(img)
                char_new_img = imutils.resize(char_new_img, width=700, height=400, inter=cv2.INTER_CUBIC)
                
        
                cv2.rectangle(char_img, (x,y), (x+w,y+h), color, 2)
                cv2.putText(char_img, char_label+" "+str(char_confi)[0:4], (x,y), font, 1, (0, 0, 255), 1)
                
                cv2.imshow('Plate Recognizer',char_img)
                cv2.waitKey(1)
          
          char_final_obj.sort()
          #print(final_obj)
          number=" "
          high = " "
          for j in char_final_obj:
              char_index_no = j[4]
              
              
              if str(char_classes[char_class_ids[char_index_no]]) in num_class:
                  
                  number = number+str(char_classes[char_class_ids[char_index_no]])
              else:
                  high = high + char_classes[char_class_ids[char_index_no]]+ "  "
          localtime = time.asctime( time.localtime(time.time()) )
          lst = [high,number,localtime]
          with open('Output.csv','a') as file:
              w_obj = csv.writer(file)
              w_obj.writerow(lst)
          
          print("place is : " + high )
          print("number "+ number) 
          if(video==0):
              showOutputOnPopUp(high,number)
          else:
              showOutputOnInterfce(high,number)
        



Welcomeguide()
createmenu()
label1 = Label(root)
label1.pack()
r_label = Label(root,text="NO IMAGE OR VIDEO IS SELECTED",font=("times",12, "bold"))
r_label.pack()
root.mainloop()
