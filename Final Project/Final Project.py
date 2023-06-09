import cv2
import threading
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import customtkinter

blurred = False
rotated = False
pixelate = False
EmojiOnFace = False
Sunglasses = False

face_detector = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")
eyes_detector = cv2.CascadeClassifier('src/frontalEyes35x16.xml')
mouth_detector  = cv2.CascadeClassifier('src/mouth.xml')

sticker1 = cv2.imread("img/emoji1.png")
lip = cv2.imread('img/lips.png')
sunglasses = cv2.imread('img/sunglasses.png')


def button1_clicked():
    thread = threading.Thread(target=videoLoop, args=())
    thread.start()

def Emoji_clicked():
    global EmojiOnFace 
    EmojiOnFace = not EmojiOnFace

def Sunglasses_clicked():
    global Sunglasses
    Sunglasses = not Sunglasses

def blurred_clicked():
    global blurred
    blurred = not blurred

def pixelate_clicked():
    global pixelate
    pixelate = not pixelate

def rotated_clicked():
    global rotated
    rotated = not rotated

def videoLoop(mirror=True):

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1050)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 950)   

    while True:

        ret, frame = cap.read()
        if ret == False:
            break

        if mirror is True:
            frame = frame[:,::-1]

        faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    
        for (x, y, w, h) in faces:

            face_pos = frame[y:y+h, x:x+w]
            eyes = eyes_detector.detectMultiScale(face_pos, scaleFactor=1.2, minNeighbors=5)
            mouth = mouth_detector.detectMultiScale(face_pos, scaleFactor=1.3, minNeighbors=50)


        if EmojiOnFace:
            sticker = cv2.resize(sticker1, (w,h))
            img2gray = cv2.cvtColor(sticker,cv2.COLOR_BGR2GRAY)

            _,mask= cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            background1 = cv2.bitwise_and(face_pos,face_pos,mask = mask_inv)
            mask_sticker = cv2.bitwise_and(sticker, sticker, mask=mask)

            finalsticker =cv2.add(mask_sticker ,background1)
            frame[y:y+h, x:x+h] = finalsticker

        if Sunglasses:
            for (ex, ey, ew, eh) in eyes:

                sunglasses_resize = cv2.resize(sunglasses, (ew,eh))
                sunglasses2gray = cv2.cvtColor(sunglasses_resize,cv2.COLOR_BGR2GRAY)

                _,mask= cv2.threshold(sunglasses2gray,10,255,cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                eyes_pos = cv2.bitwise_and(face_pos[ey:ey+eh, ex:ex+ew],face_pos[ey:ey+eh, ex:ex+ew],mask = mask)
                mask_glasses = cv2.bitwise_and(sunglasses_resize, sunglasses_resize, mask=mask_inv)

                finalsticker =cv2.add(mask_glasses ,eyes_pos)
                face_pos[ey:ey+eh, ex:ex+ew] = finalsticker
                
            for (mx, my, mw, mh) in mouth:

                mouth_resize = cv2.resize(lip, (mw,mh))
                mouth2gray = cv2.cvtColor(mouth_resize,cv2.COLOR_BGR2GRAY)

                _,mask = cv2.threshold(mouth2gray,10,255,cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                nose_pos = cv2.bitwise_and(face_pos[my:my+mh, mx:mx+mw],face_pos[my:my+mh, mx:mx+mw],mask = mask_inv)
                mask_mouth = cv2.bitwise_and(mouth_resize, mouth_resize, mask=mask)

                finalmouth =cv2.add(mask_mouth ,nose_pos)
                face_pos[my:my+mh, mx:mx+mw] = finalmouth

        if blurred:          
            frame[y:y+h, x:x+h] = cv2.blur(frame[y:y+h, x:x+h],(30,30))

        if pixelate:
            for (x, y, w, h) in faces:
                square = cv2.resize(frame[y:y+h,x:x+w], (10,10))
            output = cv2.resize(square, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+h] = output
                                  
        if rotated:
            frame[y:y+h, x:x+h] = cv2.rotate(frame[y:y+h, x:x+h], cv2.ROTATE_180)


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        panel = tk.Label(image=image)
        panel.image = image
        panel.place(x=50, y=50)

        
    return panel

root = tk.Tk()
root.geometry("1920x1080+0+0")

button1 = tk.Button(root, text="Start", bg="#fff", font=("",20), command=button1_clicked)
button1.place(x=1100, y=100, width=120, height=70)

Emoji = tk.Button(root, text="Emoji", bg="#fff", font=("",20), command=Emoji_clicked)
Emoji.place(x=1100, y=170, width=120, height=70)

Sunglasses = tk.Button(root, text="Sunglasses", bg="#fff", font=("",15), command=Sunglasses_clicked)
Sunglasses.place(x=1100, y=240, width=120, height=70)

Blurred = tk.Button(root, text="Blurred", bg="#fff", font=("",20), command=blurred_clicked)
Blurred.place(x=1100, y=310, width=120, height=70)

Pixelate = tk.Button(root, text="Pixelate", bg="#fff", font=("",20), command=pixelate_clicked)
Pixelate.place(x=1100, y=380, width=120, height=70)

Rotated = tk.Button(root, text="Rotated", bg="#fff", font=("",20), command=rotated_clicked)
Rotated.place(x=1100, y=450, width=120, height=70)

root.title("Face Filter Final Project")
root.mainloop()