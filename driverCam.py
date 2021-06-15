import torch as t
from torchreid.utils import FeatureExtractor
from torchreid.metrics.distance import compute_distance_matrix
import glob
from pygame import mixer
import cv2
from tkinter import *
import tkinter.messagebox
from tkinter import filedialog

model = t.load("osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth")
model = t.nn.Linear(1*500+12, 512)

# Initialize optimizer
optimizer = t.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print("Model's state_dict:")

for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Model weight:")    
print(model.weight)
print("Model bias:")    
print(model.bias)
print("---")
print("Optimizer's state_dict:")

for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
    
print("Model loaded successfullly")

extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    model_path='C:/Users/Burhan Sabir/deep-person-reid/Integeration/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth',
    device='cpu'
        )

def new():
    pass

def hel():
   help(cv2)

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.Muhammad Burhan \n2. Muhammad Saim \n3. Muhammad Barak Ullah \n")

def anotherWin():
   tkinter.messagebox.showinfo("About",'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')


def extractWebImages(pathOut):
#1 Create an object. Zero for external cam
    video = cv2.VideoCapture(0)
    count = 0
    a = 0
    while True:
        a = a + 1
        check , frame = video.read()
        print(check)
        print(frame)
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        cv2.imshow("Web-Cam", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break   
        video.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        print ('Read a new frame: ', check)
        cv2.imwrite( pathOut + "\\frame%d.jpg" % count, frame)
        count = count + 1
    video.release()
    cv2.destroyAllWindows()
    counter = 1
    shapes = []
    feature = []
    for img in glob.glob("C:/Users/Burhan Sabir/deep-person-reid/gui/WebCamFrames/*.jpg"):
        print("Reading Image in Matrix form: ")
        n= cv2.imread(img)
        print(n)
        tensor = extractor(img)
        feature.append(tensor)
        print("Feature is extracted of image ", counter)
        counter = counter + 1
        print("Now passing tensor to model...")    
        model.eval()
        out = model(tensor)
        print("The shape of tensor is ",out.shape)
        shapes.append(out.shape)
    print("Calculating shapes of all images...")
    print("Shapes of all images are: ",shapes)
    
    
def webCam():
    pathOut = 'C:/Users/Burhan Sabir/deep-person-reid/gui/WebCamFrames'
    extractWebImages(pathOut)
    
#=================================================================================================

def extractIpImages(pathIn, pathOut):
    count = 0
    while True:
        vidcap = cv2.VideoCapture(pathIn)
        success,frame = vidcap.read()
        cv2.imshow("Capturing",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        print ('Read a new frame: ', success)
        cv2.imwrite( pathOut + "\\frame%d.jpg" % count, frame)     # save frame as JPEG file
        count = count + 1
    vidcap.release()
    cv2.destroyAllWindows()
    counter = 1
    shapes = []
    feature = []
    for img in glob.glob("C:/Users/Burhan Sabir/deep-person-reid/gui/IpFrames/*.jpg"):
        print("Reading Image in Matrix form: ")
        n= cv2.imread(img)
        print(n)
        tensor = extractor(img)
        feature.append(tensor)
        print("Feature is extracted of image ", counter)
        counter = counter + 1
        print("Now passing tensor to model...")    
        model.eval()
        out = model(tensor)
        print("The shape of tensor is ",out.shape)
        shapes.append(out.shape)
    print("Calculating shapes of all images...")
    print("Shapes of all images are: ",shapes)
    
    
def IpCam():
    url = 'https://192.168.0.106:8080/shot.jpg'
    pathOut = 'C:/Users/Burhan Sabir/deep-person-reid/gui/IpFrames'
    extractIpImages(url, pathOut)

#=================================================================================================


def fromGallery():
    pathout = 'C:/Users/Burhan Sabir/deep-person-reid/gui/Galleryframes'
    vidcap = cv2.VideoCapture(openfn())
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(pathout + "\\frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1    
    counter = 1
    shapes = []
    feature = []
    for img in glob.glob("C:/Users/Burhan Sabir/deep-person-reid/gui/Galleryframes/*.jpg"):
        print("Reading Image in Matrix form: ")
        n= cv2.imread(img)
        print(n)
        tensor = extractor(img)
        feature.append(tensor)
        print("Feature is extracted of image ", counter)
        counter = counter + 1
        print("Now passing tensor to model...")    
        model.eval()
        out = model(tensor)
        print("The shape of tensor is ",out.shape)
        shapes.append(out.shape)
    print("Calculating shapes of all images...")
    print("Shapes of all images are: ",shapes)    
    
def report():
    pass

def setting():
    pass

def exitt():
    root.destroy()
   
root=Tk()
root.geometry('492x750')
#root.state("zoomed")
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Raven-The person Detector')
root.iconbitmap(r'favicon.ico')
frame.config(background='dimgray')
label = Label(frame, text="Raven",bg='dimgray',font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(file = "raven.png")
background_label = Label(frame , image=filename)
background_label.pack(side=TOP)

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="File",menu=subm1)
subm1.add_command(label="New",command=new)
subm1.add_command(label="Exit",command=exitt)

subm2 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm2)
subm2.add_command(label="Open CV Docs",command=hel)

subm3 = Menu(menu)
menu.add_cascade(label="About",menu=subm3)
subm3.add_command(label="Raven",command=anotherWin)
subm3.add_command(label="Contributors",command=Contri)


but1=Button(frame , padx=0,pady=0,width=40,bg='lightgray',fg='black',relief=GROOVE,command=webCam,text='Open WebCam',font=('helvetica 15 bold'))
but1.place(x=0,y=104)

but2=Button(frame,padx=0,pady=0,width=40,bg='lightgray',fg='black',relief=GROOVE,command=IpCam,text='Open IP-Cam',font=('helvetica 15 bold'))
but2.place(x=0,y=176)

but3=Button(frame,padx=0,pady=0,width=40,bg='lightgray',fg='black',relief=GROOVE,command=fromGallery,text='Import Video from Gallery',font=('helvetica 15 bold'))
but3.place(x=0,y=250)

but4=Button(frame,padx=0,pady=0,width=40,bg='lightgray',fg='black',relief=GROOVE,command=report,text='Report',font=('helvetica 15 bold'))
but4.place(x=0,y=322)

but5=Button(frame,padx=0,pady=0,width=40,bg='lightgray',fg='black',relief=GROOVE,command=setting,text='Settings',font=('helvetica 15 bold'))
but5.place(x=0,y=400)

but6=Button(frame,padx=0,pady=0,width=40,bg='lightgray',fg='black',relief=GROOVE,text='EXIT',command=exitt,font=('helvetica 15 bold'))
but6.place(x=0,y=478)

root.mainloop()


















