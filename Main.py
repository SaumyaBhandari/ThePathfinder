from tkinter import *
from tkinter import ttk
from Program import Program


win = Tk()
window_height = 200
window_width = 700

screen_width = win.winfo_screenwidth()
screen_height = win.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

win.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


def get_data():
    text = entry.get()
    win.destroy()
    Program(text)


label= Label(win, text='Type in "Webcam" to use webcam.\nType in the directory of the file for group segmentation. ', font=('Courier 14 bold'))
label.pack(pady=8)
entry = Entry(win, width=42)
entry.pack(pady=8)


ttk.Button(win, text= "Okay", command=get_data).pack(pady=16)
win.bind('<Return>',lambda event:get_data())

win.mainloop()