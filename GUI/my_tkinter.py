import csv
import tkinter as tk
import numpy as np

data = {}
for i in range(20):
    with open('./training_data/itr_250/path%02d.csv' % i, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        data[i] = []
        for n, row in enumerate(spamreader):
            if n != 0:
                 data[i].append([float(el) for el in row])
        data[i] = np.array(data[i])

master = tk.Tk()
master.title('Trajectory')
master.geometry('480x360')


s = 0
def select():
    global s
    s = (s + 1) % 20
    w.delete("all")
    for i in range(len(data[s])):
        if i == len(data[s]) - 1:
            break
        p11 = data[s][i][0] * width * 0.5
        p12 = data[s][i][2] * height * 0.5
        p21 = data[s][i+1][0] * width * 0.5
        p22 = data[s][i+1][2] * height * 0.5
        w.create_line(p11, p12, p21, p22, fill="#476042", width=3)

b1 = tk.Button(master, text='print selection', width=15,
              height=2, command=select)
b1.pack()

width, height = 360, 360
w = tk.Canvas(master, width=width, height=height)
w.pack()

master.mainloop()
