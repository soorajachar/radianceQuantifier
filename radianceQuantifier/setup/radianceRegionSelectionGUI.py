#! /usr/bin/env python3
import pickle, os, json, math, subprocess, numpy as np, pandas as pd, tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, ImageTk
from radianceQuantifier.dataprocessing.inVivoRadianceProcessing import calculate_radiance

if os.name == 'nt':
    dirSep = '\\'
else:
    dirSep = '/'

class RadianceRegionSelectionWindow(tk.Frame):
    def __init__(self, master, backPage, originPage, selectedExperiment, maxWidth, maxHeight):
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10)
        
        # l1 = tk.Label(mainWindow,text='Set color bar limits:')
        # v = tk.StringVar(value='manual')
        # rb1a = tk.Radiobutton(mainWindow, text="Automatically (pytesseract)",padx = 20, variable=v, value='auto')
        # rb1b = tk.Radiobutton(mainWindow,text="Manually",padx = 20, variable=v, value='manual')
        # l1.grid(row=0,column=0)
        # rb1a.grid(row=1,column=0,sticky=tk.W)
        # rb1b.grid(row=2,column=0,sticky=tk.W)
        # tk.Label(mainWindow,text='Min:').grid(row=2,column=1,sticky=tk.W)
        # minEntry = tk.Entry(mainWindow,width=5)
        # minEntry.grid(row=2,column=2,sticky=tk.W)
        # tk.Label(mainWindow,text='Max:').grid(row=2,column=3,sticky=tk.W)
        # maxEntry = tk.Entry(mainWindow,width=5)
        # maxEntry.grid(row=2,column=4,sticky=tk.W)

        global selected_rect, canvas
        selected_rect = None  # declare selected_rect as a global variable

        # Function to handle mouse click on the canvas
        def on_canvas_click(event):
            global selected_rect, x1, y1
            if selected_rect is not None:
                canvas.delete(selected_rect)  # Clear the previous rectangle
            x1, y1 = event.x, event.y
            selected_rect = canvas.create_rectangle(x1, y1, x1, y1, outline="red", width=1)
            update_rect_coordinates(x1, y1, x1, y1)

        # Function to handle mouse drag on the canvas
        def on_canvas_drag(event):
            global selected_rect, x2, y2
            if selected_rect is not None:
                x2, y2 = event.x, event.y
                canvas.coords(selected_rect, x1, y1, x2, y2)
                update_rect_coordinates(x1, y1, x2, y2)

        # Function to update the coordinates label
        def update_rect_coordinates(x1, y1, x2, y2):
            global sel_left, sel_right, sel_top, sel_bottom
            # adjust coordinates to stay within the canvas boundaries
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            x1 = max(0, min(x1, canvas_width))
            x2 = max(0, min(x2, canvas_width))
            y1 = max(0, min(y1, canvas_height))
            y2 = max(0, min(y2, canvas_height))

            # reassing min to be top left; max to be bottom right
            sel_left = min(x1,x2)
            sel_right = max(x1,x2)
            sel_top = min(y1,y2)
            sel_bottom = max(y1,y2)

            rect_coordinates.set(f'Selected Coordinates:\nleft={sel_left}\ntop={sel_top}\nright={sel_right}\nbottom={sel_bottom}')


        def handle_checkbutton_region_selection(): 
            '''
            Function to handle the checkbutton selection for the region selection to calculate radiance from images.
            '''
            # define dictionary with regions for radiance calculation
            # [(top,bottom),(left,right))]
            regionDict = { 'all':[(  0, -1),( 0,-1)],
                        'snout':[(  0, 60),(20,165)],
                        'lungs':[( 60,110),(20,165)],
                        'liver':[(110,210),(20,165)],
                    'abdomen':[(210,240),(20,165)],
                        'bmRm':[(240,320),(20,92)], # bone marrow (mouse right)
                        'bmLm':[(240,320),(93,165)]} # bone marrow (mouse left)

            # handle the checkbuttons #
            
            # selected entire image
            if check1_var.get() == 1: 
                calculate_radiance(left=0,right=maxWidth,top=0,bottom=maxHeight,text='all')
            # selected snout
            if check2_var.get() == 1:
                calculate_radiance(left=regionDict['snout'][1][0],right=regionDict['snout'][1][1],top=regionDict['snout'][0][0],bottom=regionDict['snout'][0][1],text='snout')
            # selected lungs
            if check3_var.get() == 1:
                calculate_radiance(left=regionDict['lungs'][1][0],right=regionDict['lungs'][1][1],top=regionDict['lungs'][0][0],bottom=regionDict['lungs'][0][1],text='lungs')
            # selected liver
            if check4_var.get() == 1:
                calculate_radiance(left=regionDict['liver'][1][0],right=regionDict['liver'][1][1],top=regionDict['liver'][0][0],bottom=regionDict['liver'][0][1],text='liver')
            # selected abdomen
            if check5_var.get() == 1:
                calculate_radiance(left=regionDict['abdomen'][1][0],right=regionDict['abdomen'][1][1],top=regionDict['abdomen'][0][0],bottom=regionDict['abdomen'][0][1],text='abdomen')
            # selected bone marrow (right)
            if check6_var.get() == 1:
                calculate_radiance(left=regionDict['bmRm'][1][0],right=regionDict['bmRm'][1][1],top=regionDict['bmRm'][0][0],bottom=regionDict['bmRm'][0][1],text='bmRm')
            # selected bone marow (left)
            if check7_var.get() == 1:
                calculate_radiance(left=regionDict['bmLm'][1][0],right=regionDict['bmLm'][1][1],top=regionDict['bmLm'][0][0],bottom=regionDict['bmLm'][0][1],text='bmLm')
            # selected custom region
            if (check8_var.get() == 1) & (selected_rect is not None):
                calculate_radiance(sel_left,sel_right,sel_top,sel_bottom,text='custom')

            # error handling
            error_flag = False
            if (check8_var.get() == 1) & (selected_rect is None): # selected custom region, but didn't draw rectangle
                print('Please choose a custom region.')
                error_flag = True
            if (check1_var.get() == 0) & (check2_var.get() == 0) & (check3_var.get() == 0) & (check4_var.get() == 0) & (check5_var.get() == 0) & (check6_var.get() == 0) & (check7_var.get() == 0) & (check8_var.get() == 0): # no buttons selected
                print('Please select a button.')
                error_flag = True

            if ~error_flag: tk.messagebox.showinfo(title='Success', message='Radiance calculation complete!\nSelect another region or click "Finished" to return to main window.')


        # Display average image
        avg_image_path = f'plots/Image Processing/avg_merged_image-{os.getcwd().split(dirSep)[-1]}.png'
        avg_image_original = Image.open(avg_image_path)
        avg_image = avg_image_original.resize((maxWidth, maxHeight), Image.ANTIALIAS)  # Rescale image
        avg_image = ImageTk.PhotoImage(avg_image)

        # Create a frame to display the image
        avg_image_label = tk.Label(mainWindow, image=avg_image)
        avg_image_label.image = avg_image  # To prevent image from being garbage collected
        avg_image_label.grid(row=0, column=0, rowspan=4)

        # Create a canvas for drawing rectangles
        canvas = tk.Canvas(mainWindow, width=maxWidth, height=maxHeight)
        canvas.place(in_=avg_image_label, anchor='nw') # place the canvas on the image
        canvas.create_image(0, 0, anchor='nw', image=avg_image) # display the image on the canvas

        # Initialize variables
        rect_coordinates = tk.StringVar()
        rect_coordinates_label = tk.Label(mainWindow, textvariable=rect_coordinates)
        rect_coordinates_label.grid(row=8, column=3, sticky='nw')

        # Set up check buttons
        # initialize
        check1_var = tk.IntVar() # entire mouse
        check2_var = tk.IntVar() # snout
        check3_var = tk.IntVar() # lungs
        check4_var = tk.IntVar() # liver
        check5_var = tk.IntVar() # abdomen
        check6_var = tk.IntVar() # bone marrow (right)
        check7_var = tk.IntVar() # bone marrow (left)
        check8_var = tk.IntVar() # custom region
        # make button
        check1 = tk.Checkbutton(mainWindow, text="Entire Image", variable=check1_var)
        check2 = tk.Checkbutton(mainWindow, text="Snout", variable=check2_var)
        check3 = tk.Checkbutton(mainWindow, text="Lungs", variable=check3_var)
        check4 = tk.Checkbutton(mainWindow, text="Liver", variable=check4_var)
        check5 = tk.Checkbutton(mainWindow, text="Abdomen", variable=check5_var)
        check6 = tk.Checkbutton(mainWindow, text="Bone Marrow (right)", variable=check6_var)
        check7 = tk.Checkbutton(mainWindow, text="Bone Marrow (left)", variable=check7_var)
        check8 = tk.Checkbutton(mainWindow, text="Custom Region", variable=check8_var)
        # layout for buttons
        check1.grid(row=0, column=3, sticky='w')
        check2.grid(row=1, column=3, sticky='w',)
        check3.grid(row=2, column=3, sticky='w')
        check4.grid(row=3, column=3, sticky='w')
        check5.grid(row=4, column=3, sticky='w')
        check6.grid(row=5, column=3, sticky='w')
        check7.grid(row=6, column=3, sticky='w')
        check8.grid(row=7, column=3, sticky='w')

        # Create a button to confirm the selection
        confirm_button = tk.Button(mainWindow, text="Calculate Radiance", command=(lambda : handle_checkbutton_region_selection()))
        confirm_button.grid(row=9, column=3, sticky='n')

        # Bind mouse events for rectangle drawing
        canvas.bind("<Button-1>", on_canvas_click)
        canvas.bind("<B1-Motion>", on_canvas_drag)



        # navigation buttons
        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=(tk.TOP), pady=10)
        tk.Button(buttonWindow, text='Finished', command=(lambda : master.switch_frame(originPage, selectedExperiment))).grid(row=10, column=3)
        tk.Button(buttonWindow, text='Back', command=(lambda : master.switch_frame(backPage, selectedExperiment))).grid(row=10, column=4)
        tk.Button(buttonWindow, text='Quit', command=quit).grid(row=10, column=5)
