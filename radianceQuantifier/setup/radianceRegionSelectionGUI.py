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
            # handle the checkbuttons #
            
            # error handling
            error_flag = False
            if (check8_var.get() == 1) & (selected_rect is None): # selected custom region, but didn't draw rectangle
                error_flag = True
                tk.messagebox.showinfo(title='ERROR', message='Please draw a custom region.')
            if (check1_var.get() == 0) & (check8_var.get() == 0): # no buttons selected
                error_flag = True
                tk.messagebox.showinfo(title='ERROR', message='Please select a button.')
            if (check8_var.get() == 1) & (selected_rect is not None) & (len(roi_entry.get()) == 0):
                error_flag = True
                tk.messagebox.showinfo(title='ERROR', message='Please enter a name for the selected region.')
            if (check8_var.get() == 1) & (selected_rect is not None) & (roi_entry.get().lower() == 'all'):
                error_flag = True
                tk.messagebox.showinfo(title='ERROR', message='Invalid custom name. Please enter a different name for the selected region.')

            # selected entire image
            finished_bool = False
            if (check1_var.get() == 1) & (error_flag == False): 
                calculate_radiance(left=0,right=maxWidth,top=0,bottom=maxHeight,text='all')
                finished_bool = True
            # selected custom region
            if (check8_var.get() == 1) & (selected_rect is not None) & (len(roi_entry.get()) > 0) & (error_flag == False):
                calculate_radiance(sel_left,sel_right,sel_top,sel_bottom,text=roi_entry.get())
                finished_bool = True
            
            if finished_bool == True: tk.messagebox.showinfo(title='Success', message='Radiance calculation complete!\nSelect another region or click "Finished" to return to main window.')


        # Display average image
        avg_image_path = f'plots/Image Processing/avg_merged_image-{os.getcwd().split(dirSep)[-1]}.png'
        avg_image_original = Image.open(avg_image_path)
        avg_image = avg_image_original.resize((maxWidth, maxHeight), Image.LANCZOS)  # Rescale image
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
        rect_coordinates_label.grid(row=5, column=0, sticky='nw')

        # Set up check buttons
        # initialize
        check1_var = tk.IntVar() # entire mouse
        check8_var = tk.IntVar() # custom region
        
        # make buttons
        check1 = tk.Checkbutton(mainWindow, text="Entire Image", variable=check1_var)
        check8 = tk.Checkbutton(mainWindow, text="Custom Region", variable=check8_var)
        roi_entry_label = tk.Label(mainWindow, text="ROI Name:")
        roi_entry = tk.Entry(mainWindow)
       
        # layout for buttons
        check1.grid(row=0, column=3, sticky='w')
        check8.grid(row=1, column=3, sticky='w')
        roi_entry_label.grid(row=1, column=4, sticky='w')
        roi_entry.grid(row=1, column=5, sticky='w')
        
        # Create a button to confirm the selection
        confirm_button = tk.Button(mainWindow, text="Calculate Radiance", command=(lambda : handle_checkbutton_region_selection()))
        confirm_button.grid(row=2, column=3, sticky='n')

        # Bind mouse events for rectangle drawing
        canvas.bind("<Button-1>", on_canvas_click)
        canvas.bind("<B1-Motion>", on_canvas_drag)

        # navigation buttons
        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=(tk.TOP), pady=10)
        tk.Button(buttonWindow, text='Finished', command=(lambda : master.switch_frame(originPage, selectedExperiment))).grid(row=10, column=1)
        tk.Button(buttonWindow, text='Quit', command=quit).grid(row=10, column=3)
