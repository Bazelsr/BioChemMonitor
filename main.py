"""Model controller"""
import time
from threading import Thread
import tkinter as tk
from bleak import BleakScanner
import asyncio
import DiscoverPage as dp

##STATICS
IS_PAGE_INIT = False
DISCOVERED_DEVICES = []
CLICKED_DEVICE:tk.StringVar = None
##STATIC GUIS
WINDOW:tk.Tk = None
CURRENT_FRAME:tk.Frame = None


def init_system():
    pass


def clear_parent_widget(parent):##Take form bioschool/biochem app
    pass


def build_patient_page(master):
    pass

def watch_dog(stringa):
    print(stringa)
    WINDOW.after(500,watch_dog,stringa)

def launch_app():
    global WINDOW,CURRENT_FRAME
    init_system()
    WINDOW = tk.Tk()
    loop = asyncio.get_event_loop()
    CURRENT_FRAME = dp.build_discover_page(WINDOW,loop)
    CURRENT_FRAME.pack()
    def bleak_thread(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = Thread(target=bleak_thread, args=(loop,))
    t.start()
    WINDOW.after(2000, watch_dog,"hi")
    WINDOW.mainloop()
    loop.call_soon_threadsafe(loop.stop)
    ##Enter loop

launch_app()
