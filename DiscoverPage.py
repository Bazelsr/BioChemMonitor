import DiscoverPage_backend as backend
import tkinter as tk

IS_PAGE_INIT = False
DISCOVERED_DEVICES:list = []
CLICKED_DEVICE:tk.StringVar = None
CURRENT_LOOP = None
def init_page(master,loop):
    global WINDOW, CURRENT_LOOP
    global CLICKED_DEVICE ,IS_PAGE_INIT,DISCOVERED_DEVICES
    WINDOW = master
    CURRENT_LOOP = loop
    if(not IS_PAGE_INIT):
        CLICKED_DEVICE = tk.StringVar()
        DISCOVERED_DEVICES = [""]
        IS_PAGE_INIT = True

def on_click_scan():
    global DISCOVERED_DEVICES, WINDOW
    DISCOVERED_DEVICES = backend.discover_devices(CURRENT_LOOP)
    WINDOW.update()

def on_click_connect():
    global CLICKED_DEVICE ,WINDOW

    mac_add = CLICKED_DEVICE.get()
    res = discover.connect(mac_add)



def build_discover_page(master,loop):
    global CLICKED_DEVICE ,DISCOVERED_DEVICES
    init_page(master,loop)
    frame = tk.Frame(master)
    msg_box = tk.Message(frame, text="Scan and then connect to device:")
    drop = tk.OptionMenu(frame, CLICKED_DEVICE, *DISCOVERED_DEVICES)
    btn_scan = tk.Button(frame, text="Scan devices", command=lambda: on_click_scan())
    btn_connect = tk.Button(frame, text="Connect", command=lambda: on_click_connect())
    msg_box.grid(row=0, column=0)
    drop.grid(row=0, column=2)
    btn_scan.grid(row=2, column=1)
    btn_connect.grid(row=3, column=1)
    return frame
