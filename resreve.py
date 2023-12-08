async def scan():
    devices = await BleakScanner.discover()
    for device in devices:
        print(device)
def startScan():
    # call startScan() from main thread
    asyncio.run_coroutine_threadsafe(scan(), loop)

if __name__ == "__main__":
    window = tk.Tk()
    # ...
    loop = asyncio.get_event_loop()
    def bleak_thread(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = Thread(target=bleak_thread, args=(loop,))
    t.start()
    btn = tk.Button(window,text="scan",command=lambda: startScan())
    btn.pack()
    window.mainloop()
    startScan()
    loop.call_soon_threadsafe(loop.stop)