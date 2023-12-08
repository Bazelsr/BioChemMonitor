import biochemDev
def startScan():
    # call startScan() from main thread
    asyncio.run_coroutine_threadsafe(scan(), loop)
def discover_devices(loop):
    """Returns the list of available bluetooth devices """
    biochem = biochemDev.Biochem("")
    asyncio.run_coroutine_threadsafe(biochem.discover_devices(),loop)
    pass