import asyncio
import threading
from datetime import datetime
import bleak
import re
import sys
from colorama import Fore, Back, Style
bold_start = '\033[1m'
bold_end = '\033[0m'
red_start = '\033[1m\033[91m'
color_end = '\033[1m\033[00m'
green_start = '\033[1m\033[92m'
yellow_start = '\033[93m'

class BLE:

    def __init__(self,mac):
        self.__client:bleak.BleakClient = bleak.BleakClient(mac)
        self.__discovered_devices = []

    def get_discovered_devices(self):
        return list(self.__discovered_devices)
    async def get_services(self):
        if self.__client is not None:
            services = self.__client.services
            if services is not None:
                print("Discovered services")
                print("================================================================")
                for service in services:
                    print(service)
                    print("------------------------------------------------------------------")
                    chars = service.characteristics
                    for char in chars:
                        print(f"|\t{str(char).ljust(100)}{char.properties}")
                    print("------------------------------------------------------------------")
                    print(" ")
                print("================================================================")
            else:
                print("Please connect to a Biochem device first using \"connect\" <MAC address> ")
        else:
            print("Please connect to a Biochem device first using \"connect\" <MAC address> ")


    async def connect_device(self,mac):
        is_mac = re.search("([0-9A-Fa-f]{2}:){5}([0-9A-Fa-f]{2})", mac)
        mac_address = None
        if is_mac is not None:
            mac_address = is_mac.string
        else:
            if self.__discovered_devices[int(mac)] is not None:
                mac_address = self.__discovered_devices[int(mac)].address
        if mac_address is not None:
            if(self.__client is not None):
                if(self.__client.is_connected): await self.__client.disconnect()
            self.__client = bleak.BleakClient(mac)
            await self.__client.connect()
            print("")
            print(f"connected to {self.__client.address}")
            print("")
            # services = client.services
            # for service in services:
            #    chars = service.characteristics
            #    for char in chars:
            #        print(char.properties)
            await self.get_services()

    async def scan_for_devices(self):
        devices = await bleak.BleakScanner.discover()
        index = 0
        for d in devices:
            if d.name is not None:
                if "BIOCHEM" in d.name or "Biochem" in d.name:
                    print(f"{index}. {d} ")
                    self.__discovered_devices.append(d)
        if len(self.__discovered_devices) == 0:
            print( "No compatible devices found!")

    async def disconnect(self):
       await self.__client.disconnect()

    def is_connected(self):
        return self.__client.is_connected

    async def start_notify(self,comm, notif_handler1):
        await self.__client.start_notify(comm,notif_handler1)
    async def stop_notify(self,charr_uuid):
        await self.__client.stop_notify(charr_uuid)

    async def read_gatt_char(self,charr):
        try:
            res = await self.__client.read_gatt_char(charr)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))
            return []
        return res

    async def write_gatt_char(self,charr, data):
        try:
            await self.__client.write_gatt_char(charr,data)
            return True
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))
            return False

