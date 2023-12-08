import math
import sys

from bluetooth import BLE
import threading
import datetime
import errno
import os
import asyncio
from AsyncQueue import AsyncQueue
import numpy as np
import pandas as pd
import struct
from database import get_data_type_path as db_get_data_type_path
import Analysis.plotter as plt

handler_files_names = ["file1", "file2", "file3", "file4"]
EIS_MODE        = 0x01
CV_MODE         = 0x02
OPT_MODE        = 0x03
EEG_MODE        = 0x04
##################
START_ACQ       = 0X01
STOP_ACQ        = 0x00
STREAM_MODE     = 0X00
EXPERIMENT_MODE = 0x02
##################
BLOCK_SIZE           = 240
DATA_INFO_SIZE       = 4
PREDICTION_INFO_SIZE = 4
##################
EISCV_MAX_CH = 8
OPT_MAX_CH   = 7

CHAR1 = "0000fff1-0000-1000-8000-00805f9b34fb"
WORK_CHAR2 = "0000fff2-0000-1000-8000-00805f9b34fb"
ECO_DATA_CHAR3 = "0000fff3-0000-1000-8000-00805f9b34fb" #ECO stands for EIS, CV and Optics. This is their characteristic address.
CONT_DATA_CHAR4 = "0000fff4-0000-1000-8000-00805f9b34fb"
DEV_INFO_CHAR6 = "0000fff6-0000-1000-8000-00805f9b34fb"


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class Biochem:

    class BioChemDataBlock:

        def __init__(self,data):
            """Gets data as array of bytes (As retrieved from BioChem)"""
            self.__data:bytearray = data

        def set_data(self,data):
            self.__data = data
        def extends_data(self,new_data):
            self.__data.extend(new_data)

        def get_all_data_size(self):
            """In bytes..."""
            return len(self.__data)
        def get_experiment_type(self):
            return int(self.__data[0])

        def get_channel(self):
            return int(self.__data[1])

        def get_block_index(self):
            return int(self.__data[2])

        def get_total_blocks(self):
            return int(self.__data[3])

        def get_block_number_status(self):
            return (self.get_block_index(),self.get_total_blocks())

        def is_one_man_block(self):
            return (self.get_total_blocks() == 0)

        def get_block_size(self):
            """Data size in bytes"""
            if(not self.is_one_man_block()): return int(len(self.__data) - (DATA_INFO_SIZE)) #+ PREDICTION_INFO_SIZE)
            return int(self.__data[2])

        def is_last_block(self):
            #(self.get_total_blocks() == self.get_block_index()
            return self.is_one_man_block()

        def is_prediction(self):
            if(int(self.__data[0]) > 0xf0):
                return True
            return False
        def get_prediction(self):
            """#TODO"""
            pass

        def to_numpy(self):
            """Only converts the data and not the datat info onto numpy 1d array. Every 4 bytes is a number..."""
            if(self.is_prediction()):
                size = 2
            else: size = self.get_block_size()//4 #always shoudld work...
            res = np.zeros((size))
            for i in range(len(res)):
                baseind = DATA_INFO_SIZE + i*4
                bytes = bytearray([self.__data[baseind + 0],self.__data[baseind + 1],
                                   self.__data[baseind + 2],self.__data[baseind + 3]])
                array = struct.unpack('f', bytes)
                res[i] = array[0]
            return res

        def to_hex(self):
            return self.__data.hex()



    def __init__(self,mac,electrodes_num=8):
        self.__ble:BLE = BLE(mac)
        self.mac = mac
        self.__notify_flag = False
        self.__channel_data_size = 0
        self.__current_mod = 0x0
        self.__electrodes_num = electrodes_num
        self.__files_to_convert:AsyncQueue = AsyncQueue()
        self.__written_file = []
        self.__is_stream_mode_on = False
        self.__frequencies = self.create_frequencies_range(100,180000,50)
        self.__data_buffer = Biochem.BioChemDataBlock(bytearray([]))
        self.__volts = self.create_volts_range(0,100,100)
        self.__active_channels = 0xff
        self.__pred_counter = 0
        self.__fig = None
        self.__axEIS = None
        self.__axCV = None
        self.__axPred = None
        self.__axOPT = None
        self.__auto_plot = False
        #x = plt.subplots(nrows=2, ncols=2)
        #self.__fig = x[0]
        #self.__axEIS = x[1][0][0]
        #self.__axCV = x[1][0][1]
        #self.__axOPT = x[1][1][0]
        #self.__axPred = x[1][1][1]



    async def get_mac_address(self):
        return self.mac

    async def get_data_size(self):
        return self.__channel_data_size
    async def get_notify_flag(self):
        return self.__notify_flag


    async def get_services(self):
        await self.__ble.get_services()

    async def __get_current_mod(self):
        return self.__current_mod

    def __is_auto_plot(self):
        return self.__auto_plot

    async def __get_written_file(self):
        if(await self.__is_write_file()): return self.__written_file[0]
        else: return ""
    async def __is_write_file(self):
        return len(self.__written_file) == 1

    async def __get_is_stream_mode_on(self):
        return self.__is_stream_mode_on

    def get_frequencies(self):
        return list(self.__frequencies)

    async def set_write_file(self,filepath):
        if(await self.__is_write_file()):
            if(filepath == ""):self.__written_file.pop()
            else: self.__written_file[0] = filepath
        else: self.__written_file.append(filepath)

    async def set_mac_address(self,mac):
        self.mac = mac
    def set_channels(self,ch):
        self.__active_channels = ch

    async def __set_notify_flag(self,val:bool):
        self.__notify_flag = val

    async def __set_data_size(self,size):
        self.__channel_data_size = size

    def enable_auto_plot(self):
        self.__auto_plot = True
    def disable_auto_plot(self):
        self.__auto_plot = False
    async def __add_data_size(self,val):
        self.__channel_data_size += val

    def str_state(self,state:int):
        if(state == 1):
                return "EIS"
        elif(state == 2):
                return "CV"
        elif (state == 3):
                return "OPT"
        elif (state == 0xf1):
                return "EIS_Pr"#prediction
        elif (state == 0xf2):
                return "CV_Pr"#prediction
        elif (state == 0xf3):
                return "OPT_Pr"#prediction

    async def __turn_on_state_mod(self,state):
        """State represents bit number is mod of biochem..."""
        mod = await self.__get_current_mod()
        not_allowed_states = (0x1<<(EIS_MODE-1) + 0x1<<(CV_MODE-1) + 0x1<<(OPT_MODE-1))
        if(state & (not_allowed_states)):
            mod = mod ^ not_allowed_states
        mod = mod or state
        await self.__set_current_mod(mod)
        return True

    async def __turn_off_state_mod(self,state):
        mod = await self.__get_current_mod()
        mod = mod ^ state
        await self.__set_current_mod(mod)
        return True

    async def __set_current_mod(self,new_mod):
        self.__current_mod = new_mod

    async def connect(self):
       await self.__ble.connect_device(self.mac)

    async def discover_devices(self):
        res = await self.__ble.scan_for_devices()

    async def disconnect(self):
        await self.__ble.disconnect()

    def is_connected(self):
        return self.__ble.is_connected()

    def convert_str_to_byte_arr(self,line):
        return bytearray.fromhex(line)

    def create_frequencies_range(self,start,finish,n_points):
        arr = np.zeros(n_points)
        for i in range(1,n_points+1):
            arr[i-1] = math.pow(10,(((math.log(finish, 10) - math.log(start, 10)) / (n_points - 1)) * (i - 1)) + math.log(100, 10))
        return arr
    def create_volts_range(self,start,end,n_points):
        step = ((end-start)*2)//n_points
        res = np.zeros(n_points)
        for i in range(n_points//2):
            res[i] = start + i*step
        for i in range(n_points//2):
            res[i+n_points//2] = (end-step) - i*step
        return res

    def split_eis_data(self,data):
        """Gets numpy!"""
        mags = np.zeros(len(data)//2)
        phases = np.zeros(len(data)//2)
        for i in range(len(mags)):
            mags[i] = data[i*2]
            phases[i] = data[i*2 + 1]
        return mags,phases

    def read_biochem_data_file(self,file):
        channels = {}
        vals = []
        with open(file) as f:
            mylist = f.read().splitlines()
        for line in mylist:
            line = self.convert_str_to_byte_arr(line)
            b_data = Biochem.BioChemDataBlock(line)
            ch = str(b_data.get_channel())
            if(not ch in channels.keys()):
                channels[ch] = []
                if (b_data.get_experiment_type() == EIS_MODE):
                    channels[ch+"_phase"] = []
            if(b_data.get_experiment_type() == EIS_MODE):
                vals,phases = self.split_eis_data(b_data.to_numpy())
                channels[ch + "_phase"].extend(phases)
            else: vals = b_data.to_numpy()
            channels[ch].extend(vals)
        df = pd.DataFrame(channels)
        if(b_data.get_experiment_type() == EIS_MODE):
            df.index = self.__frequencies
            df.index.name = "frequencies"
        if(b_data.get_experiment_type() == CV_MODE):
            df.index = self.__volts
            df.index.name = "volts[mV]"
        return b_data.get_experiment_type(),df

    async def wait_until_finish(self,mode,limit_blocks_number):
        """#TODO: Change to end at last block and not limit the blocks number"""
        while(await self.get_notify_flag() and await self.get_data_size() < limit_blocks_number):
            await asyncio.sleep(1)


    async def convert_data_files_to_csv(self):
        while(not await self.__files_to_convert.is_empty()):
            try:
                file = await self.__files_to_convert.pop()
                exp_type,df = self.read_biochem_data_file(file)
                now = datetime.datetime.now()
                dir_name = now.strftime("%m_%d_%Y")
                save_path = db_get_data_type_path("raw",self.str_state(exp_type))
                save_path = os.path.join(save_path,dir_name)
                if(not os.path.exists(save_path)): mkdir_p(save_path)
                elif(not os.path.isdir(save_path)): os.mkdir(save_path)
                save_path = os.path.join(save_path,os.path.basename(file)[:-4]+".csv")
                df.to_csv(save_path)
                print(f"Wrote file {save_path}")
                return save_path

            except Exception as error:
                print(f"Houston, we got a problem...: biochemDev->convert_data_files_to_csv: {error}")
                return ""


    def draw_channel(self,yvals,ch_number,exp_type):
        if(exp_type == EIS_MODE):
            figs = []
            mags,phases = self.split_eis_data(yvals[:100])
            #freqs = self.create_frequencies_range(1000,1800000,len(mags))
            figs.append(plt.bk_plot(self.__frequencies, [mags], lines_names=[f"{exp_type}_ch{ch_number}_mags[Ohm]"], is_show=False, xlabel="Frequencies", ylabel="Mags"))
            figs.append(plt.bk_plot(self.__frequencies, [phases], lines_names=[f"{exp_type}_ch{ch_number}_phases"], is_show=False,
                                    xlabel="Frequencies", ylabel="Phases[Hz]"))

            grid_content = plt.make_grid_content(figs, col_num=1)
            content = plt.bk_gridplot(grid_content, width=550, height=450)
            plt.show(content)
        elif(exp_type == CV_MODE):
            yvals = yvals[:100]
            xpoints = self.create_volts_range(0,100,len(yvals))
            #xpoints = np.arange(len(yvals))
            plt.bk_plot(xpoints,[yvals],lines_names=[f"{exp_type}_ch{ch_number}"],xlabel="Voltage[mV]",ylabel="Current[uA]",is_show=True)
        elif(exp_type == OPT_MODE):
            xpoints = np.arange(len(yvals))
            plt.bokeh_plot_circle(xpoints[0], yvals[0],fig=None, fig_title="",color="red",is_show=True)
        else:
            xpoints = np.arange(len(yvals))
            plt.bk_plot([xpoints], [yvals], lines_names=[f"{exp_type}_ch{ch_number}"], xlabel="x-axis",
                        ylabel="y-axis", is_show=True)

    async def notif_ECO_handler(self,sender, data):
        print("Notified ECO Handler")
        b_data = Biochem.BioChemDataBlock(data)
        if(b_data.is_prediction()):
            print("============================================")
            pred = b_data.to_numpy()[1]
            #th = threading.Thread(target=self.scatter,
                                  #args=(self.__pred_counter,pred))
            #th.start()
            print(f"PREDICTION {self.str_state(b_data.get_experiment_type())} FOR CH {str(b_data.get_channel())}IS {pred}")
            self.__pred_counter += 1
            return
        ##If last and first block
        if(self.__data_buffer.get_all_data_size() == 0):
            if(b_data.is_last_block()):
                self.__data_buffer.extends_data(data[:b_data.get_block_size()+DATA_INFO_SIZE])
            else: self.__data_buffer.extends_data(data)
        else: self.__data_buffer.extends_data(data[DATA_INFO_SIZE:])
        exp_type = b_data.get_experiment_type()
        title = self.str_state(exp_type)
        print(f"Experiment {title} for channel {b_data.get_channel()}.\n Block number {b_data.get_block_index()} out of {b_data.get_total_blocks()}")
        print(f"Effective Block size: {b_data.get_block_size()}")
        if(exp_type == EIS_MODE):
            max_ch = EISCV_MAX_CH
        elif(exp_type == CV_MODE):  max_ch = EISCV_MAX_CH
        elif(exp_type == OPT_MODE):
            max_ch = OPT_MAX_CH
        else: max_ch = 0
        if(not await self.get_notify_flag()): return
        try:
            if(not await self.__is_write_file()):
                now = datetime.datetime.now()
                dir_name = now.strftime("%m_%d_%Y")
                datestr = now.strftime("%m_%d_%Y_%H_%M_%S")
                dir_path = os.path.join('Data', dir_name)
                if(os.path.exists(dir_path)):
                    if(not os.path.isdir(dir_path)): mkdir_p(dir_path)
                else: os.mkdir(dir_path)
                completepath = os.path.join(dir_path, f"{title}_{datestr}.txt")
                await self.set_write_file(completepath)
            else:
                completepath = await self.__get_written_file()
            #print (b_data.to_hex())
            f = open(completepath, "a")
            f.write(f"{b_data.to_hex()}\n")#HEREEE
            f.close()
            await self.__add_data_size(1)
            print(f"Received blocks for exp {title} ch {b_data.get_channel()} is {await self.get_data_size()}")
            if(await self.__get_is_stream_mode_on()):
                if(b_data.is_last_block()):
                    await self.__set_data_size(0)
                    print(f"The last block for channel {b_data.get_channel()}")
                    if(self.__is_auto_plot()):
                        th = threading.Thread(target=self.draw_channel, args=(self.__data_buffer.to_numpy(),b_data.get_channel(),exp_type))
                        th.start()
                    self.__data_buffer.set_data(bytearray([]))
                    if(b_data.get_channel()+1 == max_ch):
                        await self.__files_to_convert.insert(await self.__get_written_file())
                        await self.set_write_file("")
                        print(f"Finished receiving blocks for exp {title}")
                        print("##########")

        except Exception as error:
            print(f"error writing to {title} file:", error)

    async def notify_ECO_start(self,comm):
        try:
            if self.is_connected():
                await self.__ble.start_notify(comm, self.notif_ECO_handler)
                while await self.get_notify_flag():
                    await asyncio.sleep(1)
                    await self.convert_data_files_to_csv()
            return True
        except Exception as error:
            tb = sys.exc_info()[2]
            print("Houston, we got a problem...: biochemDev.py->class Biochem->notify_ECO_start: ",error.with_traceback(tb))
            return False


    def notify_ECO_thread(self,comm):
        #print(self.__frequencies)
        th = asyncio.new_event_loop()
        th.run_until_complete(self.notify_ECO_start(comm))


    async def start_notify_ECO_service(self):
        noti_thread = threading.Thread(target=self.notify_ECO_thread, args=(ECO_DATA_CHAR3,))
        noti_thread.start()
        return noti_thread
    async def stop_notify_ECO_service(self):
        await self.__ble.stop_notify(ECO_DATA_CHAR3)

    async def start_ECO_data(self,mode,channel_num):
        if(mode == 0x03): state = 0x10
        else: state = mode
        chars3data = await self.__ble.read_gatt_char(ECO_DATA_CHAR3)
        if(len(chars3data) == 0): print("Houston, we got a problem...: biochemDev.py->start_ECO_data: read empty data from bluetooth device")
        chars3data[0] = mode  # for eis mode
        chars3data[1] = int(channel_num)
        res = await self.__ble.write_gatt_char(ECO_DATA_CHAR3, chars3data)
        if(not res): print("Houston, we got a problem...: biochemDev.py->start_ECO_data: wcouldn't perform  write data to bluetooth device")
        else: await self.__turn_on_state_mod(state)

    async def cv(self,channel_num:int):
        await self.__set_notify_flag(True)
        th = await self.start_notify_ECO_service()
        await self.start_ECO_data(CV_MODE,channel_num)
        await self.wait_until_finish(CV_MODE,limit_blocks_number=15)##TODO:Should be data length instead of blocks_number
        await self.stop_notify_ECO_service()
        await self.__set_notify_flag(False)
        th.join()
        await self.__set_data_size(0)
        await self.__turn_off_state_mod(CV_MODE)
        print("CV data Received")

    async def eis(self,channel_num):
        await self.__set_notify_flag(True)
        await self.start_notify_ECO_service()
        await self.start_ECO_data(EIS_MODE, channel_num)
        await self.wait_until_finish(EIS_MODE,
                                     limit_blocks_number=10000)  ##TODO:Should be data length instead of blocks_number
        await self.__set_notify_flag(False)
        await self.__turn_off_state_mod(EIS_MODE)
        print("EIS data Received")

    async def optics(self,channel_num):
        await self.__set_notify_flag(True)
        await self.start_notify_ECO_service()
        await self.start_ECO_data(OPT_MODE, channel_num)
        await self.wait_until_finish(OPT_MODE,
                                     limit_blocks_number=1)  ##TODO:Should be data length instead of blocks_number
        await self.__set_notify_flag(False)
        await self.__turn_off_state_mod(OPT_MODE)
        print("Optical data Received")

    async def cv_all(self):
        print("CV_all")
        for i in range(self.__electrodes_num):
            await self.cv(i)

    async def eis_all(self):
        print("eis all")
        for i in range(self.__electrodes_num):
            await self.eis(i)

    async def optics_all(self):
        print("optics_all")
        for i in range(self.__electrodes_num):
            await self.optics(i)
    ###Handlers


    ##TODO: NEEDS TEST
    async def activate_stream_mode(self):
        char1data = await self.__ble.read_gatt_char(CHAR1)
        #char2data[0] = 0x01
        #char1data[1] = 1<<self.__active_channels
        #await self.__ble.write_gatt_char(WORK_CHAR2,char2data)
        await self.__set_notify_flag(True)
        th = await self.start_notify_ECO_service()
        self.__is_stream_mode_on = True
        return th

    ##TODO: NEEDS TEST
    async def deactive_stream_mode(self):
        char1data = await self.__ble.read_gatt_char(CHAR1)
        #char1data[2] = self.__active_channels
        #char1data[1] = EXPERIMENT_MODE
        #await self.__ble.write_gatt_char(WORK_CHAR2, char1data)
        await self.__set_notify_flag(False)
        th = await self.stop_notify_ECO_service()
        self.__is_stream_mode_on = False
        return th



    ##TODO: NEEDS TEST
    async def enable_eeg(self):
        # write char 2 to start EEG acquisition
        chars2data = await self.__ble.read_gatt_char(WORK_CHAR2)
        chars2data[0] = START_ACQ  # start data acquisition
        await self.__ble.write_gatt_char(WORK_CHAR2, chars2data)
        await self.__turn_on_state_mod(EEG_MODE)

    async def disable_eeg(self):
        chars2data = await self.__ble.read_gatt_char(WORK_CHAR2)
        chars2data[0] = 0x0  # start data acquisition
        await self.__ble.write_gatt_char(WORK_CHAR2, chars2data)
        await self.__turn_off_state_mod(EEG_MODE)

    ##TODO: NEEDS TEST
    async def continuous_data_handler(self,sender, data):
        now = datetime.datetime.now()
        dir_name = now.strftime("%m_%d_%Y")
        datestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        dir_path = os.path.join('Data/', dir_name)
        mkdir_p(dir_path)

        completepath = os.path.join(dir_path + '/', f"continuous_data.txt")
        f = open(completepath, "a")
        f.write(str(bytes(data))+"\n")
        f.close()

    async def device_events_handler(self, sender, data):
        now = datetime.datetime.now()
        dir_name = now.strftime("%m_%d_%Y")
        datestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        dir_path = os.path.join('Data/', dir_name)
        mkdir_p(dir_path)

        completepath = os.path.join(dir_path + '/', f"device_events.txt")
        f = open(completepath, "a")
        f.write(str(bytes(data))+"\n")
        f.close()

    ##TODO: NEEDS TEST
    async def notify_continuous_data(self,char_uuid):
        try:
            if self.is_connected():
                await self.__ble.start_notify(char_uuid, self.continuous_data_handler)
                while self.is_connected() and self.get_notify_flag():
                    await asyncio.sleep(1)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))

    ##TODO: NEEDS TEST
    async def notify_device_events(self,char_uuid):
        try:
            if self.is_connected():
                await self.__ble.start_notify(char_uuid, self.device_events_handler)
                while self.is_connected():
                    await asyncio.sleep(1)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))


    def notify_continuous_data_thread(self,char_uuid):
        th = asyncio.new_event_loop()
        th.run_until_complete(self.notify_continuous_data(char_uuid))

    ##TODO: NEEDS TEST
    def notify_device_events_thread(self,char_uuid):
        th = asyncio.new_event_loop()
        th.run_until_complete(self.notify_device_events(char_uuid))


    ##TODO: NEEDS TEST
    async def start_notify(self,char_uuid,target):
        noti_thread = threading.Thread(target=target, args=(char_uuid,))
        noti_thread.start()

    ##TODO: NEEDS TEST
    async def enable_device_info_notify(self):
        await self.start_notify(CONT_DATA_CHAR4,target= self.notify_continuous_data_thread)
        await self.start_notify(DEV_INFO_CHAR6,target= self.notify_device_events_thread)
        while(True):
            await asyncio.sleep(1)
        return True

    async def notif_handler1(self,sender, data):
        print(f"Notified {data}")
        try:
            now = datetime.datetime.now()
            dir_name = now.strftime("%m_%d_%Y")
            datestr = now.strftime("%m_%d_%Y_%H_%M_%S")
            dir_path = os.path.join('Data/', dir_name)
            mkdir_p(dir_path)

            completepath = os.path.join(dir_path + '/', f"{handler_files_names[0]}.txt")
            f = open(completepath, "a")
            if "Evoked_Data" in handler_files_names[0]:

                if(data[0] == 1):
                    f.write(f"EIS channel {str(data[1])} {datestr}\n")
                    #if (data[1][244] == "0"): self.set_eis_notif_flag(False)
                if(data[0] == 2):
                    f.write(f"CV channel {str(data[1])} {datestr}\n")
                    #if (data[1][244] == "0"): self.set_cv_notif_flag(False)
                if(data[0] == 3):
                    f.write(f"OPT channel {str(data[1])} {datestr}\n")
                    #if (data[1][244] == "0"): self.set_optics_notif_flag(False)

            f.write(f"{data.hex()}\n")
            f.close()
        except Exception as error:
            print(f"error writing to {handler_files_names[0]} file:", error)


    async def notify1_cv_start(self,comm):
        try:
            if self.is_connected():
                await self.__ble.start_notify(comm, self.notif_handler1)
                while self.is_connected() and await self.get_notify_flag():
                    await asyncio.sleep(1)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))

    async def notify1_eis_start(self,comm):
        try:
            if self.is_connected():
                await self.__ble.start_notify(comm, self.notif_handler1)
                while self.is_connected() and await self.get_notify_flag():
                    await asyncio.sleep(1)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))

    async def notify1_optics_start(self,comm):
        try:
            if self.is_connected():
                await self.__ble.start_notify(comm, self.notif_handler1)
                while self.is_connected() and await self.get_notify_flag():
                    await asyncio.sleep(1)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))



    def notify1_thread(self,comm):
        th = asyncio.new_event_loop()
        th.run_until_complete(self.notify1_start(comm))#ask Jameel about it!


    async def notif_handler2(self,sender, data):
        now = datetime.now()
        dir_name = now.strftime("%m_%d_%Y")
        datestr = now.strftime("%m_%d_%Y_%H_%M_%S")
        dir_path = os.path.join('Data/', dir_name)
        mkdir_p(dir_path)

        completepath = os.path.join(dir_path + '/', f"{handler_files_names[1]}.txt")
        f = open(completepath, "a")
        f.write(bytes(data))
        f.close()

    async def notify2_start(self,comm):
        await self.__ble.start_notify(comm, self.notif_handler2)
        while self.is_connected():
            await asyncio.sleep(1)


    def notify2_thread(self,comm):
        th = asyncio.new_event_loop()
        th.run_until_complete(self.notify2_start(comm))



    async def notif_handler3(self,sender, data):
        try:
            now = datetime.now()
            dir_name = now.strftime("%m_%d_%Y")
            datestr = now.strftime("%m_%d_%Y_%H_%M_%S")
            dir_path = os.path.join('Data/', dir_name)
            mkdir_p(dir_path)

            completepath = os.path.join(dir_path + '/', f"{handler_files_names[2]}.txt")
            f = open(completepath, "a")
            f.write(f"{data.hex()}\n")
            f.close()
        except Exception as error:
            print(f"error writing to {handler_files_names[2]} file:", error)

    async def notify3_start(self,comm):
        try:
            if self.is_connected():
                await self.__ble.start_notify(comm, self.notif_handler3)
                while self.is_connected():
                    await asyncio.sleep(1)
        except Exception as error:
            tb = sys.exc_info()[2]
            print(error.with_traceback(tb))


    def notify3_thread(self,comm):
        th = asyncio.new_event_loop()
        th.run_until_complete(self.notify3_start(comm))

    async def notify_char(self,char_uuid, handler_index, file_store_name):
        if handler_index is not None:
            hi = int(handler_index)
            if(hi == 1):
                if file_store_name is not None:
                    handler_files_names[int(handler_index) - 1] = file_store_name
                    noti_thread = threading.Thread(target=self.notify1_thread, args=(char_uuid,))
                    noti_thread.start()
            if(hi == 2):
                if file_store_name is not None:
                    handler_files_names[int(handler_index) - 1] = file_store_name
                    noti_thread = threading.Thread(target=self.notify2_thread, args=(char_uuid,))
                    noti_thread.start()
            if(hi == 3):
                if file_store_name is not None:
                    handler_files_names[int(handler_index) - 1] = file_store_name
                    noti_thread = threading.Thread(target=self.notify3_thread, args=(char_uuid,))
                    noti_thread.start()
