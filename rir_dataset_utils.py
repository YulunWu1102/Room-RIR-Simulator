import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import random
from scipy.io.wavfile import read, write
import csv
import pandas as pd
from tqdm import tqdm
import os 


RT60_MIN = 0
RT60_MAX = 0.2
ROOM_DIAMETER_RANGE_MIN = 1
ROOM_DIAMETER_RANGE_MAX = 5
MIC_NUM = 8
MIC_RADIUS = 0.1
SOURCE_NUM = 8
ROOM_NUM = 5
ROOM_LAYOUT_NUM = 10
FS = 16000
DATASET_PATH = "dataset_rir"



class RIR_Generator():
    def __init__(self, randseed=1, rt60_min=RT60_MIN, rt60_max=RT60_MAX, room_diameter_range_min=ROOM_DIAMETER_RANGE_MIN, 
                room_diameter_range_max=ROOM_DIAMETER_RANGE_MAX, mic_num=MIC_NUM, mic_radius=MIC_RADIUS, 
                source_num=SOURCE_NUM, room_layout_num=ROOM_LAYOUT_NUM, fs=FS):
        self.rt60_min = rt60_min
        self.rt60_max = rt60_max
        self.room_diameter_range_min = room_diameter_range_min
        self.room_diameter_range_max = room_diameter_range_max
        self.mic_num = mic_num
        self.mic_radius = mic_radius
        self.source_num = source_num
        self.room_layout_num = room_layout_num
        self.fs = fs
        self.rng = np.random.default_rng(randseed)
    
    
    # ================== helpers ==================:
    def location_3d_in_range(self, range_array, num_location, minimum=0):
        locations = []
        for _ in range(num_location):
            curr_loc = np.array([self.rng.uniform(low=minimum, high=range_array[0]), 
                                 self.rng.uniform(low=minimum, high=range_array[1]), 
                                 self.rng.uniform(low=minimum, high=range_array[2])])
            locations.append(curr_loc)
        return locations


    def truncate_rir(self, room_rir, num_source, num_mic):
        rir_len_lst = []
        for i in range(num_mic):
            for j in range(num_source):
                rir_len_lst.append(len(room_rir[i][j]))
                
        min_rir_len = min(rir_len_lst)
        for i in range(num_mic):
            for j in range(num_source):
                room_rir[i][j] = room_rir[i][j][:min_rir_len]
                
        return np.vstack(np.array(room_rir))   

    
    def initiate_csv_file(self, csv_file_name="dataset_rir/rir_lookup.csv"):
        headers = [
            "room_idx",
            "room_layout_idx",
            "source_locations",
            "mic_location"
        ]
    
        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    
    
    def write_csv_data(self, room_idx, room_layout_idx, csv_file_name="dataset_rir/rir_lookup.csv"):
        data = [[room_idx, room_layout_idx, self.source_locations, self.mic_center]]
        with open(csv_file_name, 'a') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    
    
    def plot_room(self, room):
        fig, ax = room.plot()
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 6])
        ax.set_zlim([0, 6]);
        
    
    # ================== room construction ==================:
    def construct_room_basic(self, show_plot=False):
        self.room_dim =  np.array([self.rng.uniform(low=self.room_diameter_range_min, high=self.room_diameter_range_max) for _ in range(3)])
        while True:
            try:
                e_absorption, max_order = pra.inverse_sabine(self.rng.uniform(low=self.rt60_min, high=self.rt60_max) , self.room_dim)
                break # if it worked then just break out of the loop
            except ValueError:
                continue
            
        self.room = pra.ShoeBox(self.room_dim, fs=self.fs, materials=pra.Material(e_absorption), max_order=max_order)
        
        if show_plot:
            self.plot_room(self.room)


    def add_mic_and_sources(self, show_plot=False):
        # add mic array
        self.mic_center =  self.location_3d_in_range(self.room.shoebox_dim, 1, minimum=self.mic_radius)[0]
        R = pra.circular_2D_array(center=self.mic_center[:2], M=self.mic_num, phi0=0, radius=self.mic_radius)
        R = np.concatenate((R, np.ones((1, self.mic_num)) * self.mic_center[2]), axis=0)
        mics = pra.MicrophoneArray(R, self.fs)
        self.room.add_microphone_array(mics)
        
        # add sources
        self.source_locations = self.location_3d_in_range(self.room.shoebox_dim, self.source_num, minimum=0)
        for i in range(self.source_num):
            self.room.add_source(self.source_locations[i], delay=0)

        if show_plot:
            self.plot_room(self.room)
        
    
    
    def remove_mic_and_sources(self, show_plot=False):
        for i in range(self.source_num):
            del self.room.sources[0]
        
        self.room.mic_array = None               
    
        if show_plot:
            self.plot_room(self.room)
    
    
    def compute_and_save_room_rir(self, room_idx, room_layout_idx, show_plot=False):        
        self.room.image_source_model()
        self.room.compute_rir()
        
        if show_plot:
            fig, ax = self.room.plot(img_order=3)
            fig.set_size_inches(18.5, 10.5)
            self.room.plot_rir()
            fig = plt.gcf()
            fig.set_size_inches(20, 10)
         
        # save room rir to wav
        room_rir_arr = self.truncate_rir(self.room.rir, self.source_num, self.mic_num)
        room_name = str(room_idx) + '_' + str(room_layout_idx)
        rir_name = 'dataset_rir/' + str(room_idx) + '/' + room_name + '.wav'
        write(rir_name, self.fs, room_rir_arr.T)
        
        # save room data to csv        
        self.write_csv_data(room_idx, room_layout_idx) 




class RIR_Loader():
    def __init__(self, dataset_path, csv_path):
        self.dataset_path = dataset_path
        self.csv_path = csv_path
    
    
    def get_layout_row(self, df, room_idx, room_layout_idx):
        # Find rows where both conditions are met
        condition = (df['room_idx'] == room_idx) & (df['room_layout_idx'] == room_layout_idx)
        # Get the index of the row
        indices = df.index[condition]
        return indices.tolist()  # Convert index object to a list
    
    
    def get_room_layout(self, room_idx, room_layout_idx):
        room_layout_dict = {}
        csv_name = self.csv_path + "/" + "rir_lookup.csv"
        df = pd.read_csv(csv_name)
        layout_row = self.get_layout_row(df, room_idx, room_layout_idx)[0]
        
        # get source locations
        source_locations = df['source_locations'].iloc[layout_row]
        source_locations = eval(source_locations, {'array': np.array})
        room_layout_dict["source_locations"] = source_locations
        
        # get mic location
        mic_location = source_locations = df['mic_location'].iloc[0]
        mic_location = np.fromstring(mic_location.strip("[]"), sep=' ')
        room_layout_dict["mic_location"] = mic_location
        
        return room_layout_dict
    
    def get_room_rir(self, room_idx, room_layout_idx):
        wav_name = self.dataset_path + "/" + str(room_idx) + "/" + str(room_idx) +  "_" + str(room_layout_idx) + ".wav"
        fs, rir = wavfile.read(wav_name)
        return rir
    
    
# Additional Helper Functions

def initiate_csv_file(csv_file_name="dataset_rir/rir_lookup.csv"):
    headers = [
        "room_idx",
        "room_layout_idx",
        "source_locations",
        "mic_location"
    ]

    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        
    
def sample_room_layouts(rir_generator, room_idx, room_layout_num = ROOM_LAYOUT_NUM):
    rir_generator.construct_room_basic()
    desc = "processing room " + str(room_idx) + "..."
    # for i in tqdm(range(room_layout_num), desc=desc):
    for i in range(room_layout_num):
        rir_generator.add_mic_and_sources()
        rir_generator.compute_and_save_room_rir(room_idx, i)
        rir_generator.remove_mic_and_sources()
        
        
def sample_rooms(randseed, room_start, room_end, room_layout_num=ROOM_LAYOUT_NUM, is_csv_exist=False, dataset_path=DATASET_PATH):
    rir_generator = RIR_Generator(randseed=randseed)
    if not is_csv_exist:
        rir_generator.initiate_csv_file()
    
    for i in range(room_start, room_end):
        dir_name = dataset_path + "/" + str(i)
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            print("already exist")
            
        sample_room_layouts(rir_generator, i, room_layout_num)
        

def sample_rooms_parallel(args):
    randseed, room_start, room_end, room_layout_num, is_csv_exist, dataset_path = args
    rir_generator = RIR_Generator(randseed=randseed)
    if not is_csv_exist:
        rir_generator.initiate_csv_file()
    
    desc = "thread " + str(room_start) + " - " + str(room_end-1)
    for i in tqdm(range(room_start, room_end), desc=desc):
        dir_name = dataset_path + "/" + str(i)
        try:
            os.mkdir(dir_name)
        except FileExistsError:
            print("already exist")
            
        sample_room_layouts(rir_generator, i, room_layout_num)