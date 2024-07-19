import argparse
from rir_dataset_utils import sample_rooms_parallel, initiate_csv_file
from multiprocessing import Pool


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


    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run a specific function.")    
    
    # parser.add_argument('randseed', type=int,
    #                 help='random seed for room sampling')
    
    # parser.add_argument('room_start', type=int,
    #                 help='start index of room')
    
    # parser.add_argument('room_end', type=int,
    #                 help='end index of room')
    
    # parser.add_argument('room_layout_num', type=int,
    #                 help='number of layouts per room')
    
    
    # args = parser.parse_args()
    
    initiate_csv_file()
    
    # tasks = [(i+1, 30*i, 30*(i+1), 10, True, DATASET_PATH) for i in range(32)]
    tasks = [(i+1, 10*i, 10*(i+1), 10, True, DATASET_PATH) for i in range(32)]

    
    with Pool(processes=32) as pool:  # Adjust the number of processes based on your CPU
        results = pool.map(sample_rooms_parallel, tasks)
    
    # sample_rooms(randseed=1, room_start=args.room_start, room_end=args.room_end, room_layout_num=args.room_layout_num)