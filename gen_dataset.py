import argparse
from rir_dataset_utils import sample_rooms
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a specific function.")    
    
    parser.add_argument('randseed', type=int,
                    help='random seed for room sampling')
    
    parser.add_argument('room_start', type=int,
                    help='start index of room')
    
    parser.add_argument('room_end', type=int,
                    help='end index of room')
    
    parser.add_argument('room_layout_num', type=int,
                    help='number of layouts per room')
    
    
    args = parser.parse_args()
   
    sample_rooms(randseed=1, room_start=args.room_start, room_end=args.room_end, room_layout_num=args.room_layout_num)