import os
from typing_extensions import Self
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import re
class Feeder_ODHG(Dataset):
    """
    Feeder for skeleton-based gesture recognition in shrec21-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(
            self,
            data_path="data/ODHG2016",
            set_name="training",
            window_size=10,
            aug_by_sw=False,
            is_segmented=False,
            binary_classes=False,
            num_joint=22
    ):
        self.data_path = data_path
        self.set_name = set_name
        self.classes = [
                        "NO GESTURE",
                        "RIGHT",
                        "KNOB",
                        "CROSS",
                        "THREE",
                        "V",
                        "ONE",
                        "FOUR",
                        "GRAB",
                        "DENY",
                        "MENU",
                        "CIRCLE",
                        "TAP",
                        "PINCH",
                        "LEFT",
                        "TWO",
                        "OK",
                        "EXPAND",
                        
                        ]
        self.class_to_idx = {class_l: idx for idx,
                             class_l in enumerate(self.classes)}
        self.window_size = window_size
        self.aug_by_sw = aug_by_sw
        self.num_joint=num_joint
        self.is_segmented = is_segmented
        self.binary_classes=binary_classes
        self.load_data()

    def load_data(self):
        self.dataset = []
        info_files=list(filter(lambda s : "_infos_sequences.txt" in s, os.listdir(self.data_path)))
        for info in info_files :
            sub_name=info.split('_infos_sequences.txt')[0]
            lines=[]
            
            
            
            with open(f'{self.data_path}/{info}',
                            mode="r") as f:
                for line in f :
                    lines.append(line)
            
            num_sequences=len(lines) // 3

            for seq_idx in range(num_sequences):
                labels=re.sub(' +', ' ', lines[seq_idx*3]).split(" ")
                frames_start_end=re.sub(' +', ' ', lines[seq_idx*3+2]).split(" ")
                gesture_infos = []
                for g_idx in range(len(labels)): 
                    gesture_start=frames_start_end[g_idx*2]
                    gesture_end=frames_start_end[g_idx*2+1]
                    gesture_label=labels[g_idx]
                    gesture_infos.append(
                                            (gesture_start, gesture_end, int(gesture_label)))
                self.dataset.append((sub_name,f"sequence_{seq_idx+1}",gesture_infos))
        self.dataset= self.dataset[:int(.7*len(self.dataset))] if self.set_name=="training" else self.dataset[int(.7*len(self.dataset)):]



    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        def parse_seq_data(src_file):
            '''
            Retrieves the skeletons sequence for each gesture
            '''
            video = []
            for line in src_file:
                line = line.split("\n")[0]
                data=re.sub(' +', ' ', line.strip()).split(" ")

                frame = []
                point = []
                for data_ele in data:
                    point.append(float(data_ele))
                    if len(point) == 3:
                        frame.append(np.array(point))
                        point = []
                frame=np.array(frame)
                video.append(frame)
            return np.array(video)


        def sample_window(data_num, stride):
            # sample #window_size frames from whole video

            sample_size = self.window_size

            idx_list = [0, data_num - 1]
            for i in range(sample_size):
                if index not in idx_list and index < data_num:
                    idx_list.append(index)
            idx_list.sort()

            while len(idx_list) < sample_size:
                idx = random.randint(0, data_num - 1)
                if idx not in idx_list:
                    idx_list.append(idx)
            idx_list.sort()
            return idx_list
        # output shape (C, T, V, M)
        # get data

        def get_segmented(sub_name, seq_idx, gesture_infos):
            with open(f'{self.data_path}/{sub_name}/{seq_idx}/skeletons_world_enhanced.txt', mode="r") as seq_f:
                sequence = parse_seq_data(seq_f)
            labeled_sequence = [(f, 0) for f in sequence]
            for gesture_start, gesture_end, gesture_label in gesture_infos:
                labeled_sequence = [
                    (np.array(f), gesture_label if int(gesture_start) <=
                     idx <= int(gesture_end) and int(label) == 0 else label)
                    for
                    idx, (f, label) in enumerate(labeled_sequence)]

            frames = [f for f, l in labeled_sequence]
            labels_per_frame = [ (1 if l!=0 else 0)  if self.binary_classes else l
                                for f, l in labeled_sequence]
            
            gestures = []
            windows_sub_sequences_per_gesture = {
                i: [] for i in range(len(self.classes))}

            for gesture_start, gesture_end, gesture_label in gesture_infos:
                gesture_start = int(gesture_start)
                gesture_end = int(gesture_end)
                g_frames = frames[gesture_start:gesture_end]
                g_label = labels_per_frame[gesture_start:gesture_end]
                gestures.append((g_frames, g_label))
                label = gesture_label
                if self.aug_by_sw:
                    num_windows = len(g_frames) // self.window_size

                    for stride in range(1, self.window_size):
                        l = len(g_frames)
                        if l // stride >= self.window_size:
                            window_indices = sample_window(l, stride)
                            window = [g_frames[idx] for idx in window_indices]
                            windows_sub_sequences_per_gesture[int(label)].append(
                                (window, int(label)))

            ng_sequences = []
            ng_seq = []
            l = len(frames)
            indices_ng = []
            for i in range(len(frames)-1):
                f_curr = frames[i]
                f_next = frames[i+1]
                l_curr = labels_per_frame[i]
                l_next = labels_per_frame[i+1]

                if l_curr == 0 and l_next == 0:
                    indices_ng.append(i)
                    ng_seq.append(f_curr)
                    if i == l-2:
                        ng_seq.append(f_next)
                        ng_sequences.append((ng_seq, 0))
                        ng_seq = []
                        continue
                elif l_curr == 0 and l_next != 0:
                    indices_ng.append(i)
                    ng_seq.append(f_curr)
                    ng_sequences.append((ng_seq, 0))
                    ng_seq = []
                    continue
            # max_l=0
            # for k in windows_sub_sequences_per_gesture.keys() :
            #     max_l= len(windows_sub_sequences_per_gesture[k]) if len(windows_sub_sequences_per_gesture[k]) > max_l else max_l
            # for k in windows_sub_sequences_per_gesture.keys():
            #     while len(windows_sub_sequences_per_gesture[k]) < max_l :
                    
            #         windows_sub_sequences_per_gesture[k]=[*windows_sub_sequences_per_gesture[k],*windows_sub_sequences_per_gesture[k]]
            #     windows_sub_sequences_per_gesture[k]=windows_sub_sequences_per_gesture[k][:max_l]
            # gestures=[]
            # for k in windows_sub_sequences_per_gesture.keys() :
            #     gestures=[*gestures,*windows_sub_sequences_per_gesture[k]]
            return gestures, ng_sequences, windows_sub_sequences_per_gesture

        def get_full_sequences(sub_name, seq_idx, gesture_infos):
            with open(f'{self.data_path}/{sub_name}/{seq_idx}/skeletons_world_enhanced.txt', mode="r") as seq_f:
                sequence = parse_seq_data(seq_f)
            labeled_sequence = [(f, 0) for f in sequence]
            
            for gesture_start, gesture_end, gesture_label in gesture_infos:
                labeled_sequence = [
                    (np.array(f), gesture_label if int(gesture_start) <=
                     idx <= int(gesture_end) and int(label) == 0 else label)
                    for
                    idx, (f, label) in enumerate(labeled_sequence)]

            frames = [f for f, l in labeled_sequence]

            labels_per_frame = [ (1 if self.class_to_idx[l]!=0 else 0)  if self.binary_classes else l
                                for f, l in labeled_sequence]
            
            return labeled_sequence, np.array(frames), labels_per_frame
        sub_name, seq_idx, gesture_infos = self.dataset[index]

        if self.is_segmented:
            return get_segmented(sub_name, seq_idx, gesture_infos)
        else:
            return get_full_sequences(sub_name, seq_idx, gesture_infos)


def get_window_label(label, num_classes=15):

    W = len(label)
    sum = torch.zeros((num_classes))
    for t in range(W):
        sum[label[t]] += 1
    return sum.argmax(dim=-1).item()


def gendata(
        data_path,
        set_name,
        max_frame,
        window_size=20,
        num_joint=22,
        aug_by_sw=False,
        is_segmented=False,
        binary_classes=False,
        
):
    feeder = Feeder_ODHG(
        data_path=data_path,
        set_name=set_name,
        window_size=window_size,
        aug_by_sw=aug_by_sw,
        num_joint=num_joint,
        is_segmented=is_segmented,
        binary_classes=binary_classes
    )
    dataset = feeder.dataset
    if is_segmented:
        data = []
        ng_sequences_data = []
        windows_sub_sequences_data = {i: []
                                      for i in range(len(feeder.classes))}
        for i, s in enumerate(tqdm(dataset)):
            data_el, ng_sequences, windows_sub_sequences_per_gesture = feeder[i]
            ng_sequences_data = [*ng_sequences_data, *ng_sequences]
            l = len(data_el)
            # for w in range(num_windows):
            for idx, gesture in enumerate(data_el):
                current_skeletons_window = np.array(gesture[0])
                label = gesture[1]
                label = get_window_label(label)
                windows_sub_sequences_data[label] = [
                    *windows_sub_sequences_data[label], *windows_sub_sequences_per_gesture[label]]
                for sub_g in windows_sub_sequences_per_gesture[label] :
                    data.append(sub_g)
                data.append((current_skeletons_window,label))    

        return data, ng_sequences_data, windows_sub_sequences_data
    else:
        data = []
        for i, s in enumerate(tqdm(dataset)):
            labeled_seq, frames, labels = feeder[i]
            data.append((frames, labels))

        return data