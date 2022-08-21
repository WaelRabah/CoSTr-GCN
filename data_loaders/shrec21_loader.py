import os
import torch
import yaml
import pickle
import yaml
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


from typing import List, Tuple



import random



def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return (
        data_numpy.reshape(C, T / step, step, V, M)
        .transpose((0, 1, 3, 2, 4))
        .reshape(C, T / step, V, step * M)
    )


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    T, V, C = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((size, V, C))
        data_numpy_paded[begin : begin + T, :,:] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M 随机选择其中一段，不是很合理。因为有0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin : begin + size, :, :]


def random_move(
    data_numpy,
    angle_candidate=[-10.0, -5.0, 0.0, 5.0, 10.0],
    scale_candidate=[0.9, 1.0, 1.1],
    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate=[1],
):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i] : node[i + 1]] = (
            np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        )
        s[node[i] : node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i] : node[i + 1]] = np.linspace(
            T_x[i], T_x[i + 1], node[i + 1] - node[i]
        )
        t_y[node[i] : node[i + 1]] = np.linspace(
            T_y[i], T_y[i + 1], node[i + 1] - node[i]
        )

    theta = np.array(
        [[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]]
    )  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M 偏移其中一段
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias : bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert C == 3
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0 : T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0 : T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = rank == m
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert np.all(forward_map >= 0)

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(
            1, 2, 0
        )
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    _, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)  # noqa: E741
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, inward: List[Tuple[int, int]], num_node: int):
        self.num_node = num_node
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = inward
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        self.A = get_spatial_graph(
            self.num_node, self.self_link, self.inward, self.outward
        )

    def print(self, image=False):

        if image:
            import matplotlib.pyplot as plt

            for i in self.A:
                plt.imshow(i, cmap="gray")
                plt.show()


num_joint = 20
max_frame = 600

class Feeder_SHREC21(Dataset):
    """
    Feeder for skeleton-based gesture recognition in shrec21-skeleton dataset
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(
            self,
            data_path="SHREC21",
            set_name="training"
    ):
        self.data_path = data_path
        self.set_name = set_name

        self.load_data()


    def load_data(self):
        self.dataset = []
        # load file list
        classes = set([''])
        self.classes = []
        with open(
                f'{self.data_path}/{self.set_name}_set/annotations_revised.txt' if self.set_name == "test" else f'{self.data_path}/{self.set_name}_set/annotations_revised_{self.set_name}.txt',
                mode="r") as f:

            for line in f:
                fields = line.split(';')
                seq_idx = fields[0]
                gestures = fields[1:-1]
                nb_gestures = len(gestures) // 3
                gesture_infos = []
                for i in range(nb_gestures):
                    gesture_info = gestures[i * 3:(i + 1) * 3]
                    gesture_label = gesture_info[0]
                    gesture_start = gesture_info[1]
                    gesture_end = gesture_info[2]
                    gesture_infos.append((gesture_start, gesture_end, gesture_label))
                    classes.add(gesture_label)
                self.dataset.append((seq_idx, gesture_infos))

        self.classes = list(classes)
        # with open('datasets/shrec21/classes.yaml', mode="w") as f:
        #     yaml.dump(self.classes, f, explicit_start=True, default_flow_style=False)

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
            mode="pos"
            for line in src_file:

                line = line.split("\n")[0]

                data = line.split(";")[:-1]

                frame = []
                point = []
                for data_ele in data:
                    if len(data_ele)==0:
                        continue
                    point.append(float(data_ele))

                    if len(point) == 3 and mode=="pos":
                        frame.append(point)
                        point = []
                        mode="quat"
                    elif len(point) == 4 and mode=="quat":
                        frame.append(point)
                        point = []
                        mode="pos"
                if len(frame) > 0 :
                    positions   =[]
                    quats=[]

                    for i in range(num_joint):
                        positions.append(frame[i*2])
                        quats.append(frame[i*2+1])

                    video.append(positions)
            return np.array(video)

        # output shape (C, T, V, M)
        # get data
        seq_idx, gesture_infos = self.dataset[index]
        with open(f'{self.data_path}/{self.set_name}_set/sequences/{seq_idx}.txt', mode="r") as seq_f:
            sequence = parse_seq_data(seq_f)
        labeled_sequence = [(f, "") for f in sequence]
        # if len(labeled_sequence) > max_frame:
        #     max_frame = len(labeled_sequence)
        for gesture_start, gesture_end, gesture_label in gesture_infos:
            labeled_sequence = [
                (np.array(f), gesture_label if int(gesture_start) <= idx <= int(gesture_end) and label == "" else label)
                for
                idx, (f, label) in enumerate(labeled_sequence)]

        frames = [f for f, l in labeled_sequence]

        labels_per_frame = [self.classes.index(l) for f, l in labeled_sequence]
        return labeled_sequence, np.array(frames), labels_per_frame


def gendata(
        data_path,
        set_name,
        max_frame
):
    feeder = Feeder_SHREC21(
        data_path=data_path,
        set_name=set_name,
    )
    dataset = feeder.dataset

    fp = np.zeros(
        (len(dataset), max_frame, num_joint, 3), dtype=np.float32
    )
    total_labels = []
    for i, s in enumerate(tqdm(dataset)):
        labeled_seq, data, labels = feeder[i]
        fp[i, :, :, :] = data[:max_frame,:,:]
        total_labels.append(labels[:max_frame])

    # with open(label_out_path, "wb") as f:
    #     pickle.dump((set_name, list(total_labels)), f)

    # np.save(data_out_path, fp)
    return fp, list(total_labels)

class GraphDataset(Dataset):
    def __init__(
        self,
        data_path,
        set_name,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        mmap_mode="r",
    ):
        """Initialise a Graph dataset

        Args:
            data_path ([type]): Path to data
            label_path ([type]): Path to labels
            random_choose (bool, optional): Randomly choose a portion of the input sequence. Defaults to False.
            random_shift (bool, optional): Randomly pad zeros at the begining or end of sequence. Defaults to False.
            random_move (bool, optional): Randomly move joints. Defaults to False.
            window_size (int, optional): The length of the output sequence. Defaults to -1.
            normalization (bool, optional): Normalize input sequence. Defaults to False.
            mmap_mode (str, optional): Use mmap mode to load data, which can save the running memory. Defaults to "r".
        """
        self.data_path = data_path
        self.set_name= set_name
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.mmap_mode = mmap_mode
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Data: N C V T M
        self.data, self.label= gendata(
                self.data_path,
                self.set_name,
                self.window_size
        )        

    def get_mean_map(self):
        data = self.data
        N, C, T, V = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 3, 1))
            .reshape((N * T, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        data_numpy = self.data[index]
        label = self.label[index]

        data_numpy = np.array(data_numpy)
        
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = random_shift(data_numpy)
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)
        
        return data_numpy, label, index



def load_data_sets():
     
    train_ds=GraphDataset("./data/SHREC21","training",window_size=max_frame)
    test_ds=GraphDataset("./data/SHREC21","test",window_size=max_frame)
    INWARD = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (0, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (0, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (0, 16),
        (16, 17),
        (17, 18),
        (18, 19),


    ]

    NUM_NODES = 20

    graph = Graph(inward=INWARD, num_node=NUM_NODES)
    
    return train_ds, test_ds, torch.from_numpy(graph.A)