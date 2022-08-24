import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import numpy as np
from random import shuffle
import random
import torch.nn.functional as F

from typing import List, Tuple



import random




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
        window_size,
        use_data_aug=False,
        normalize=True,
        scaleInvariance=False,
        translationInvariance=False,
        isPadding=False,
        useSequenceFragments=False,
        useRandomMoving=False,
        useMirroring=False,
        useTimeInterpolation=False,
        useNoise=False,
        useScaleAug=False,
        useTranslationAug=False,
        mmap_mode="r",
    ):
        """Initialise a Graph dataset
        """
        self.data_path = data_path
        self.set_name= set_name
        self.use_data_aug = use_data_aug
        self.window_size = window_size
        self.compoent_num = 20
        self.normalize = normalize
        self.scaleInvariance = scaleInvariance
        self.translationInvariance = translationInvariance
        # self.transform = transform
        self.isPadding = isPadding
        self.useSequenceFragments = useSequenceFragments
        self.useRandomMoving = useRandomMoving
        self.useMirroring = useMirroring
        self.useTimeInterpolation = useTimeInterpolation
        self.useNoise = useNoise
        self.useScaleAug = useScaleAug
        self.useTranslationAug = useTranslationAug
        self.mmap_mode = mmap_mode
        self.load_data()
        if self.use_data_aug:
            print("Augmenting data ....")
            augmented_data = []
            aug_labels=[]
            for idx,data_el in enumerate(self.data):
                augmented_skeletons = self.data_aug(self.preprocessSkeleton(
                    torch.from_numpy(np.array(data_el)).float()))
                for s in augmented_skeletons:
                    augmented_data.append(s)
                    aug_labels.append(self.label[idx])
            self.data = augmented_data
            self.label=aug_labels
        # if normalization:
        #     self.get_mean_map()

    def load_data(self):
        # Data: N C V T M
        self.data, self.label= gendata(
                self.data_path,
                self.set_name,
                self.window_size
        )        
    def data_aug(self, skeleton):

        def scale(skeleton):
            ratio = 0.2
            low = 1 - ratio
            high = 1 + ratio
            factor = np.random.uniform(low, high)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] *= factor
            skeleton = np.array(skeleton)
            return skeleton

        def shift(skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    skeleton[t][j_id] += offset
            skeleton = np.array(skeleton)
            return skeleton

        def noise(skeleton):
            low = -0.1
            high = -low
            # select 4 joints
            all_joint = list(range(self.compoent_num))
            shuffle(all_joint)
            selected_joint = all_joint[0:4]
            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(skeleton.shape[0]):
                    skeleton[t][j_id] += noise_offset

            skeleton = np.array(skeleton)
            return skeleton

        def time_interpolate(skeleton):
            skeleton = np.array(skeleton)
            video_len = skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []

            for i in range(1, video_len):
                displace = skeleton[i] - skeleton[i - 1]  # d_t = s_t+1 - s_t
                displace *= r
                result.append(skeleton[i - 1] + displace)  # r*disp

            while len(result) < self.window_size:
                result.append(result[-1])  # padding
            result = np.array(result)
            return result

        def random_sequence_fragments(sample):
            samples = [sample]
            sample = torch.from_numpy(sample)
            n_fragments = 2
            T, V, C = sample.shape
            if T <= self.window_size:
                return samples
            for _ in range(n_fragments):

                # fragment_len=int(T*fragment_len)
                fragment_len = self.window_size
                max_start_frame = T-fragment_len

                random_start_frame = random.randint(0, max_start_frame)
                new_sample = sample[random_start_frame:random_start_frame+fragment_len]
                samples.append(new_sample.numpy())

            return samples

        def mirroring(data_numpy):
            T, V, C = data_numpy.shape
            data_numpy[:, :, 0] = np.max(
                data_numpy[:, :, 0]) + np.min(data_numpy[:, :, 0]) - data_numpy[:, :, 0]
            return data_numpy

        def random_moving(data_numpy,
                          angle_candidate=[-10., -5., 0., 5., 10.],
                          scale_candidate=[0.9, 1.0, 1.1],
                          transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                          move_time_candidate=[1]):
            # input: T,V,C
            data_numpy = np.transpose(data_numpy, (2, 0, 1))
            new_data_numpy = np.zeros(data_numpy.shape)
            C, T, V = data_numpy.shape
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
                a[node[i]:node[i + 1]] = np.linspace(
                    A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
                s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                     node[i + 1] - node[i])
                t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                       node[i + 1] - node[i])
                t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                       node[i + 1] - node[i])

            theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                              [np.sin(a) * s, np.cos(a) * s]])

            # perform transformation
            for i_frame in range(T):
                xy = data_numpy[0:2, i_frame, :]
                new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))

                new_xy[0] += t_x[i_frame]
                new_xy[1] += t_y[i_frame]

                new_data_numpy[0:2, i_frame, :] = new_xy.reshape(2, V)

            new_data_numpy[2, :, :] = data_numpy[2, :, :]

            return np.transpose(new_data_numpy, (1, 2, 0))

        skeleton = np.array(skeleton)
        skeletons = [skeleton]
        if self.useTimeInterpolation:
            skeletons.append(time_interpolate(skeleton))

        if self.useNoise:
            skeletons.append(noise(skeleton))

        if self.useScaleAug:
            skeletons.append(scale(skeleton))

        if self.useTranslationAug:
            skeletons.append(shift(skeleton))

        if self.useSequenceFragments:
            n_skeletons = []
            for s in skeletons:
                n_skeletons = [*n_skeletons, *random_sequence_fragments(s)]
            skeletons = [*skeletons, *n_skeletons]

        if self.useRandomMoving:
            # aug_skeletons = []
            # for s in skeletons:
            #     aug_skeletons.append(random_moving(s))
            skeletons.append(random_moving(skeleton))

        if self.useMirroring:
            aug_skeletons = []
            for s in skeletons:
                aug_skeletons.append(mirroring(s))
            skeletons = [*skeletons, *aug_skeletons]

        return skeletons

    def auto_padding(self, data_numpy, size, random_pad=False):
        C, T, V = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((C, size, V))
            data_numpy_paded[:, begin:begin + T, :] = data_numpy
            return data_numpy_paded
        else:
            return data_numpy

    def upsample(self, skeleton, max_frames):
        tensor = torch.unsqueeze(torch.unsqueeze(
            torch.from_numpy(skeleton), dim=0), dim=0)

        out = nn.functional.interpolate(
            tensor, size=[max_frames, tensor.shape[-2], tensor.shape[-1]], mode='trilinear')
        tensor = torch.squeeze(torch.squeeze(out, dim=0), dim=0)

        return tensor

    def sample_frames(self, data_num):
        # sample #window_size frames from whole video

        sample_size = self.window_size
        each_num = (data_num - 1) / (sample_size - 1)
        idx_list = [0, data_num - 1]
        for i in range(sample_size):
            index = round(each_num * i)
            if index not in idx_list and index < data_num:
                idx_list.append(index)
        idx_list.sort()

        while len(idx_list) < sample_size:
            idx = random.randint(0, data_num - 1)
            if idx not in idx_list:
                idx_list.append(idx)
        idx_list.sort()
        return idx_list
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
    def preprocessSkeleton(self, skeleton):
        def translationInvariance(skeleton):
            # normalize by palm center value at frame=1
            skeleton -= torch.clone(skeleton[0][1])
            skeleton = skeleton.float()
            return skeleton

        def scaleInvariance(skeleton):

            x_c = torch.clone(skeleton)

            distance = torch.sqrt(torch.sum((x_c[0, 1]-x_c[0, 0])**2, dim=-1))

            factor = 1/distance

            x_c *= factor

            return x_c

        def normalize(skeleton):

            # if self.transform:
            #     skeleton = self.transform(skeleton.numpy())
            skeleton = F.normalize(skeleton)

            return skeleton
        if self.normalize:
            skeleton = normalize(skeleton)
        if self.scaleInvariance:
            skeleton = scaleInvariance(skeleton)
        if self.translationInvariance:
            skeleton = translationInvariance(skeleton)

        return skeleton
    def __getitem__(self, index):
        
        data_numpy = self.data[index]
        label = self.label[index]

        skeleton = np.array(data_numpy)
        
        # if self.data_aug :
        #     pass

        data_num = skeleton.shape[0]
        if self.isPadding:
            # padding
            skeleton = self.auto_padding(skeleton, self.max_seq_size)
            # label


            return skeleton, label, index

        if data_num >= self.window_size:
            idx_list = self.sample_frames(data_num)
            skeleton = [skeleton[idx] for idx in idx_list]
            skeleton = np.array(skeleton)
            skeleton = torch.from_numpy(skeleton)
        else:
            skeleton = self.upsample(skeleton, self.window_size)


        return skeleton, label, index



def load_data_sets():

    train_ds=GraphDataset("./data/SHREC21","training",window_size=max_frame,
                                use_data_aug=True,
                                normalize=False, 
                                scaleInvariance=False,
                                translationInvariance=False, 
                                useRandomMoving=True, 
                                isPadding=False, 
                                useSequenceFragments=False, 
                                useMirroring=False,
                                useTimeInterpolation=False,
                                useNoise=True,
                                useScaleAug=False,
                                useTranslationAug=False)
    test_ds=GraphDataset("./data/SHREC21","test",window_size=max_frame, use_data_aug=False,
                                normalize=False, scaleInvariance=False, translationInvariance=False, isPadding=False)
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