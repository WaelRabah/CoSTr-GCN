import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from random import shuffle
import random
import torch.nn.functional as F


torch.manual_seed(42)
torch.cuda.manual_seed(42)
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


class Graph():

    def __init__(self,
                 layout='DHG14/28',
                 strategy='uniform',
                 max_hop=2,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'DHG14/28':
            self.num_node = 22
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1),
                             (0, 2),
                             (1, 0),
                             (1, 6),
                             (1, 10),
                             (1, 14),
                             (1, 18),
                             (2, 0),
                             (2, 3),
                             (3, 2),
                             (3, 4),
                             (4, 3),
                             (4, 5),
                             (5, 4),
                             (6, 1),
                             (6, 7),
                             (7, 6),
                             (7, 8),
                             (8, 7),
                             (8, 9),
                             (9, 8),
                             (10, 1),
                             (10, 11),
                             (11, 10),
                             (11, 12),
                             (12, 11),
                             (12, 13),
                             (13, 12),
                             (14, 1),
                             (14, 15),
                             (15, 14),
                             (15, 16),
                             (16, 15),
                             (16, 17),
                             (17, 16),
                             (18, 1),
                             (18, 19),
                             (19, 18),
                             (19, 20),
                             (20, 19),
                             (20, 21),
                             (21, 20)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "SHREC21":
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
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
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == "FPHA":
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 0),
                (1, 6),
                (2, 0),
                (2, 7),
                (3, 0),
                (3, 8),
                (4, 0),
                (4, 9),
                (5, 0),
                (5, 10),
                (6, 1),
                (6, 11),
                (7, 2),
                (7, 12),
                (8, 3),
                (8, 13),
                (9, 4),
                (9, 14),
                (10, 5),
                (10, 15),
                (11, 6),
                (11, 16),
                (12, 7),
                (12, 17),
                (13, 8),
                (13, 18),
                (14, 9),
                (14, 19),
                (15, 10),
                (15, 20),
                (16, 11),
                (17, 12),
                (18, 13),
                (19, 14),
                (20, 15)
            ]
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(
            A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

    def normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD


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
            set_name="training",
            window_size=10,
            aug_by_sw=False
    ):
        self.data_path = data_path
        self.set_name = set_name
        self.classes=["",
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
        self.class_to_idx={ class_l:idx for idx, class_l in enumerate(self.classes)}
        self.window_size=window_size
        self.aug_by_sw=aug_by_sw
        self.load_data()


    def load_data(self):
        self.dataset = []
        # load file list
        # classes = set([''])
        # self.classes = []
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
                    # classes.add(gesture_label)
                self.dataset.append((seq_idx, gesture_infos))

        # self.classes = list(classes)
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
        def sample_window(data_num,stride):
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
        # print(len(self.classes))
        labels_per_frame = [self.class_to_idx[l] for f, l in labeled_sequence]
        gestures=[]
        windows_sub_sequences_per_gesture={ i:[] for i in range(len(self.classes))}

        for gesture_start, gesture_end, gesture_label in gesture_infos: 
            gesture_start=int(gesture_start)
            gesture_end=int(gesture_end)
            g_frames=frames[gesture_start:gesture_end]
            g_label=labels_per_frame[gesture_start:gesture_end]
            gestures.append((g_frames,g_label))
            label=self.class_to_idx[gesture_label]
            if self.aug_by_sw :
                num_windows=len(g_frames) // self.window_size
                
                for stride in range(1,self.window_size) :
                    l=len(g_frames)
                    if l // stride >= self.window_size :
                        window_indices=sample_window(l,stride)
                        window=[ g_frames[idx] for idx in window_indices]
                        windows_sub_sequences_per_gesture[label].append((window,label))
                

        ng_sequences=[]
        ng_seq=[]
        l=len(frames)
        indices_ng=[]
        for i in range(len(frames)-1):
            f_curr=frames[i]
            f_next=frames[i+1]
            l_curr=labels_per_frame[i]
            l_next=labels_per_frame[i+1]

            if l_curr==0 and l_next==0 :
                indices_ng.append(i)
                ng_seq.append(f_curr)
                if i==l-2:
                    ng_seq.append(f_next)
                    ng_sequences.append((ng_seq,0))
                    ng_seq=[]
                    continue
            elif l_curr==0 and l_next!=0 :
                indices_ng.append(i)
                ng_seq.append(f_curr)
                ng_sequences.append((ng_seq,0))
                ng_seq=[]
                continue
        
        return  gestures, ng_sequences, windows_sub_sequences_per_gesture

def get_window_label(label,num_classes=18):

    W=len(label)
    sum=torch.zeros((num_classes))
    for t in range(W):
        sum[ label[t]] += 1
    return  sum.argmax(dim=-1).item()

def gendata(
        data_path,
        set_name,
        max_frame,
        window_size=20,
        aug_by_sw=False
):
    feeder = Feeder_SHREC21(
        data_path=data_path,
        set_name=set_name,
        window_size=window_size,
        aug_by_sw=aug_by_sw
    )
    dataset = feeder.dataset

    data = []
    ng_sequences_data=[]
    windows_sub_sequences_data={ i:[] for i in range(len(feeder.classes))}
    for i, s in enumerate(tqdm(dataset)):
        data_el,ng_sequences, windows_sub_sequences_per_gesture = feeder[i]
        ng_sequences_data=[*ng_sequences_data,*ng_sequences]
        l=len(data_el)
        # for w in range(num_windows):
        for idx,gesture in enumerate(data_el) :
            current_skeletons_window = np.array(gesture[0])
            label = gesture[1]
            label = get_window_label(label)
            windows_sub_sequences_data[label]=[*windows_sub_sequences_data[label],*windows_sub_sequences_per_gesture[label]]
            data.append((current_skeletons_window,label)) 
    
    data_classes_count={}
    
    for seq,label in data:
        if label in data_classes_count:
            data_classes_count[label]+=1
        else :
            data_classes_count[label]=1
    data_classes_count[0]=len(ng_sequences_data)

        
    # with open(label_out_path, "wb") as f:
    #     pickle.dump((set_name, list(total_labels)), f)

    # np.save(data_out_path, fp)

    return data, data_classes_count, ng_sequences_data, windows_sub_sequences_data

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
        use_aug_by_sw=False,
        nb_sub_sequences=10,
        sample_classes=False,
        mmap_mode="r",
        number_of_samples_per_class=0
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
        self.use_aug_by_sw=use_aug_by_sw
        self.number_of_samples_per_class=number_of_samples_per_class
        self.classes=["No gesture",
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
        self.load_data()
        self.sample_no_gesture_class()
        print("Number of gestures per class in the original "+self.set_name+" set :")
        self.print_classes_information()
        if sample_classes :
            self.sample_classes(nb_sub_sequences)
            
        
        
        data=[]
        
        for idx,data_el in enumerate(self.data):
            try :
                if np.array(data_el[0]).shape[0]>0 :   
                    data.append(data_el)
            except :
                print(data_el)
            
                
                
                
        self.data=data

        if self.use_data_aug:
            print("Augmenting data ....")
            augmented_data = []
            for idx,data_el in enumerate(self.data):
                augmented_skeletons = self.data_aug(self.preprocessSkeleton(
                    torch.from_numpy(np.array(data_el[0])).float()))
                for s in augmented_skeletons:
                    augmented_data.append((s,data_el[1]))
            self.data = augmented_data
        if self.use_aug_by_sw or self.use_data_aug :
            print("Number of gestures per class in the "+self.set_name+" set after augmentation:")
            self.print_classes_information()
        # if normalization:
        #     self.get_mean_map()

    def load_data(self):
        # Data: N C V T M
        self.data, data_classes_count, self.ng_sequences_data, self.gesture_sub_sequences_data= gendata(
                self.data_path,
                self.set_name,
                max_frame,
                self.window_size,
                self.use_aug_by_sw
        )
        
    def print_classes_information(self):
        data_dict={ i:0 for i in range(len(self.classes))}
        for seq,label in self.data:
            data_dict[label]+=1
        for class_label in data_dict.keys():
            print("Class",self.classes[class_label],"has",data_dict[class_label], "samples")
    def sample_no_gesture_class(self):
        shuffle(self.ng_sequences_data)
        print(len(self.ng_sequences_data))
        samples=self.ng_sequences_data[:self.number_of_samples_per_class ]
        
        self.data=[*self.data,*samples]
        
    def sample_classes(self,nb_sub_sequences):
        # Data: N C V T M
        data_dict={ i:[] for i in range(len(self.classes))}
        data=[]
        for seq,label in self.data:
            data_dict[label].append((seq,label))
        
        
        for k in data_dict.keys():
            samples=data_dict[k][:self.number_of_samples_per_class if k!=0 else self.number_of_samples_per_class * 3]
            if self.use_aug_by_sw :
                samples=[*samples, *self.gesture_sub_sequences_data[k][:nb_sub_sequences]]
            data=[*data,*samples]
        
        self.data=data
        
        
    def __len__(self):
        return len(self.data)
        
        

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
        
        data_numpy,label = self.data[index]
        # label = self.labels[index]

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

        # print(label)
        return skeleton, label, index

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
            n_fragments = 5
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
        
        out = F.interpolate(
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

def load_data_sets(window_size=10, batch_size=32,workers=4):
    
    train_ds=GraphDataset("./data/SHREC21","training",window_size=window_size,
                                use_data_aug=False,
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
                                useTranslationAug=False,
                                  use_aug_by_sw=True,
                                sample_classes=True,
                                number_of_samples_per_class=23
                         )
    test_ds=GraphDataset("./data/SHREC21","test",window_size=window_size, use_data_aug=False,
                                normalize=False, scaleInvariance=False, translationInvariance=False, isPadding=False, number_of_samples_per_class=14,use_aug_by_sw=False,sample_classes=False)
    graph = Graph(layout="SHREC21",strategy="distance")
    print("train data num: ", len(train_ds))
    print("test data num: ", len(test_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False)
    
    return train_loader, val_loader, torch.from_numpy(graph.A)