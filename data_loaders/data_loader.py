import torch
import numpy as np
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
from  .shrec21_loader import gendata
from .graph import Graph

torch.manual_seed(42)
torch.cuda.manual_seed(42)



num_joint = 20
max_frame = 1500



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
        is_segmented=False,
        number_of_samples_per_class=0,
        binary_classes=False
    ):
        """Initialise a Graph dataset
        """
        self.data_path = data_path
        self.set_name = set_name
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
        self.use_aug_by_sw = use_aug_by_sw
        self.number_of_samples_per_class = number_of_samples_per_class
        self.is_segmented = is_segmented
        self.nb_sub_sequences=nb_sub_sequences
        self.sample_classes_=sample_classes
        self.binary_classes=binary_classes
        self.classes = ["NO GESTURE", "GESTURE"] if binary_classes else ["NO GESTURE",
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
        



        





    def load_data(self):
        # Data: N C V T M
        if self.is_segmented :
            self.data, self.ng_sequences_data, self.gesture_sub_sequences_data = gendata(
                self.data_path,
                self.set_name,
                max_frame,
                self.window_size,
                num_joint,
                self.use_aug_by_sw,
                self.is_segmented,
                self.binary_classes
            )
            self.sample_no_gesture_class()
            print("Number of gestures per class in the original "+self.set_name+" set :")
            self.print_classes_information()
            print(self.set_name)
            data = []
            for idx, data_el in enumerate(self.data):
                if np.array(data_el[0]).shape[0] > 0:
                    data.append(data_el)

            self.data = data
            if self.sample_classes_:
                self.sample_classes(self.nb_sub_sequences)
            if self.use_data_aug:
                print("Augmenting data ....")
                augmented_data = []
                for idx, data_el in enumerate(self.data):
                    augmented_skeletons = self.data_aug(self.preprocessSkeleton(
                        torch.from_numpy(np.array(data_el[0])).float()))
                    for s in augmented_skeletons:
                        augmented_data.append((s, data_el[1]))
                self.data = augmented_data
            if self.use_aug_by_sw or self.use_data_aug:
                print("Number of gestures per class in the " +
                    self.set_name+" set after augmentation:")
                self.print_classes_information()
        else :
            self.data = gendata(
                self.data_path,
                self.set_name,
                max_frame,
                self.window_size,
                num_joint,
                self.use_aug_by_sw,
                self.is_segmented,
                self.binary_classes
            )

    def print_classes_information(self):
        data_dict = {i: 0 for i in range(len(self.classes))}
        for seq, label in self.data:
            data_dict[label] += 1
        for class_label in data_dict.keys():
            print("Class", self.classes[class_label],
                  "has", data_dict[class_label], "samples")

    def sample_no_gesture_class(self):
        random.Random(4).shuffle(self.ng_sequences_data)
        print(len(self.ng_sequences_data))
        samples =self.ng_sequences_data[:len(self.data)] if self.binary_classes else self.ng_sequences_data[:self.number_of_samples_per_class*2+(self.nb_sub_sequences if self.use_aug_by_sw else 0)]


        self.data = [*self.data, *samples]

    def sample_classes(self, nb_sub_sequences):
        # Data: N C V T M
        data_dict = {i: [] for i in range(len(self.classes))}
        data = []
        for seq, label in self.data:
            data_dict[label].append((seq, label))

        for k in data_dict.keys():
            samples = data_dict[k][:self.number_of_samples_per_class if k !=
                                   0 else self.number_of_samples_per_class * 3]
            if self.use_aug_by_sw:
                samples = [
                    *samples, *self.gesture_sub_sequences_data[k][:nb_sub_sequences]]
            data = [*data, *samples]

        self.data = data

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

        data_numpy, label = self.data[index]
        # label = self.labels[index]

        skeleton = np.array(data_numpy)
        
        # if self.data_aug :
        #     pass

        data_num = skeleton.shape[0]
        if self.is_segmented==False :
            if data_num < max_frame :
                if self.isPadding:
                    # padding
                    skeleton = self.auto_padding(skeleton, max_frame)
                    # label
                    label=[*label,*[ 0 for _ in range(max_frame-len(label))]]
                else :
                    skeleton = self.upsample(skeleton, max_frame)
            else :
                idx_list = self.sample_frames(data_num, max_frame)
                skeleton = [skeleton[idx] for idx in idx_list]
                skeleton = np.array(skeleton)
                skeleton = torch.from_numpy(skeleton)

            return skeleton, label, index
            
        if data_num >= self.window_size:
            idx_list = self.sample_frames(data_num, self.window_size)
            skeleton = [skeleton[idx] for idx in idx_list]
            skeleton = np.array(skeleton)
            skeleton = torch.from_numpy(skeleton)
        else:
            if self.isPadding:
                # padding
                skeleton = self.auto_padding(skeleton, self.window_size)
            else :
                skeleton = self.upsample(skeleton, self.window_size)

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
            random.Random(4).shuffle(all_joint)
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
            skeletons = [*skeletons, random_sequence_fragments(s)]

        if self.useRandomMoving:
            skeletons.append(random_moving(skeleton))

        if self.useMirroring:
            skeletons = [*skeletons, mirroring(s)]

        return skeletons

    def auto_padding(self, data_numpy, size, random_pad=False):
        T, V, C = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((size, V, C))
            data_numpy_paded[begin:begin + T, :, :] = data_numpy
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

    def sample_frames(self, data_num,sample_size):
        # sample #window_size frames from whole video

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



def load_data_sets(window_size=10, batch_size=32, workers=4, is_segmented=False, binary_classes=False, use_data_aug=False,use_aug_by_sw=False):

    train_ds = GraphDataset("./data/SHREC21", "training", window_size=window_size,
                            use_data_aug=use_data_aug,
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
                            use_aug_by_sw=use_aug_by_sw,
                            sample_classes=False,
                            number_of_samples_per_class=23,
                            is_segmented=is_segmented, binary_classes=binary_classes
                            )
    test_ds = GraphDataset("./data/SHREC21", "test",
                           window_size=window_size,
                           use_data_aug=False,
                           normalize=False,
                           scaleInvariance=False,
                           translationInvariance=False,
                           isPadding=False,
                           number_of_samples_per_class=14,
                           use_aug_by_sw=False,
                           sample_classes=False,
                           is_segmented=is_segmented, binary_classes=binary_classes)
    graph = Graph(layout="SHREC21", strategy="distance")
    # print("train data num: ", len(train_ds))
    print("test data num: ", len(test_ds))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=False)

    return train_loader,val_loader,test_loader, torch.from_numpy(graph.A)
