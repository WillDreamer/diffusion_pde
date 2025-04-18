import os
import re
import math as mt
import numpy as np
import torch
import h5py
from . import video_transforms

def extract_parameters(filename):
    # 定义正则表达式模式来匹配 M, Eta 和 Zeta
    pattern = r'M([\d.]+)_Eta([\d.eE-]+)_Zeta([\d.eE-]+)'

    # 使用正则表达式搜索匹配的内容
    match = re.search(pattern, filename)
    
    if match:
        # 提取匹配的组
        M = float(match.group(1))
        Eta = float(match.group(2))
        Zeta = float(match.group(3))
        return M, Eta, Zeta
    else:
        raise ValueError("Filename does not contain the expected pattern.")

class PDEBench_npy(torch.utils.data.Dataset):
    """load pde according to the npy file.

    """

    def __init__(
        self,
        args=None, 
        data_path='/local2/shared_data/DBench_data/PDEBench/',
        filename=['2D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Aug4000.npy'],
        num_frames=16, # number of frames to be generated + conditioned frames (set in train.py)
        image_size=32,
        use_spatial_sample=False,
        num_channels=4,
        normalize=False,
        use_coordinates=True,
        frame_interval=1,
        is_train=True,
        train_ratio=0.9
    ):
        self.data_path = data_path
        self.filename = filename
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_channels = num_channels
        self.normalize = normalize
        self.use_spatial_sample = use_spatial_sample
        self.use_coordinates = use_coordinates
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.reduced_resolution = 512 // image_size
        if self.is_train:
            self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
            if self.use_spatial_sample:
                self.spatial_sample = video_transforms.SpatialRandomCrop(16)
        # 重新使用内存映射读取数据
        print("Reading data from memory-mapped file...")
        memmapped_array = np.memmap(os.path.join(data_path, filename[0]), dtype='float32', mode='r', shape=(1000, 21, 512, 512, 4))
        if self.use_coordinates:
            x = np.load(os.path.join(data_path, 'x_coordinate.npy'))
            y = np.load(os.path.join(data_path, 'x_coordinate.npy'))
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x, y)
            self.grid = torch.stack((X, Y), dim=-1)[::self.reduced_resolution, ::self.reduced_resolution]

        if self.normalize:
            batch_stat = 1000 # 计算统计量时样本的下采样率
            reduced_resolution_stat = 1 # 计算统计量时空间的下采样率
            stat_data = memmapped_array[:batch_stat, ::reduced_resolution_stat, ::reduced_resolution_stat]
            stat_dim = tuple(range(4)) # 除了最后的channel维，其他维度都是统计维度
            self.means = torch.from_numpy(stat_data.mean(stat_dim))
            self.stds = torch.from_numpy(stat_data.std(stat_dim))
            # self.means = torch.from_numpy(np.load('dataset_means.npy')).float()[0] # 存下来时多了一维
            # self.stds = torch.from_numpy(np.load('dataset_stds.npy')).float()[0]
            print(f"Mean: {self.means}, Std: {self.stds}")
        
        self.data = memmapped_array

    def __getitem__(self, index):
        # THWC -> HWTC
        sample = torch.from_numpy(np.copy(self.data[index])).permute(1, 2, 0, 3)
        sample = sample.sub(self.means).div(self.stds)
        total_frames = sample.shape[2]
        # Sampling video frames
        # temporal_sample returns the start and end frame indices, if num_frames > total_frames, then start_frame_ind = 0, end_frame_ind = total_frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames) if self.is_train else (0, self.num_frames)
        start_pixel_xind, start_pixel_yind = self.spatial_sample() if self.is_train and self.use_spatial_sample else (0, 0)
        frame_indice = np.arange(start_frame_ind, end_frame_ind, dtype=int)

        video = sample[start_pixel_xind::self.reduced_resolution, start_pixel_yind::self.reduced_resolution, frame_indice, :self.num_channels]
        grid = self.grid if self.use_coordinates else None

        # HWTC -> TCHW
        video = video.permute(2, 3, 0, 1)
        grid = grid.permute(2, 0, 1) if grid is not None else None
        return {'video': video, 'equation_name': 0}
    
    def __len__(self):
        return self.data.shape[0]
    
class PDEBench(torch.utils.data.Dataset):
    """load pde according to the npy file.

    """

    def __init__(
        self,
        args=None, 
        pde_names=["2DCFD"],
        data_path='/local2/shared_data/DBench_data/PDEBench/',
        filename=['2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'],
        num_frames=16, # number of frames to be generated + conditioned frames (set in train.py)
        image_size=32,
        use_spatial_sample=False,
        num_channels=4,
        normalize=False,
        use_coordinates=True,
        frame_interval=1,
        is_train=True,
        train_ratio=0.9
    ):
        self.pde_names = pde_names
        self.data_path = data_path
        self.filename = filename
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_channels = num_channels
        self.normalize = normalize
        self.use_spatial_sample = use_spatial_sample
        self.use_coordinates = use_coordinates
        self.is_train = is_train
        self.train_ratio = train_ratio
        if self.is_train:
            self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
            if self.use_spatial_sample:
                self.spatial_sample = video_transforms.SpatialRandomCrop(16)
        for k, dataset in enumerate(self.pde_names):
            if dataset == 'Burgers':
                '''
                1D Burger, 12 files, Nu 0.001-4.0, 
                ['t-coordinate', 'tensor', 'x-coordinate']
                (202,1): 0->2
                (10000, 201, 1024)
                (1024,1): 0->1
                '''
                filename = '1D_Burgers_Sols_Nu0.001.hdf5'
                reduced_resolution = 8
                reduced_resolution_t = 5
            elif dataset == 'NS_incom':
                assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
                
                sub_root = '2D/NS_incom/'
                filename = 'ns_incom_inhom_2d_512-0.h5'
                root_path = os.path.abspath(self.root + sub_root + filename)
            
            elif dataset == '2DCFD':
                
                # sub_root = ''
                data_all = []
                data_size_all = []
                text_all = []
                equation_name_all = []
                if self.use_coordinates:
                    grid_all = []
                for file_idx, filename in enumerate(self.filename):
                    M, Eta, Zeta = extract_parameters(filename)
                    text_all.append(f"2D Navier-Stokes, M={M}, Eta={Eta}, Zeta={Zeta}, Periodic, grid size is {image_size}x{image_size}.")
                    root_path = os.path.join(self.data_path, filename)
                    reduced_batch = 1
                    reduced_resolution_t = 1
                    with h5py.File(root_path, 'r') as f:
                        keys = list(f.keys())
                        keys.sort()
                        _data = np.array(f['density'], dtype=np.float32) # 为了能够计算出一致的统计量来归一化，必须读取全部数据
                        idx_cfd = _data.shape # 单个物理量的形状(N, T, H, W)
                        assert idx_cfd[-1] % image_size == 0, f"image_size {image_size} should be a divisor of {idx_cfd[-1]}"
                        self.reduced_resolution = idx_cfd[-1] // image_size # 本来的代码里需要指定reduced_resolution，这里换成根据image_size来计算
                        
                        data = np.zeros([idx_cfd[0],
                                        idx_cfd[2],
                                        idx_cfd[3],
                                        mt.ceil(idx_cfd[1]/reduced_resolution_t),
                                        4],
                                        dtype=np.float32) # 最后物理量的形状(N, H, W, T, C)
                        
                        # 循环读取数据
                        field_names = ['Vx', 'Vy', 'density', 'pressure']
                        for i, field in enumerate(field_names):
                            _data = np.array(f[field], dtype=np.float32)
                            _data = _data[:,::reduced_resolution_t,::1,::1] # 我不打算下采样时间，所以没有对原代码进行改动
                            # from (N, T, H, W) to (N, H, W, T)
                            _data = np.transpose(_data, (0, 2, 3, 1))
                            data[...,i] = _data
                        
                        if use_coordinates:
                            x = np.array(f["x-coordinate"], dtype=np.float32)
                            y = np.array(f["y-coordinate"], dtype=np.float32)
                            x = torch.tensor(x, dtype=torch.float)
                            y = torch.tensor(y, dtype=torch.float)
                            X, Y = torch.meshgrid(x, y) # (H, W)
                            self.grid = torch.stack((X, Y), dim=-1)[::reduced_resolution, ::reduced_resolution] # (H, W, 2)
                        
                        if normalize:
                            reduced_batch_stat = 1 # 计算统计量时样本的下采样率
                            reduced_resolution_stat = 8 # 计算统计量时空间的下采样率
                            stat_data = data[::reduced_batch_stat, ::reduced_resolution_stat, ::reduced_resolution_stat]
                            stat_dim = tuple(range(len(idx_cfd)-1)) # 除了最后的channel维，其他维度都是统计维度
                            means = stat_data.mean(stat_dim, keepdims=True)
                            stds = stat_data.std(stat_dim, keepdims=True)
                            # means = np.array([-2.3081107e-03, -2.6687531e-04, 4.5385528e+00, 2.4850531e+01], dtype=np.float32)
                            # stds = np.array([1.3211658, 1.3148949, 2.869468, 23.5342], dtype=np.float32)
                            data = (data - means) / (stds) # 为了让数据归一化到[-1, 1]之间，除以1倍的标准差
                        data_all.append(data)
                        data_size_all.append(data.shape[0])
                        equation_name_all.append(file_idx)
            
            val_start_all = [int(self.train_ratio * len(data)) for data in data_all]
            if self.is_train:
                self.data = [data[:val_start] for data, val_start in zip(data_all, val_start_all)]
            else:
                self.data = [data[val_start:] for data, val_start in zip(data_all, val_start_all)]

            self.lengths = [len(data) for data in self.data]
            self.ends_data = np.cumsum(self.lengths)
            self.data = np.concatenate(self.data, axis=0) 
            self.text = text_all
            self.equation_name = equation_name_all
            self.means = means
            self.stds = stds

    def __getitem__(self, index):
        sample = self.data[index]
        data_idx = np.searchsorted(self.ends_data, index+1)
        text = self.text[data_idx]
        equation_name = self.equation_name[data_idx]
        
        total_frames = self.data.shape[3]
        # Sampling video frames
        # temporal_sample returns the start and end frame indices, if num_frames > total_frames, then start_frame_ind = 0, end_frame_ind = total_frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames) if self.is_train else (0, self.num_frames)
        start_pixel_xind, start_pixel_yind = self.spatial_sample() if self.is_train and self.use_spatial_sample else (0, 0)
        frame_indice = np.arange(start_frame_ind, end_frame_ind, dtype=int)

        video = torch.from_numpy(sample[start_pixel_xind::self.reduced_resolution, start_pixel_yind::self.reduced_resolution, frame_indice, :self.num_channels])
        grid = self.grid if self.use_coordinates else None
        # 与grid拼接
        # if self.use_coordinates:
        #     video = torch.cat((video, self.grid.unsqueeze(2).repeat(1, 1, self.num_frames, 1)), dim=-1)

        # HWTC -> TCHW
        video = video.permute(2, 3, 0, 1)
        grid = grid.permute(2, 0, 1) if grid is not None else None
        
        if not self.use_coordinates:
            return {'video': video, 'equation_name': equation_name, 'text': text}
        return {'video': video, 'equation_name': equation_name, 'text': text, 'grid': grid}

    def __len__(self):
        return self.data.shape[0]