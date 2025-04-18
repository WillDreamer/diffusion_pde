from .pdebench_datasets import PDEBench, PDEBench_npy
from torchvision import transforms
from datasets import video_transforms
from .CNS_data_utils import DatasetSingle


def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'pdebench':
        return PDEBench(args)

    else:
        raise NotImplementedError(args.dataset)