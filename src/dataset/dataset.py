from mindspore import dtype as mstype
import mindspore.dataset as ds
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2


def get_wukong_dataset(dataset_path, columns_list, num_parallel_workers, shuffle, num_shards, shard_id, batch_size):
    wukong_dataset = ds.MindDataset(dataset_path,
                                    columns_list=columns_list,
                                    num_parallel_workers=num_parallel_workers,
                                    shuffle=shuffle,
                                    num_shards=num_shards,
                                    shard_id=shard_id)
    wukong_dataset = wukong_dataset.batch(batch_size)
    return wukong_dataset


def get_dataset(dataset_path, batch_size):
    norm_mean = (0.48145466, 0.4578275, 0.40821073)
    norm_std = (0.26862954, 0.26130258, 0.27577711)
    norm_mean_2 = tuple(map(lambda x: x * 255, norm_mean))
    norm_std_2 = tuple(map(lambda x: x * 255, norm_std))
    val_dataset = ds.ImageFolderDataset(dataset_path, num_parallel_workers=4)
    val_dataset = val_dataset.map(
        [C.Decode(),
         C.Normalize(mean=norm_mean_2, std=norm_std_2),
         C.Resize(224, Inter.BICUBIC),
         C.CenterCrop(224),
         C.HWC2CHW(),
         C2.TypeCast(mstype.float32)],
        input_columns=["image"])
    val_dataset = val_dataset.batch(batch_size)
    return val_dataset
