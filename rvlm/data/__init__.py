import json
import itertools

from .waterbirds import WaterBirds
from .celeba import CelebA
from .imagenet import *
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch
from .transforms import transforms_preprcs
from .templates import imagenet_templates
from .utils import generate_desc


DATASET2CLSNUM = {
    "waterbirds": 2,
    "celeba": 2,
    "imagenet": 1000,
}

def load_dataset(data_dir, dataset_name, split, transform=None, prompt_id=0, cl=False):
    if transform is None:
        try: transform = transforms_preprcs[dataset_name][split]
        except: 
            print(f'There is no pre-defined transform for {dataset_name}.')
            # raise NotImplementedError(f'There is no pre-defined transform for {dataset_name}.')
    
    if dataset_name == "waterbirds":
        project_logits = None
        dataset = WaterBirds(root=f"{data_dir}/{dataset_name}", split=split, transform=transform)    # 4795
        classnames = dataset.classnames
    elif dataset_name == "celeba":
        project_logits = None
        dataset = CelebA(root=f"{data_dir}/{dataset_name}", split=split, transform=transform)
        classnames = dataset.classnames
    elif "imagenet" in dataset_name:
        if dataset_name == "imagenet":
            dataset = ImageNet(transform, data_dir, get_class_text=cl)
        else:
            if split=="train":
                raise NotImplementedError(f'train split for {dataset_name} is not supported.')
            else:
                if dataset_name == "imagenet-r": 
                    dataset = ImageNetR(transform, data_dir)
                elif dataset_name == "imagenet-a":
                    dataset = ImageNetA(transform, data_dir)
                elif dataset_name == "imagenet-v2":
                    dataset = ImageNetV2(transform, data_dir)
                elif dataset_name == "imagenet-sketch":
                    dataset = ImageNetSketch(transform, data_dir)
                else:
                    raise NotImplementedError(f'{dataset_name} is not supported.')
        
        classnames = dataset.classnames
        project_logits = getattr(dataset, 'project_logits', None)
        dataset = dataset.train_dataset if split=='train' else dataset.test_dataset
        dataset.classnames = classnames
        dataset_name = "imagenet"
    else:
        raise NotImplementedError(f'{dataset_name} is not supported.')
    
    dataset.project_logits = project_logits

    os.makedirs(f"{data_dir}/{dataset_name}", exist_ok=True)
    if os.path.exists(f"{data_dir}/{dataset_name}/descs{prompt_id}.json"):
        with open(f"{data_dir}/{dataset_name}/descs{prompt_id}.json", 'r') as f:
            dataset.class_descs = json.load(f)
    else:
        dataset.class_descs = generate_desc(classnames, prompt_id=prompt_id)
        with open(f"{data_dir}/{dataset_name}/descs{prompt_id}.json", 'w') as f:
            json.dump(dataset.class_descs, f, indent='\t')
    
    # all_descs = list(set(itertools.chain(*list(dataset.class_descs.values()))))
    # if dataset_name == "waterbirds":
    #     all_descs = [f"bird with {desc.lower()}" for desc in all_descs]
    # elif dataset_name == "celeba":
    #     all_descs = [f"human face with {desc.lower()}" for desc in all_descs]
    # elif dataset_name == "imagenet":
    #     all_descs = [f"{desc.lower()}" for desc in all_descs]

    
    return dataset