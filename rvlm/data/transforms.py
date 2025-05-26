import torchvision.transforms as T



# transforms_preprcs = {
#     "waterbirds": {
#         "train": T.Compose(
#             [
#                 T.RandomResizedCrop(
#                     (224,224),
#                     scale=(0.7, 1.0),
#                     ratio=(0.75, 1.3333333333333333),
#                     interpolation=2),
#                 T.RandomHorizontalFlip(),
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ]    
#         ),
#         "valid": T.Compose(
#             [
#                 T.Resize((256, 256)),
#                 T.CenterCrop((224,224)),
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ]
#         ),
#         "test": T.Compose(
#             [
#                 T.Resize((256, 256)),
#                 T.CenterCrop((224,224)),
#                 T.ToTensor(),
#                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ]
#         ),
#     },
# }

transforms_preprcs = {
    "waterbirds": {
        "train": T.Compose(
            [
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                # T.RandomResizedCrop(
                #     (224,224),
                #     scale=(0.7, 1.0),
                #     ratio=(0.75, 1.3333333333333333),
                #     interpolation=2),
                # T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]    
        ),
        "valid": T.Compose(
            [
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "test": T.Compose(
            [
                T.Resize((256, 256)),
                T.CenterCrop((224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    },
}