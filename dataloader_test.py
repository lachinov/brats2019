import dataloader




if __name__ == '__main__':
    path = '../data/train_resampled'
    reader = dataloader.SimpleReader(path=path, patch_size=(128,128,128))

    data = reader.__getitem__(0)

    print(data.shape)