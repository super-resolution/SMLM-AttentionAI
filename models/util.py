#Utility functions not in use right now

def gaussian_initializer(shape=(10,10,40,40)):
    im = torch.zeros((100,40,40))
    for i in range(10):
        for j in range(10):
            im[i*10+j,i*39//9,j*39//9] = 1
    gaussian = lambda x: (torch.exp(-(x) ** 2 / (2 * (2) ** 2)))
    w = gaussian(torch.arange(-10,11,1.0))
    im = F.pad(im, (10,10,10,10), "reflect")
    res = F.conv2d(im[:,None], w[None,None,None,:], padding="valid")
    res = F.conv2d(res, w[None,None,:,None], padding="valid")
    res =  res.flatten(start_dim=1)

    return res
def patchify(images, patch_size):
    # todo: patchify and unpatchify?
    patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = torch.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2], -1))
    return patches

def unpatchify(patches, patch_size, n_channels=8):
    h = int(math.sqrt(patches.shape[1]))#only takes quadratic patches
    patches = patches[:, :, :].unfold(2, patch_size ** 2, patch_size ** 2)
    patches = torch.permute(patches, dims=(0, 2, 1, 3))
    patches = patches.unfold(2, h, h).unfold(3, patch_size, patch_size)  # n patches, patch size

    # 8 feauture maps
    images = patches.reshape((patches.shape[0], n_channels, h*patch_size, h*patch_size))
    return images

def decoder_patchify(images,n_patches):
    # in shape b,f,10,10
    #out shape b, n_p², -1
    n,c,h,w = images.shape
    #patches = images.unfold(1, n_patches, n_patches)
    #patches = torch.permute(patches,(0,1,3,4,2))
    return images.reshape((n,n_patches,-1))

def decoder_unpatch(patches,patch_size):
    # in shape b, n_p², f
    # out shape b, n_p²*<-n_f,h,w can be done by view?
    #patches = patches[:, :, :].unfold(2, patch_size ** 2, patch_size ** 2)
    #patches = torch.permute(patches, dims=(0, 2, 1, 3))
    #patches = patches.unfold(3, patch_size, patch_size)
    c1,c2 = 36,8
    return patches.reshape((patches.shape[0],c1*c2,10,10))