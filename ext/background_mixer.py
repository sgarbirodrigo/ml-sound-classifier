import numpy as np

def get_background_mixer(backgrounds, random_ratio=0.5):
    def bg_mixer(img):
        ratio = np.random.rand() * random_ratio
        bgidx = np.random.randint(0, len(backgrounds))
        bg    = backgrounds[bgidx]
        img_h, img_w = img.shape[:2]
        bg_h, bg_w   = bg.shape[:2]
        # crop from bg to the size of img
        top    = np.random.randint(0, np.max([1, bg_h - img_h]))
        height = np.min([img_h, bg_h - top])
        left   = np.random.randint(0, np.max([1, bg_w - img_w]))
        width  = np.min([img_w, bg_w - left])
        c_top  = (img_h - height) // 2 # centering cropped bg
        c_left = (img_w - width)  // 2
        cropped= np.zeros((img_h, img_w, 1))
        cropped[c_top:c_top+height, c_left:c_left+width, :] = bg[top:top+height, left:left+width, :]
        # mix up
        return (img + cropped*ratio) / (1 + ratio)

    return bg_mixer

