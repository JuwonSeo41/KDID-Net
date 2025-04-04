import numpy as np
import torch
import torch.nn as nn
# from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()

        class_name = data['class_name']
        class_to_index = {
            "AP_C": 0, "AP_H": 1, "AP_R": 2, "AP_S": 3, "BB_H": 4, "CH_H": 5, "CH_P": 6, "CO_B": 7, "CO_C": 8, "CO_H": 9,
            "CO_R": 10, "GR_B": 11, "GR_E": 12, "GR_H": 13, "GR_R": 14, "OR_H": 15, "PB_B": 16, "PB_H": 17, "PH_B": 18,
            "PH_H": 19, "PO_EB": 20, "PO_H": 21, "PO_LB": 22, "RE_H": 23, "SO_H": 24, "SQ_H": 25, "ST_H": 26,
            "ST_S": 27, "TO_BS": 28, "TO_EB": 29, "TO_H": 30, "TO_LB": 31, "TO_M": 32, "TO_SL": 33, "TO_SS": 34,
            "TO_TS": 35, "TO_V": 36, "TO_Y": 37
        }

        # class_to_index = {
        #     "Cassava_Bacterial_Blight": 0, "Cassava_Brown_Leaf_Spot": 1, "Cassava_Healthy": 2, "Cassava_Mosaic": 3,
        #     "Cassava_Root_Rot": 4, "Corn_Blight": 5, "Corn_Brown_Spots": 6, "Corn_Cercosporiose": 7, "Corn_Charcoal": 8,
        #     "Corn_Chlorotic_Leaf_Spot": 9, "Corn_Healthy": 10, "Corn_Insect_Damages": 11, "Corn_Mildiou": 12,
        #     "Corn_Purple_Discoloration": 13, "Corn_Rust": 14, "Corn_Smut": 15, "Corn_Streak": 16, "Corn_Stripe": 17,
        #     "Corn_Violet_Decoloration": 18, "Corn_Yellow_Spots": 19, "Corn_Yellowing": 20, "Tomato_Brown_Spots": 21,
        #     "Tomato_Healthy": 22, "Tomato_leaf_Curling": 23, "Tomato_Mildiou": 24, "Tomato_Mosaic": 25
        # }
        target_class = [class_to_index.get(name) for name in class_name]
        target_class = torch.tensor(target_class).cuda()

        return inputs, targets, target_class

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        # ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        # return psnr, ssim, vis_img
        return psnr, vis_img


def get_model(model_config):
    return DeblurModel()
