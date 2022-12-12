from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize
from facexlib.utils.misc import img2tensor, imwrite
import time

class test(FaceRestoreHelper):
    def __init__(self, upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None,
                 model_rootpath=None):
        super(test, self).__init__(upscale_factor,
                 face_size,
                 crop_ratio,
                 det_model,
                 save_ext,
                 template_3points,
                 pad_blur,
                 use_parse,
                 device,
                 model_rootpath)


    def tmp(self, i, basename=0):
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # Add an offset to inverse affine matrix, for more precise back alignment
            if self.upscale_factor > 1:

                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0


            inverse_affine[:, 2] += extra_offset

            if self.use_parse:
                # inference
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)

                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()

                mask = np.zeros((512, 512))
                mask2 = np.zeros((512, 512))

                MASK_COLORMAP2 = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                # MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                # print(out)
                for idx, color in enumerate(MASK_COLORMAP2):
                    mask[out == idx] = color

                for idx, color in enumerate(MASK_COLORMAP2):
                    mask2[out == idx] = color


                mask[:307, :] = 0

                #  blur the mask
                mask = cv2.GaussianBlur(mask, (101, 101), 11)
                mask = cv2.GaussianBlur(mask, (101, 101), 11)


                cv2.imwrite('tor/1_k.jpeg', mask)

                # remove the black borders
                thres = 10
                mask[:thres, :] = 0
                mask[-thres:, :] = 0
                mask[:, :thres] = 0
                mask[:, -thres:] = 0
                mask = mask / 255.

                mask = cv2.resize(mask, restored_face.shape[:2])
                mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)
                inv_soft_mask = mask[:, :, None]

                inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

                pasted_face = inv_restored

            original_unchanged = cv2.imread('a/{}.png'.format(basename), cv2.IMREAD_COLOR)
            upsample_img = cv2.resize(original_unchanged, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)
        return upsample_img