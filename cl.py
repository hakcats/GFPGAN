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


    # def tmp(self, mask, upsample_img):
    def tmp(self, i, basename=0):
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        # if upsample_img is None:
        #     # simply resize the background
        #     upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        # else:
        #     upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        #     # upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        # print(i)
        upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            # Add an offset to inverse affine matrix, for more precise back alignment
            if self.upscale_factor > 1:

                extra_offset = 0.5 * self.upscale_factor
            else:
                extra_offset = 0


            inverse_affine[:, 2] += extra_offset


            # cv2.imwrite('tor/1_a.jpeg', inv_restored)
            if self.use_parse:
                # inference
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)

                # f = 'tor/{}.pth'.format(1)
                # out = torch.load(f)

                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()
                # print(out.shape)

                # torch.save(out, 'data/gwen/{}.pth'.format(i))

                mask = np.zeros((512, 512))
                mask2 = np.zeros((512, 512))
                # MASK_COLORMAP2 = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 255, 0]
                MASK_COLORMAP2 = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]
                # MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                # print(out)
                for idx, color in enumerate(MASK_COLORMAP2):
                    mask[out == idx] = color

                for idx, color in enumerate(MASK_COLORMAP2):
                    mask2[out == idx] = color

                # nmask = np.zeros((512, 512))
                # MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                # for idx, color in enumerate(MASK_COLORMAP):
                #     nmask[out == idx] = color
                #
                # cv2.imwrite('tor/mask.jpeg', mask)
                # cv2.imwrite('tor/nmask.jpeg', nmask)


                mask_data = mask2.copy()
                # mask[307:460, 120:400] = 255
                mask[:307, :] = 0

                # mask2 = mask[255:, 63:448]
                # cv2.imwrite('tor/1_c.jpeg', out)
                # cv2.imwrite('tor/1_b.jpeg', mask)
                # cv2.imwrite('tor/1_d.jpeg', mask2)
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

                # thres = 1
                # mask_data[:thres, :] = 0
                # mask_data[-thres:, :] = 0
                # mask_data[:, :thres] = 0
                # mask_data[:, -thres:] = 0

                cv2.imwrite('tor/1_km.jpeg', mask_data)
                # torch.save(mask, 'data/gwen/{}.pth'.format(i))

                mask = cv2.resize(mask, restored_face.shape[:2])
                mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)
                inv_soft_mask = mask[:, :, None]


                # print(mask_left.shape)
                # exit(0)
                # orig_image = cv2.imread('results/gwen/restored_faces/{}_00.png'.format(basename))
                # orig_image_left = orig_image[358:, 80:192]
                # orig_image_right = orig_image[358:, 320:432]

                # new_f = restored_face[358:, 80:192]

                # print(mask_data.shape)
                mask_left = mask_data[307:460, 153:307]
                mask_left = cv2.GaussianBlur(mask_left, (15, 15), 1)

                cv2.imwrite('tor/1_l.jpeg', mask_left)
                mask_right = mask_data[368:470, 320:402]
                mask_right = cv2.GaussianBlur(mask_right, (15, 15), 1)

                # u_green = np.array([104, 153, 70])
                # l_green = np.array([30, 30, 0])
                #
                # masking = cv2.inRange(restored_face, l_green, u_green)
                # image = np.zeros((614, 614, 3))
                # image[:] = [0, 255, 0]
                #
                # restored_face = cv2.bitwise_and(restored_face, image, mask=masking)

                # for h in range(153):
                #     for k in range(154):
                # #
                # #         diff = np.sqrt(((restored_face[377 + h][130 + k] - restored_face[377 + h][130 + k + 1])**2).mean()/255.)
                # #         # print(diff)
                #         if int(mask_left[h][k]) == 0:
                #             restored_face[306 + h][319 + k] = restored_face[306 + h][319 + k - 1]
                #         else:
                #             break
                        # if int(mask_right[h][k]) == 0:
                        #     restored_face[367 + h][310 + k] = restored_face[367 + h][310 + k + 1]

                # blured = cv2.GaussianBlur(restored_face[358:468, 110:192], (21, 21), 1)
                # mask = np.zeros(restored_face[358:468, 110:192].shape, np.uint8)

                # grey = cv2.cvtColor(restored_face[358:468, 110:192], cv2.COLOR_BGR2GREY)
                # tresh = cv2.threshold(grey, 60, 255, cv2.TRESH_binary)[2]
                # restored_face[358:468, 110:192] = cv2.GaussianBlur(restored_face[358:468, 110:192], (51, 41), 3)

                # exit(0)
                # cv2.imwrite('tor/1_f.jpeg', new_f)

                # new_f2 = cv2.blur(new_f, (3, 3))
                #
                # cv2.imwrite('tor/1_g.jpeg', new_f2)
                inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))
                cv2.imwrite('tor/1_a.jpeg', inv_restored)

                pasted_face = inv_restored
                # print(pasted_face.shape, restored_face.shape)
                # masking change

            # mask = cv2.resize(mask, restored_face.shape[:2])
            # mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up), flags=3)

            # inv_soft_mask = mask[:, :, None]

            # pasted_face = inv_restored
            original_unchanged = cv2.imread('a/{}.png'.format(basename), cv2.IMREAD_COLOR)
            upsample_img = cv2.resize(original_unchanged, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite('tor/1_u.jpeg', upsample_img)
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