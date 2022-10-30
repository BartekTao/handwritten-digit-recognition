import os
import cv2
import numpy as np

origin_path = './data/train'
resized_path = './resized/train'
ext = '.png'

print('start converting...')
for i in range(0,10):
    from_dir_path = os.path.join(origin_path, str(i))
    to_dir_path = os.path.join(resized_path, str(i))
    os.makedirs(to_dir_path, exist_ok=True)

    pics = sorted(os.listdir(from_dir_path))
    print(f'number[{i}] pictures = {len(pics)}')

    for i, pic in enumerate(pics):
        from_path = os.path.join(from_dir_path, pic)
        to_path = os.path.join(to_dir_path, f'{i}{ext}')

        image = cv2.imread(from_path)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 腐蚀范围2x2
        kernel = np.ones((5,5),np.uint8)
        # 迭代次数 iterations=1
        erosion = cv2.erode(image,kernel,iterations = 3)
        # cv2.imshow('erosion',erosion)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image_resize', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        # cv2.imshow('erosion_resize', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(to_path, erosion)

print('convert ok')
