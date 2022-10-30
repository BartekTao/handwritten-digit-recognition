import os
import cv2

origin_path = 'origin'
resized_path = 'resized'
ext = '.png'

print('start converting...')
for i in range(3,6):
    from_dir_path = os.path.join(origin_path, str(i))
    to_dir_path = os.path.join(resized_path, str(i))
    os.makedirs(to_dir_path, exist_ok=True)

    pics = sorted(os.listdir(from_dir_path))
    print(f'number[{i}] pictures = {len(pics)}')

    for i, pic in enumerate(pics):
        from_path = os.path.join(from_dir_path, pic)
        to_path = os.path.join(to_dir_path, f'{i}{ext}')

        image = cv2.imread(from_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(to_path, image)

print('convert ok')
