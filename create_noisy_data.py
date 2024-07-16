import numpy as np
import json
import glob
import cv2
import glob
from tqdm import tqdm
import os
from perturbation import ModalityAugmentation
from PIL import Image
import random

def inspect_json(val_type):
    results = json.loads(f'anns/refcoco/{val_type}.json')
    print(results)

def create_refcoco_dataset(severity=1, noisy_image_root='', ms_image_root=''):
    with open('/ocean/projects/cis220031p/kqiu/anns/refcoco/val.json', 'r') as f:
        results = json.load(f)
    f.close()
    augmenter = ModalityAugmentation()
    for res in tqdm(results):
        prob = np.random.random()
        # # 0.2 both, 0.4 visual, 0.4 textual
        if prob < 0.6:
            noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
            save_path = res['img_name'].replace('.jpg', f'_{severity}.jpg')
            img = Image.open(ms_image_root,res['img_name']).convert('RGB')
            img = augmenter.apply(img, "image", noise_type, severity=severity)
            img = Image.fromarray(np.uint8(img))
            img.save(os.path.join(noisy_image_root, save_path))
            res['img_name'] = save_path
        if prob > 0.4:
            for sentence in res['sentences']:
                noise_type = random.choice(list(augmenter.text_noise_functions.keys()))
                sentence['sent'] = augmenter.apply(sentence['sent'], 'text', noise_type, severity=severity)

    results = json.dumps(results)
    with open(f'/ocean/projects/cis220031p/kqiu/anns/refcoco/val_{severity}.json', 'w') as f:
        f.write(results)
    f.close()

def create_davis_dataset(severity=1, dynamic=False):
    augmenter = ModalityAugmentation()
    video_list = sorted(glob.glob('DAVIS/JPEGImages/480p/*'))
    for video in video_list:
        for image_path in sorted(glob.glob(f'{video}/*')):
            noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
            if dynamic:
                severity = random.choice([0, 2, 4])
            save_path = image_path.replace('DAVIS', 'DAVIS_dyn' if dynamic else f'DAVIS_{severity}')
            img = Image.open(img_path).convert('RGB')
            img = augmenter.apply(img, "image", noise_type, severity=severity)
            img.save(save_path)

def create_youtube_vos_dataset(severity=1, dynamic=False):
    augmenter = ModalityAugmentation()
    video_list = sorted(glob.glob('valid/JPEGImages/*'))
    for video in video_list:
        for image_path in sorted(glob.glob(f'{video}/*')):
            noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
            if dynamic:
                severity = random.choice([0, 2, 4])
            save_path = image_path.replace('JPEGImages', 'JPEGImages_dyn' if dynamic else f'JPEGImages_{severity}')
            img = Image.open(img_path).convert('RGB')
            img = augmenter.apply(img, "image", noise_type, severity=severity)
            img.save(save_path)
    
def create_rvos_dataset(severity=1, dynamic=False):
    with open('/mnt/data/refvos/meta_expressions/valid/meta_expressions.json', 'r') as f:
        results = json.load(f)
    f.close()
    video_list = list(results['videos'].keys())
    augmenter = ModalityAugmentation()
    with tqdm(total=len(video_list)) as pbar:
        for video in video_list:
            prob = np.random.random()
            # 0.2 both, 0.4 visual, 0.4 textual
            # add visual noise
            noisy_video_root = '/mnt/data/noisy_refvos/valid/JPEGImages'
            if prob < 0.6:
                img_paths = glob.glob(os.path.join(noisy_video_root, video, '*'))
                noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
                save_path = os.path.join(noisy_video_root, video,).replace('JPEGImages',
                                                                           f'JPEGImages_{"dyn" if dynamic else severity}')
                # print(save_path)
                os.makedirs(save_path, exist_ok=True)
                for img_path in img_paths:
                    img = Image.open(img_path).convert('RGB')
                    if dynamic:
                        severity = random.choice([0, 2, 4])
                    img = augmenter.apply(img, "image", noise_type, severity=severity)
                    img = Image.fromarray(np.uint8(img))
                    img.save(img_path.replace('JPEGImages', f'JPEGImages_{"dyn" if dynamic else severity}'))

            # add text noiose
            if prob >= 0.6 or prob < 0.2:
                for k, v in results['videos'][video]['expressions'].items():
                    sentence = v['exp']
                    if dynamic:
                        severity = random.choice([0, 2, 4])
                    noise_type = random.choice(list(augmenter.text_noise_functions.keys()))
                    sentence = augmenter.apply(sentence, 'text', noise_type, severity=severity)
                    results['videos'][video]['expressions'][k] = {'exp': sentence}
            pbar.update(1)
        results = json.dumps(results)
        with open(f'/mnt/data/noisy_refvos/meta_expressions/valid/meta_expressions_{"dyn" if dynamic else severity}.json', 'w') as f:
            f.write(results)
        f.close()


if __name__ == '__main__':
    # inspect_json('val')
    create_davis_dataset()
    # create_rvos_dataset(dynamic=True)
    # for severity in [0, 2, 4,]:
    #     create_rvos_dataset(severity=severity, dynamic=False)

