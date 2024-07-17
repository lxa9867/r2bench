
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
import soundfile as sf
import pandas as pd

# def inspect_json(val_type):
#     results = json.loads(f'/mnt/data/refvos/meta_expressions/valid/meta_expressions_{val_type}.json')
#     print(results)

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

def gen_demo_imgs():
    augmenter = ModalityAugmentation()
    for noise_type in list(augmenter.image_noise_functions.keys()):
        img = Image.open('/home/mcg/r2agent/toy_dataset/image/car.png').convert('RGB')
        img = augmenter.apply(img, "image", noise_type, severity=3)
        img = Image.fromarray(np.uint8(img))
        img.save('/home/mcg/r2agent/toy_dataset/noisy_image/'+f'{noise_type}.png')

def create_rvos_dataset(severity=1, dynamic=False):
    with open('/mnt/data/refvos/meta_expressions/valid/meta_expressions.json', 'r') as f:
        results = json.load(f)
    with open('/mnt/data/refvos/meta_expressions/test/meta_expressions.json', 'r') as f:
        results_ = json.load(f)
    valid_test_videos = set(results['videos'].keys())
    test_videos = set(results_['videos'].keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted(list(valid_videos))
    augmenter = ModalityAugmentation()
    with tqdm(total=len(video_list)) as pbar:
        for video in video_list:

            prob = np.random.random()
            # 0.2 both, 0.4 visual, 0.4 textual
            # add visual noise
            noisy_video_root = '/mnt/data/noisy_refvos/valid/JPEGImages'
            save_path = os.path.join(noisy_video_root, video, ).replace('JPEGImages',
                                                                        f'JPEGImages_{"dyn" if dynamic else severity}')
            os.makedirs(save_path, exist_ok=True)
            img_paths = glob.glob(os.path.join(noisy_video_root, video, '*'))
            noise_type = random.choice(list(augmenter.image_noise_functions.keys()))

            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                if dynamic:
                    severity = random.choice([1, 3, 5])
                if prob < 0.6:
                    img = augmenter.apply(img, "image", noise_type, severity=severity)
                    img = Image.fromarray(np.uint8(img))
                img.save(img_path.replace('JPEGImages', f'JPEGImages_{"dyn" if dynamic else severity}'))

            # add text noiose
            if prob >= 0.6 or prob < 0.2:
                for k, v in results['videos'][video]['expressions'].items():
                    sentence = v['exp']
                    if dynamic:
                        severity = random.choice([1, 3, 5])
                    noise_type = random.choice(list(augmenter.text_noise_functions.keys()))
                    try:
                        sentence = augmenter.apply(sentence, 'text', noise_type, severity=severity)
                    except:
                        print(video)
                    results['videos'][video]['expressions'][k] = {'exp': sentence}
            pbar.update(1)

        results = json.dumps(results)
        with open(f'/mnt/data/noisy_refvos/meta_expressions/valid/meta_expressions_{"dyn" if dynamic else severity}.json', 'w') as f:
            f.write(results)


def create_refdavis_dataset(severity=1, dynamic=False):
    with open('/mnt/data/noisy_ref-davis/meta_expressions/valid/meta_expressions.json', 'r') as f:
        results = json.load(f)
    f.close()
    video_list = sorted(list(results['videos'].keys()))
    augmenter = ModalityAugmentation()
    with tqdm(total=len(video_list)) as pbar:
        for video in video_list:
            prob = np.random.random()
            # 0.2 both, 0.4 visual, 0.4 textual
            # add visual noise
            noisy_video_root = '/mnt/data/noisy_ref-davis/valid/JPEGImages'
            save_path = os.path.join(noisy_video_root, video, ).replace('JPEGImages',
                                                                        f'JPEGImages_{"dyn" if dynamic else severity}')
            os.makedirs(save_path, exist_ok=True)
            img_paths = glob.glob(os.path.join(noisy_video_root, video, '*'))

            # saved_paths = glob.glob(os.path.join(save_path, '*'))
            # if len(saved_paths) != 0:
            #     continue
            # print(save_path)
            noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                ori_size = img.size
                if dynamic:
                    severity = random.choice([1, 3, 5])
                if prob < 0.6:
                    img = augmenter.apply(img, "image", noise_type, severity=severity)
                    img = Image.fromarray(np.uint8(img))
                save_size = img.size
                assert ori_size == save_size, f'{ori_size} is not equal to {save_size}'
                img.save(img_path.replace('JPEGImages', f'JPEGImages_{"dyn" if dynamic else severity}'))

            # add text noiose
            if prob >= 0.6 or prob < 0.2:
                for k, v in results['videos'][video]['expressions'].items():
                    sentence = v['exp']
                    if dynamic:
                        severity = random.choice([1, 3, 5])
                    noise_type = random.choice(list(augmenter.text_noise_functions.keys()))
                    try:
                        sentence = augmenter.apply(sentence, 'text', noise_type, severity=severity)
                    except:
                        print(video)
                    results['videos'][video]['expressions'][k] = {'exp': sentence}
            pbar.update(1)
            # except:
            #     print(video)
        results = json.dumps(results)
        with open(
                f'/mnt/data/noisy_ref-davis/meta_expressions/valid/meta_expressions_{"dyn" if dynamic else severity}.json',
                'w') as f:
            f.write(results)
        f.close()

def create_refdavis_ablate_dataset(severity=3):
    augmenter = ModalityAugmentation()
    for noise_type in list(augmenter.image_noise_functions.keys()):
        with open('/mnt/data/noisy_ref-davis/meta_expressions/valid/meta_expressions.json', 'r') as f:
            results = json.load(f)
        f.close()
        video_list = sorted(list(results['videos'].keys()))
        for video in video_list:
            noisy_video_root = '/mnt/data/noisy_ref-davis/valid/JPEGImages'
            save_path = os.path.join(noisy_video_root, video, ).replace('JPEGImages',
                                                                        f'JPEGImages_{noise_type}')
            os.makedirs(save_path, exist_ok=True)
            img_paths = glob.glob(os.path.join(noisy_video_root, video, '*'))
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                ori_size = img.size
                img = augmenter.apply(img, "image", noise_type, severity=severity)
                img = Image.fromarray(np.uint8(img))
                save_size = img.size
                assert ori_size == save_size, f'{ori_size} is not equal to {save_size}'
                img.save(img_path.replace('JPEGImages', f'JPEGImages_{noise_type}'))

        results = json.dumps(results)
        with open(
                f'/mnt/data/noisy_ref-davis/meta_expressions/valid/meta_expressions_{noise_type}.json',
                'w') as f:
            f.write(results)
        f.close()


    for noise_type in list(augmenter.text_noise_functions.keys()):
        with open('/mnt/data/noisy_ref-davis/meta_expressions/valid/meta_expressions.json', 'r') as f:
            results = json.load(f)
        f.close()
        video_list = sorted(list(results['videos'].keys()))
        for video in video_list:
            noisy_video_root = '/mnt/data/noisy_ref-davis/valid/JPEGImages'
            save_path = os.path.join(noisy_video_root, video, ).replace('JPEGImages',
                                                                        f'JPEGImages_{noise_type}')
            os.makedirs(save_path, exist_ok=True)
            img_paths = glob.glob(os.path.join(noisy_video_root, video, '*'))
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                img.save(img_path.replace('JPEGImages', f'JPEGImages_{noise_type}'))
            # add text noiose
            for k, v in results['videos'][video]['expressions'].items():
                sentence = v['exp']
                try:
                    sentence = augmenter.apply(sentence, 'text', noise_type, severity=severity)
                except:
                    print(video)
                results['videos'][video]['expressions'][k] = {'exp': sentence}
        results = json.dumps(results)
        with open(
                f'/mnt/data/noisy_ref-davis/meta_expressions/valid/meta_expressions_{noise_type}.json',
                'w') as f:
            f.write(results)
        f.close()

def create_avs_s3_dataset(severity=1, dynamic=False):
    def stereo_to_mono(audio):
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio

    def get_aud_path(df_split):
        idx = random.choice(list(range(len(df_split))))
        df_one_video = df_split.iloc[idx]
        video_name, category = df_one_video[0], df_one_video[2]
        audio_wav_path = os.path.join(audio_wav_root, subset, category, video_name + '.wav')
        return audio_wav_path

    dataset_path = '/mnt/data/AVSBench'
    subset = 'test'
    df_all = pd.read_csv(os.path.join(dataset_path, 'Single-source/s4_meta_data.csv'), sep=',')
    df_split = df_all[df_all['split'] == subset].sample(frac=1, random_state=1)
    audio_log_mel_root = os.path.join(dataset_path, 's4_data/audio_log_mel')
    audio_wav_root = os.path.join(dataset_path, 's4_data/audio_wav')
    visual_frame_root = os.path.join(dataset_path, 's4_data/visual_frames')
    os.makedirs(os.path.join(dataset_path, f's4_data_{severity}'), exist_ok=True)
    augmenter = ModalityAugmentation()
    for idx in range(len(df_split)):
        prob = np.random.random()
        df_one_video = df_split.iloc[idx]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path = os.path.join(visual_frame_root, subset, category, video_name)
        audio_wav_path = os.path.join(audio_wav_root, subset, category, video_name + '.wav')
        # audio_lm_path = os.path.join(audio_log_mel_root, subset, category, video_name + '.pkl')
        # add image noise
        noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
        img_paths = glob.glob(img_base_path + '/*.png')
        os.makedirs(img_base_path.replace('s4_data', f's4_data_{"dyn" if dynamic else severity}'), exist_ok=True)
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            ori_size = img.size
            if dynamic:
                severity = random.choice([1, 3, 5])
            if prob < 0.6:
                img = augmenter.apply(img, "image", noise_type, severity=severity)
                img = Image.fromarray(np.uint8(img))
            save_size = img.size
            assert ori_size == save_size, f'{ori_size} is not equal to {save_size}'
            img.save(img_path.replace('s4_data', f's4_data_{"dyn" if dynamic else severity}'))
        # add audio noise
        audio_save_path = os.path.join(audio_wav_root, subset, category)
        os.makedirs(audio_save_path.replace('s4_data', f's4_data_{"dyn" if dynamic else severity}'), exist_ok=True)
        samples, samplerate = sf.read(audio_wav_path)
        samples = samples.transpose()
        noise_type = random.choice(list(augmenter.audio_noise_functions.keys()))
        if prob >= 0.6 or prob < 0.2:
            if noise_type == "background_noise":
                samples = stereo_to_mono(samples)
                samples = augmenter.apply(samples, "audio", noise_type, severity=severity, sample_rate=samplerate,
                                                    background_path=get_aud_path(df_split))
                samples = np.stack([samples, samples])
            else:
                samples = augmenter.apply(samples, "audio", noise_type, sample_rate=samplerate, severity=severity)
            sf.write(audio_wav_path.replace('s4_data', f's4_data_{"dyn" if dynamic else severity}'), samples.transpose(), samplerate)
        else:
            sf.write(audio_wav_path.replace('s4_data', f's4_data_{"dyn" if dynamic else severity}'),
                     samples.transpose(), samplerate)

def create_avs_ms3_dataset(severity=1, dynamic=False):
    def stereo_to_mono(audio):
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio

    def get_aud_path(df_split):
        idx = random.choice(list(range(len(df_split))))
        df_one_video = df_split.iloc[idx]
        video_name = df_one_video.iloc[0]
        audio_wav_path = os.path.join(audio_wav_root, subset, video_name + '.wav')
        return audio_wav_path

    dataset_path = '/mnt/data/AVSBench'
    subset = 'test'
    df_all = pd.read_csv(os.path.join(dataset_path, 'Multi-sources/ms3_meta_data.csv'), sep=',')
    df_split = df_all[df_all['split'] == subset].sample(frac=1, random_state=1)
    audio_log_mel_root = os.path.join(dataset_path, 'ms3_data/audio_log_mel')
    audio_wav_root = os.path.join(dataset_path, f'ms3_data/audio_wav/')
    visual_frame_root = os.path.join(dataset_path, f'ms3_data/visual_frames/')
    os.makedirs(os.path.join(dataset_path, f'ms3_data_{severity}'), exist_ok=True)
    augmenter = ModalityAugmentation()
    for idx in range(len(df_split)):
        prob = np.random.random()
        df_one_video = df_split.iloc[idx]
        video_name = df_one_video.iloc[0]
        img_base_path = os.path.join(visual_frame_root, video_name)
        audio_wav_path = os.path.join(audio_wav_root, subset, video_name + '.wav')
        # audio_lm_path = os.path.join(audio_log_mel_root, subset, category, video_name + '.pkl')
        # add image noise
        noise_type = random.choice(list(augmenter.image_noise_functions.keys()))
        img_paths = glob.glob(img_base_path + '/*.png')
        print(img_base_path)
        os.makedirs(img_base_path.replace('ms3_data', f'ms3_data_{"dyn" if dynamic else severity}'), exist_ok=True)
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            ori_size = img.size
            if dynamic:
                severity = random.choice([1, 3, 5])
            if prob < 0.6:
                img = augmenter.apply(img, "image", noise_type, severity=severity)
                img = Image.fromarray(np.uint8(img))
            save_size = img.size
            assert ori_size == save_size, f'{ori_size} is not equal to {save_size}'
            img.save(img_path.replace('ms3_data', f'ms3_data_{"dyn" if dynamic else severity}'))
        # add audio noise
        audio_save_path = os.path.join(audio_wav_root, subset)
        os.makedirs(audio_save_path.replace('ms3_data', f'ms3_data_{"dyn" if dynamic else severity}'), exist_ok=True)
        samples, samplerate = sf.read(audio_wav_path)
        samples = samples.transpose()
        noise_type = random.choice(list(augmenter.audio_noise_functions.keys()))
        if prob >= 0.6 or prob < 0.2:
            if noise_type == "background_noise":
                samples = stereo_to_mono(samples)
                samples = augmenter.apply(samples, "audio", noise_type, severity=severity, sample_rate=samplerate,
                                                    background_path=get_aud_path(df_split))
                samples = np.stack([samples, samples])
            else:
                samples = augmenter.apply(samples, "audio", noise_type, sample_rate=samplerate, severity=severity)
            sf.write(audio_wav_path.replace('ms3_data', f'ms3_data_{"dyn" if dynamic else severity}'), samples.transpose(), samplerate)
        else:
            sf.write(audio_wav_path.replace('ms3_data', f'ms3_data_{"dyn" if dynamic else severity}'),
                     samples.transpose(), samplerate)


def create_avs_ablate_dataset(severity=3,):
    def stereo_to_mono(audio):
        return np.mean(audio, axis=0) if audio.ndim > 1 else audio

    def get_aud_path(df_split):
        idx = random.choice(list(range(len(df_split))))
        df_one_video = df_split.iloc[idx]
        video_name, category = df_one_video[0], df_one_video[2]
        audio_wav_path = os.path.join(audio_wav_root, subset, category, video_name + '.wav')
        return audio_wav_path

    dataset_path = '/mnt/data/AVSBench'
    subset = 'test'
    df_all = pd.read_csv(os.path.join(dataset_path, 'Single-source/s4_meta_data.csv'), sep=',')
    df_split = df_all[df_all['split'] == subset].sample(frac=1, random_state=1)
    audio_log_mel_root = os.path.join(dataset_path, 's4_data/audio_log_mel')
    audio_wav_root = os.path.join(dataset_path, 's4_data/audio_wav')
    visual_frame_root = os.path.join(dataset_path, 's4_data/visual_frames')
    os.makedirs(os.path.join(dataset_path, f's4_data_{severity}'), exist_ok=True)
    augmenter = ModalityAugmentation()
    for noise_type in list(augmenter.audio_noise_functions.keys()):
        if noise_type != 'time_mask':
            continue
        for idx in range(len(df_split)):
            df_one_video = df_split.iloc[idx]
            video_name, category = df_one_video[0], df_one_video[2]
            img_base_path = os.path.join(visual_frame_root, subset, category, video_name)
            audio_wav_path = os.path.join(audio_wav_root, subset, category, video_name + '.wav')
            # audio_lm_path = os.path.join(audio_log_mel_root, subset, category, video_name + '.pkl')
            # add image noise
            img_paths = glob.glob(img_base_path + '/*.png')
            os.makedirs(img_base_path.replace('s4_data', f's4_data_{noise_type}'), exist_ok=True)
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                img.save(img_path.replace('s4_data', f's4_data_{noise_type}'))
            # add audio noise
            audio_save_path = os.path.join(audio_wav_root, subset, category)
            os.makedirs(audio_save_path.replace('s4_data', f's4_data_{noise_type}'), exist_ok=True)
            samples, samplerate = sf.read(audio_wav_path)
            samples = samples.transpose()

            if noise_type == "background_noise":
                samples = stereo_to_mono(samples)
                samples = augmenter.apply(samples, "audio", noise_type, severity=severity, sample_rate=samplerate,
                                                    background_path=get_aud_path(df_split))
                samples = np.stack([samples, samples])
            else:
                samples = augmenter.apply(samples, "audio", noise_type, sample_rate=samplerate, severity=severity)
            sf.write(audio_wav_path.replace('s4_data', f's4_data_{noise_type}'), samples.transpose(), samplerate)

if __name__ == '__main__':
    dataset = 'avs_ablate' #'ref-davis_ablate'
    np.random.seed(245)
    random.seed(245)
    print(f'creating noisy version of {dataset}')
    if dataset == 'refvos':
        # create_rvos_dataset(dynamic=True)
        for severity in [3,]:
            create_rvos_dataset(severity=severity, dynamic=False)
    elif dataset == 'ref-davis':
        create_refdavis_dataset(dynamic=True)
        for severity in [1, 3, 5]:
            create_refdavis_dataset(severity=severity, dynamic=False)
    elif dataset == 'ref-davis_ablate':
        create_refdavis_ablate_dataset(severity=3)
    elif dataset =='avs_s4':
        create_avs_s3_dataset(severity=1)
        create_avs_s3_dataset(severity=3)
        create_avs_s3_dataset(severity=5)
    elif dataset =='avs_ms3':
        create_avs_ms3_dataset(severity=1)
        create_avs_ms3_dataset(severity=3)
        create_avs_ms3_dataset(severity=5)
    elif dataset =='avs_ablate':
        create_avs_ablate_dataset(severity=5)
    elif dataset == 'demo':
        gen_demo_imgs()
    else:
        raise NotImplementedError
    # inspect_json('val')
    create_davis_dataset()
    # create_rvos_dataset(dynamic=True)
    # for severity in [0, 2, 4,]:
    #     create_rvos_dataset(severity=severity, dynamic=False)


