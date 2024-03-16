from .sensor_perturbation_codes.perturb_rgb import *
import soundfile as sf
from .text_perturbation import *
from .audio_perturbation import *

class ModalityAugmentation(ABC):
    def __init__(self, **params):
        self.params = params
        self.text_perturber = Text_Perturber()
        self.audio_noise_functions = {
                "output_lowpass": lowpass_filter,
                "highpass_filter": highpass_filter,
                "gain": gain,
                "mp3_compression": mp3_compression,
                "room_simulator": room_simulator,
                "air_absorption": air_absorption,
                "background_noise": background_noise,
                "gaussian_noise": gaussian_noise,
                "tanh_distortion": tanh_distortion,
                "peaking_filter": peaking_filter,
                "impulse_response": impulse_response,
                "time_mask": time_mask,
            }
        self. image_noise_functions = {
                "gaussian_noise": gaussian_noise_,
                "shot_noise": shot_noise,
                "impulse_noise": impulse_noise,
                "speckle_noise": speckle_noise,
                "gaussian_blur": gaussian_blur,
                "glass_blur": glass_blur,
                "defocus_blur": defocus_blur,
                "motion_blur": motion_blur,
                "zoom_blur": zoom_blur,
                "fog": fog,
                "frost": frost,
                "snow": snow,
                "spatter": spatter,
                "contrast": contrast,
                "brightness": brightness,
                "saturate": saturate,
                "jpeg_compression": jpeg_compression,
                "pixelate": pixelate,
                "elastic_transform": elastic_transform
            }
        self.text_noise_functions = {
                "misspell": self.text_perturber.misspell,
                "insert_punctuation_in_words": self.text_perturber.insert_punctuation_in_words,
                "grammer": self.text_perturber.grammar,
                "delete_random_characters": self.text_perturber.delete_random_characters
            }

    def _get_perturbation(self, x, modality, noise_type, **params):
        if modality == "audio":
            assert noise_type in self.audio_noise_functions.keys()
            return self.audio_noise_functions[noise_type](x, **params)
        elif modality == "image":

            assert noise_type in self.image_noise_functions.keys()
            return self.image_noise_functions[noise_type](x, **params)
        elif modality == "text":

            assert noise_type in self.text_noise_functions.keys()
            return self.text_noise_functions[noise_type](x, **params)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def apply(self, x, modality, noise_type, sample_rate=None, **params):
        if modality == "audio" and sample_rate is not None:
            return self._get_perturbation(x, modality, noise_type, sample_rate=sample_rate, **params)
        elif modality in ["image", "text"]:
            return self._get_perturbation(x, modality, noise_type, **params)
        else:
            raise ValueError(f"Sample rate must be provided for audio data")

if __name__ == '__main__':
    augmenter = ModalityAugmentation()
    # Example of using the class for audio
    samples, samplerate = sf.read('data/sample_0.wav')
    samples = samples.transpose()
    for noise_type in augmenter.audio_noise_functions.keys():
        if noise_type == "background_noise":
            samples = stereo_to_mono(samples)
            augmented_samples = augmenter.apply(samples, "audio", noise_type, severity=1, sample_rate=samplerate, background_path='data/sample_0.wav')
            augmented_samples = np.stack([augmented_samples, augmented_samples])
        else:
            augmented_samples = augmenter.apply(samples, "audio", noise_type, sample_rate=samplerate, severity=1)
        sf.write('augmented_samples.wav', augmented_samples.transpose(), samplerate)

    # Example of using the class for image
    image = Image.open('sample.png')
    if image.mode == 'RGBA':
        # Convert to RGB
        image = image.convert('RGB')

    for noise_type in augmenter.image_noise_functions.keys():
        augmented_image = augmenter.apply(image, "image", noise_type, severity=1)
        augmented_image_pil = Image.fromarray(np.uint8(augmented_image))
        augmented_image_pil.save('augmented_image.png')

    # Example of using the class for text
    for noise_type in augmenter.text_noise_functions.keys():
        answer = augmenter.apply("a dog that is running", "text", noise_type, severity=2)
        print(answer)

