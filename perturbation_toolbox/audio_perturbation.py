import os
import audiomentations
import soundfile as sf
import numpy as np



def gain(x, sample_rate, severity=1):
    gain_values_db = [
        (-3, 3),  # Very Low Severity: Slight volume adjustment
        (-6, 6),  # Low Severity: Moderate volume adjustment
        (-9, 9),  # Medium Severity: Noticeable volume change
        (-12, 12), # High Severity: Significant volume change
        (-15, 15), # Very High Severity: Extreme volume change
    ]
    
    severity = max(1, min(severity, 5)) - 1
    min_gain_db, max_gain_db = gain_values_db[severity]
    
    # Create the Gain transform with the selected gain value
    transform = audiomentations.Gain(
        min_gain_in_db=min_gain_db,
        max_gain_in_db=max_gain_db,
        p=1.0
    )
    augmented_audio = transform(samples=x, sample_rate=sample_rate)
    return augmented_audio


def background_noise(x, sample_rate, severity=1, background_path=None):
    if not background_path:
        return x
    
    snr_values = [
        (25, 30),  # Very Low Severity
        (20, 25),  # Low Severity
        (15, 20),  # Medium Severity
        (10, 15),  # High Severity
        (5, 10)    # Very High Severity
    ]
    
    severity = max(1, min(severity, len(snr_values))) - 1
    
    transform = audiomentations.AddBackgroundNoise(
        sounds_path=background_path,  # Update with the actual path
        min_snr_in_db=snr_values[severity][0],
        max_snr_in_db=snr_values[severity][1],
        noise_transform=audiomentations.PolarityInversion(),
        p=1.0
    )
    return transform(x, sample_rate = sample_rate)



def air_absorption(audio_samples, sample_rate, severity=1):
    severity_levels = {
        1: {
            "min_humidity": 30.0, "max_humidity": 50.0,  # Lower humidity range
            "min_distance": 10.0, "max_distance": 30.0,  # Closer microphone-source distance
        },
        2: {
            "min_humidity": 50.0, "max_humidity": 60.0,  # Moderate humidity increase
            "min_distance": 30.0, "max_distance": 50.0,  # Moderate microphone-source distance
        },
        3: {
            "min_humidity": 60.0, "max_humidity": 70.0,  # Further increase in humidity
            "min_distance": 50.0, "max_distance": 70.0,  # Further increase in distance
        },
        4: {
            "min_humidity": 70.0, "max_humidity": 80.0,  # Higher humidity range
            "min_distance": 70.0, "max_distance": 90.0,  # Larger microphone-source distance
        },
        5: {
            "min_humidity": 80.0, "max_humidity": 90.0,  # Maximum humidity range
            "min_distance": 90.0, "max_distance": 100.0,  # Maximum microphone-source distance
        },
    }

    severity = max(1, min(severity, 5))
    params = severity_levels[severity]

    transform = audiomentations.AirAbsorption(
        **params,
        p=1.0
    )
    
    augmented_samples = transform(samples=audio_samples, sample_rate=sample_rate)
    
    return augmented_samples



def room_simulator(audio_samples, sample_rate, severity=1):
    # Map severity levels to room simulator parameters
    severity_levels = {
        1: {
            "min_size_x": 3.6, "max_size_x": 4.0,
            "min_size_y": 3.6, "max_size_y": 3.7,
            "min_size_z": 2.4, "max_size_z": 2.5,
            "min_absorption_value": 0.4, "max_absorption_value": 0.6,
            "min_target_rt60": 0.15, "max_target_rt60": 0.3,
            "calculation_mode": "absorption",
        },
        2: {
            "min_size_x": 4.0, "max_size_x": 4.5,
            "min_size_y": 3.7, "max_size_y": 3.8,
            "min_size_z": 2.5, "max_size_z": 2.6,
            "min_absorption_value": 0.3, "max_absorption_value": 0.5,
            "min_target_rt60": 0.3, "max_target_rt60": 0.4,
            "calculation_mode": "absorption",
        },
        3: {
            "min_size_x": 4.5, "max_size_x": 5.0,
            "min_size_y": 3.8, "max_size_y": 3.9,
            "min_size_z": 2.6, "max_size_z": 2.7,
            "min_absorption_value": 0.2, "max_absorption_value": 0.4,
            "min_target_rt60": 0.4, "max_target_rt60": 0.5,
            "calculation_mode": "absorption",
        },
        4: {
            "min_size_x": 5.0, "max_size_x": 5.5,
            "min_size_y": 3.9, "max_size_y": 4.0,
            "min_size_z": 2.7, "max_size_z": 2.8,
            "min_absorption_value": 0.1, "max_absorption_value": 0.3,
            "min_target_rt60": 0.5, "max_target_rt60": 0.6,
            "calculation_mode": "absorption",
        },
        5: {
            "min_size_x": 5.5, "max_size_x": 5.6,
            "min_size_y": 4.0, "max_size_y": 4.1,
            "min_size_z": 2.8, "max_size_z": 3.0,
            "min_absorption_value": 0.075, "max_absorption_value": 0.2,
            "min_target_rt60": 0.6, "max_target_rt60": 0.8,
            "calculation_mode": "rt60",
        },
    }

    severity = max(1, min(severity, 5))
    params = severity_levels[severity]

    # Initialize the RoomSimulator transform with selected parameters
    transform = audiomentations.RoomSimulator(
        **params,
        p=1.0
    )
    
    # Apply the transform to the audio samples
    augmented_samples = transform(samples=audio_samples, sample_rate=sample_rate)
    
    return augmented_samples




def gaussian_noise(x, sample_rate, severity = 1):
    severity = max(1, min(severity, 5))
    c = [0.0001, 0.0025, 0.005, 0.01, 0.015, 0.02]
    
    transform = audiomentations.AddGaussianNoise(
        min_amplitude=c[severity-1],
        max_amplitude=c[severity],
        p=1.0
    )
    return transform(x, sample_rate = sample_rate)



def peaking_filter(audio_samples, sample_rate, severity=1):
    # Enhanced severity levels for the PeakingFilter to ensure a more dramatic effect
    severity_levels = {
        1: {
            "min_center_freq": 200, "max_center_freq": 1500,
            "min_gain_db": -12, "max_gain_db": 12,
            "min_q": 1.0, "max_q": 2.0,
        },
        2: {
            "min_center_freq": 500, "max_center_freq": 2500,
            "min_gain_db": -18, "max_gain_db": 18,
            "min_q": 1.5, "max_q": 2.5,
        },
        3: {
            "min_center_freq": 1000, "max_center_freq": 3500,
            "min_gain_db": -24, "max_gain_db": 24,
            "min_q": 2.0, "max_q": 3.0,
        },
        4: {
            "min_center_freq": 1500, "max_center_freq": 5500,
            "min_gain_db": -30, "max_gain_db": 30,
            "min_q": 2.5, "max_q": 3.5,
        },
        5: {
            "min_center_freq": 2000, "max_center_freq": 7500,
            "min_gain_db": -36, "max_gain_db": 36,
            "min_q": 3.0, "max_q": 4.0,
        },
    }
    
    severity = max(1, min(severity, 5))
    params = severity_levels[severity]
    
    transform = audiomentations.PeakingFilter(
        min_center_freq=params["min_center_freq"],
        max_center_freq=params["max_center_freq"],
        min_gain_db=params["min_gain_db"],
        max_gain_db=params["max_gain_db"],
        min_q=params["min_q"],
        max_q=params["max_q"],
        p=1.0  # Ensure the effect is applied
    )
    
    augmented_samples = transform(samples=audio_samples, sample_rate=sample_rate)
    
    return augmented_samples



# need to find source
def impulse_response(x, sample_rate, severity=1):
    severity = max(1, min(severity, 5))
    
    ir_path = os.path.join(f"perturbation/data/severity_{severity}.wav")
    transform = audiomentations.ApplyImpulseResponse(
        ir_path=ir_path,
        p=1.0
    )
    
    # Apply the transform to the input audio x
    augmented_audio = transform(samples=x, sample_rate=sample_rate)
    return augmented_audio



def time_mask(audio_samples, sample_rate, severity=1):
    severity_levels = {
        1: {"min_band_part": 0.05, "max_band_part": 0.1, "fade": True},  # Slightly more severe
        2: {"min_band_part": 0.1, "max_band_part": 0.2, "fade": True},    # Noticeably affects the audio
        3: {"min_band_part": 0.2, "max_band_part": 0.3, "fade": True},    # Significantly masks the audio
        4: {"min_band_part": 0.3, "max_band_part": 0.4, "fade": True},    # Very impactful, masks a large part
        5: {"min_band_part": 0.4, "max_band_part": 0.5, "fade": True},    # Extremely severe, half of the audio
    }
    
    severity = max(1, min(severity, 5))
    params = severity_levels[severity]
    
    transform = audiomentations.TimeMask(
        min_band_part=params["min_band_part"],
        max_band_part=params["max_band_part"],
        fade=params["fade"],
        p=1.0
    )
    
    augmented_samples = transform(samples=audio_samples, sample_rate=sample_rate)
    
    return augmented_samples



def tanh_distortion(audio_samples, sample_rate, severity=1):
    distortion_ranges = [
        (0.01, 0.1),  # Very Low Severity
        (0.1, 0.2),   # Low Severity
        (0.2, 0.4),   # Medium Severity
        (0.4, 0.6),   # High Severity
        (0.6, 0.7),   # Very High Severity
    ]
    
    severity_index = max(1, min(severity, 5)) - 1
    min_distortion, max_distortion = distortion_ranges[severity_index]
    
    transform = audiomentations.TanhDistortion(
        min_distortion=min_distortion,
        max_distortion=max_distortion,
        p=1.0  # Apply the effect with certainty
    )
    
    augmented_samples = transform(samples=audio_samples, sample_rate=sample_rate)
    return augmented_samples



def mp3_compression(audio_samples, sample_rate, severity=1):
    severity_levels = {
        1: {"min_bitrate": 128, "max_bitrate": 192},  # Very Low Compression (High Quality)
        2: {"min_bitrate": 96, "max_bitrate": 128},   # Low Compression
        3: {"min_bitrate": 64, "max_bitrate": 96},    # Moderate Compression
        4: {"min_bitrate": 32, "max_bitrate": 64},    # High Compression
        5: {"min_bitrate": 8, "max_bitrate": 32},     # Very High Compression (Low Quality)
    }

    severity = max(1, min(severity, 5))
    params = severity_levels[severity]

    transform = audiomentations.Mp3Compression(
        min_bitrate=params["min_bitrate"],
        max_bitrate=params["max_bitrate"],
        p=1.0
    )
    
    # Apply the transform to the audio samples
    # Note: Mp3Compression transform operates differently and might not directly accept numpy arrays.
    # The following line is conceptual and might need adjustment based on actual implementation details.
    augmented_samples = transform(samples=audio_samples, sample_rate=sample_rate)
    
    return augmented_samples




def lowpass_filter(audio, sample_rate, severity=1):
    cutoff_frequency_ranges = [
        (8000, 16000),  # Very Low Severity
        (4000, 8000),   # Low Severity
        (2000, 4000),   # Medium Severity
        (1000, 2000),   # High Severity
        (500, 1000)     # Very High Severity
    ]
    
    severity = max(1, min(severity, 5)) - 1
    min_cutoff_frequency, max_cutoff_frequency = cutoff_frequency_ranges[severity]
    
    transform = audiomentations.LowPassFilter(
        min_cutoff_freq=min_cutoff_frequency, 
        max_cutoff_freq=max_cutoff_frequency, 
        p=1.0
    )
    
    filtered_audio = transform(samples=audio, sample_rate=sample_rate)
    return filtered_audio



def highpass_filter(audio, sample_rate, severity=1):
    cutoff_frequency_ranges = [
        (100, 200),  # Very Low Severity
        (200, 400),  # Low Severity
        (400, 800),  # Medium Severity
        (800, 1600), # High Severity
        (1600, 3200) # Very High Severity
    ]
    
    severity = max(1, min(severity, 5)) - 1
    min_cutoff_frequency, max_cutoff_frequency = cutoff_frequency_ranges[severity]
    
    transform = audiomentations.HighPassFilter(
        min_cutoff_freq=min_cutoff_frequency, 
        max_cutoff_freq=max_cutoff_frequency, 
        p=1.0
    )
    
    filtered_audio = transform(samples=audio, sample_rate=sample_rate)
    return filtered_audio



def stereo_to_mono(audio):
    return np.mean(audio, axis=0) if audio.ndim > 1 else audio


if __name__ == '__main__':

    import librosa
    import soundfile as sf
    import os


    # Load the audio file
    file_path = 'data/sample_0.wav'  # Update this path to where your sample.wav is located
    background = 'data/sample_1.wav'
    audio, sample_rate = librosa.load(file_path, sr=None, mono=False, dtype='float32')  # Load as stereo, preserve dtype

    # Define your test functions and their respective output folders
    test_functions_and_folders = {
        lowpass_filter: "output_lowpass",
        highpass_filter: "output_highpass",
        gain: "output_gain",
        mp3_compression: "output_mp3compression",
        room_simulator: "output_room",
        air_absorption: "output_airabsorption",
        background_noise: "output_backgroundnoise",
        gaussian_noise: "output_gaussian",
        tanh_distortion: "output_tanh",
        peaking_filter: "output_peak",
        impulse_response: "output_impulse",
        time_mask: "output_timemask"
    }

    # Create output folders if they don't exist
    for _, folder in test_functions_and_folders.items():
        os.makedirs(folder, exist_ok=True)

    # Iterate over each function, severity level, and save the processed audio
    for func, folder in test_functions_and_folders.items():
        print(f"Testing {func.__name__}...")
        for severity in range(1, 6):
            print(f" - Severity {severity}")
            try:
                # Process the audio with the current function and severity
                if func == background_noise:
                    mono_audio = stereo_to_mono(audio)
                    augmented_audio = func(mono_audio, sample_rate, severity, background)
                    augmented_audio = np.stack([augmented_audio, augmented_audio])
                else:
                    augmented_audio = func(audio, sample_rate, severity)

                # File naming and saving
                output_file_path = f"{folder}/{func.__name__}_severity_{severity}.wav"
                sf.write(output_file_path, augmented_audio.T, sample_rate)  # Transpose array for soundfile compatibility
                print(f"   Success: Saved to {output_file_path}")

            except Exception as e:
                print(f"   Error for {func.__name__} at severity {severity}: {e}")

    print("Testing completed.")
