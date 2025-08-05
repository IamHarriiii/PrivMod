import numpy as np

def clip_gradients(weights, max_norm=1.0):
    norms = [np.linalg.norm(w) for w in weights]
    scale_factor = min(1.0, max_norm / (max(norms) + 1e-6))
    return [w * scale_factor for w in weights]

def add_dp_noise(weights, noise_multiplier=1.2):
    noisy_weights = []
    for w in weights:
        noise = np.random.normal(0, noise_multiplier, w.shape)
        noisy_weights.append(w + noise)
    return noisy_weights