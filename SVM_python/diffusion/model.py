import math
import cv2
import numpy as np

def generate_betas(beta_start=0.0001, beta_end=0.02, L=50):
    betas = [0] * L
    
    b = beta_start
    a = (beta_end - b) / (L - 1)

    for i in range(L):
        betas[i] = a * i + b

    return betas

betas = generate_betas(L=10)
img = cv2.imread(r"D:\Computer vision\Images\archive (3)\leedsbutterfly\images\0010001.png")
img = cv2.resize(img, (100, 100)).astype(np.float64) / 255

alpha_accum = 1
noised_data = [None] * 11
noised_data[0] = img

for i in range(1, 11):
    # sqrt(alpha) -> alpha * alpha moi -> sqrt()
    alpha_accum *= (1 - betas[i - 1])

    noise = np.random.normal(size=noised_data[0].shape)
    
    noised_data[i] = math.sqrt(alpha_accum) * noised_data[0] \
        + math.sqrt(1 - alpha_accum) * noise
    
for img in noised_data:
    cv2.imshow('test', cv2.resize(img, (100, 100)))
    cv2.waitKey(0)

