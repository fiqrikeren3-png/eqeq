import cv2
import numpy as np

def create_textured_image(base_color, texture_type='noise'):
    img = np.full((200, 200, 3), base_color, dtype=np.uint8)
    if texture_type == 'noise':
        noise = np.random.normal(0, 25, (200, 200, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif texture_type == 'grain':
        # Improved grain for wood: curved lines and more variation
        for i in range(0, 200, 15):
            offset = np.random.randint(-5, 5)
            cv2.line(img, (0, i + offset), (200, i + offset), (max(base_color[0]-40,0), max(base_color[1]-40,0), max(base_color[2]-20,0)), 2)
        noise = np.random.normal(0, 20, (200, 200, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif texture_type == 'weave':
        # Improved weave for fabric: more irregular
        for i in range(0, 200, 25):
            offset_x = np.random.randint(-3, 3)
            offset_y = np.random.randint(-3, 3)
            cv2.line(img, (0, i + offset_y), (200, i + offset_y), (max(base_color[0]-50,0), max(base_color[1]-50,0), max(base_color[2]-30,0)), 2)
            cv2.line(img, (i + offset_x, 0), (i + offset_x, 200), (max(base_color[0]-50,0), max(base_color[1]-50,0), max(base_color[2]-30,0)), 2)
        noise = np.random.normal(0, 25, (200, 200, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# Kayu (wood) - brownish with grain
cv2.imwrite('dataset/train/kayu/kayu_dummy1.jpg', create_textured_image([120, 80, 40], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy2.jpg', create_textured_image([130, 90, 50], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy3.jpg', create_textured_image([110, 70, 30], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy4.jpg', create_textured_image([125, 85, 45], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy5.jpg', create_textured_image([115, 75, 35], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy6.jpg', create_textured_image([135, 95, 55], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy7.jpg', create_textured_image([105, 65, 25], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy8.jpg', create_textured_image([140, 100, 60], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy9.jpg', create_textured_image([100, 60, 20], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy10.jpg', create_textured_image([145, 105, 65], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy11.jpg', create_textured_image([95, 55, 15], 'grain'))
cv2.imwrite('dataset/train/kayu/kayu_dummy12.jpg', create_textured_image([150, 110, 70], 'grain'))

# Metal - silvery with noise
cv2.imwrite('dataset/train/metal/metal_dummy1.jpg', create_textured_image([200, 200, 210], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy2.jpg', create_textured_image([210, 210, 220], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy3.jpg', create_textured_image([190, 190, 200], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy4.jpg', create_textured_image([205, 205, 215], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy5.jpg', create_textured_image([195, 195, 205], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy6.jpg', create_textured_image([215, 215, 225], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy7.jpg', create_textured_image([185, 185, 195], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy8.jpg', create_textured_image([220, 220, 230], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy9.jpg', create_textured_image([225, 225, 235], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy10.jpg', create_textured_image([180, 180, 190], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy11.jpg', create_textured_image([230, 230, 240], 'noise'))
cv2.imwrite('dataset/train/metal/metal_dummy12.jpg', create_textured_image([175, 175, 185], 'noise'))

# Kain (fabric) - lighter with weave
cv2.imwrite('dataset/train/kain/kain_dummy1.jpg', create_textured_image([160, 160, 170], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy2.jpg', create_textured_image([170, 170, 180], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy3.jpg', create_textured_image([150, 150, 160], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy4.jpg', create_textured_image([165, 165, 175], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy5.jpg', create_textured_image([155, 155, 165], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy6.jpg', create_textured_image([175, 175, 185], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy7.jpg', create_textured_image([145, 145, 155], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy8.jpg', create_textured_image([180, 180, 190], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy9.jpg', create_textured_image([185, 185, 195], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy10.jpg', create_textured_image([140, 140, 150], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy11.jpg', create_textured_image([190, 190, 200], 'weave'))
cv2.imwrite('dataset/train/kain/kain_dummy12.jpg', create_textured_image([135, 135, 145], 'weave'))

print('12 textured dummy images per class created.')
