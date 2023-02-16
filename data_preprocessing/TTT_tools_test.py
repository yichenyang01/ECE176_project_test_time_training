import TTT_tools as tt
import cv2
from PIL import Image

img1 = cv2.imread('../datasets/test_img.JPEG')
img_gn = tt.add_gaussian_noise(img1)
img_sn = tt.add_shot_noise(img1, lam=100)
img_mb = tt.add_motion_blur(img1, kernel_size=11, direction='v')
img_db = tt.add_defocus_blur(img1, radius=3)
img_b = tt.add_brightness(img1, brightness_factor=1.7)

result1 = cv2.hconcat([img1, img_gn])
cv2.imshow("Gaussian noise", result1)
cv2.waitKey(0)
cv2.destroyAllWindows()


result2 = cv2.hconcat([img1, img_sn])
cv2.imshow("Shot noise", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()

result3 = cv2.hconcat([img1, img_mb])
cv2.imshow("Motion blur", result3)
cv2.waitKey(0)
cv2.destroyAllWindows()

result4 = cv2.hconcat([img1, img_db])
cv2.imshow("Defocus blur", result4)
cv2.waitKey(0)
cv2.destroyAllWindows()

result5 = cv2.hconcat([img1, img_b])
cv2.imshow("Defocus blur", result5)
cv2.waitKey(0)
cv2.destroyAllWindows()