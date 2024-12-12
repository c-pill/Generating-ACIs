from matplotlib import pyplot as plt
import cv2
import os


results_folder = './censor results'

original_image = 'Background.png'
original_name = original_image[:-4]

# don't forget .png
img_file = 'Background.png'

# use for scrambled images
# if original image set to None
# if method != None not used
algorithm = 'EPGACI' #None 
percent   = None 

# use for 'Gaussian Blurred' images
# if neither set to None
method = 'Gaussian Blurred' #None

if algorithm is not None and percent is not None and method is None:
    # if scrambled image:
    img = cv2.imread(f"./alg results/{original_name} results/{algorithm}/{percent}%/{img_file}", cv2.IMREAD_GRAYSCALE)
    # path = os.path.realpath(f"./results/{original_name} results/{algorithm}/{percent}%/{img_file}")
    # os.startfile(path)
elif method is not None:
    # if blurred or pixelated image
    img = cv2.imread(f"./censor results/{method}/{original_name}/{img_file}", cv2.IMREAD_GRAYSCALE)
else:
    # if original image:
    img_file = original_image
    img = cv2.imread(f"./images/{img_file}", cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

plt.imshow(edges, cmap='gray')
plt.title(img_file), plt.xticks([]), plt.yticks([])

# create directory for results if one doesn't exist
if not os.path.exists(results_folder): os.mkdir(results_folder)
path = f"{results_folder}/Canny Edge Results"
if not os.path.exists(path): os.mkdir(path)
path = f"{path}/{original_name}"
if not os.path.exists(path): os.mkdir(path)
if algorithm is not None and percent is not None and method is None:
    path = f"{path}/{algorithm}"
    if not os.path.exists(path): os.mkdir(path)
elif method is not None:
    path = f"{path}/{method}"
    if not os.path.exists(path): os.mkdir(path)

# saves image
result_name = f"{path}/{img_file}"
plt.savefig(result_name)

path = os.path.realpath(path)
os.startfile(path)

# if image doesn't show remove comment and run
# plt.show()