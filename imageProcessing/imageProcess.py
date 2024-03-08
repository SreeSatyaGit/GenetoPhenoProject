image_dir = './images/'  # Path to the folder containing the images
image_files = os.listdir(image_dir)
images = np.empty((len(image_files), 480, 640,3), dtype=np.uint8)
ddepth = cv2.CV_16S
print('ddepth : ',ddepth)
kernel_size = 1
scale_percent = 100 # percent of original size

for i, image_file in enumerate(image_files):
    # Read the image
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_3c = cv2.merge([src_gray] * 3)
    resized = cv2.resize(gray_image_3c, dim, interpolation = cv2.INTER_AREA)
    dst = cv2.Laplacian(resized, ddepth, ksize=kernel_size)
    images[i] = dst