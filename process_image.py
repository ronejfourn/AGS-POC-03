from PIL import Image, ImageOps

def grayscale_invert(img):
    img = img.convert('L')
    return ImageOps.invert(img)

def resize_image(img, size):
    return img.resize(size)

def process_image(img):
    img = Image.open(img)
    img = grayscale_invert(img)
    img = resize_image(img, (28, 28))
    return img
