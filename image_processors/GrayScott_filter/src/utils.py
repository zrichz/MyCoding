def sharpen_image(image):
    from PIL import ImageFilter
    return image.filter(ImageFilter.SHARPEN)

def blur_image(image):
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius=1))
