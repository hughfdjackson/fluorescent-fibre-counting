def snip(image):
    """
    Snip out the area of interest from a successfully orientated image.
    """

    height, width = image.shape[:2]

    left = int(width * (1.0 / 18.0))
    right = int(width * (17.0 / 18.0))
    bottom = int(height * (10.5 / 13.5))
    top = int(height * (3.5 / 13.5))

    return image[top:bottom, left:right, :]