def detect_text_color(img, mnz_point):
    if not img.ndim == 3:
        raise ValueError('Color detection requires a color image as input.')

    print('Text color detection is not yet implemented ...')
    mnz_point.color_id = 0
