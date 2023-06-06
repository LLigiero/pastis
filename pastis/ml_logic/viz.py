

def reshape_patch_spectra(img):
    """Utility function to reshape patch shape from k*128*128 to 128*128*k.
    """
    reshaped_image = img.swapaxes(0,2).swapaxes(0,1)
    return reshaped_image

def get_rgb(time_series, t_show=-1):
    """Utility function to get a displayable rgb image
       from a Sentinel-2 time series.
    """
    image = time_series[t_show, [2,1,0]] #from the time series, we get the t_show element
                                         #and 3 bands [a,b,c] from the 10 available. [a,b,c]

    # Normalize image
    max_value = image.max(axis=(1,2))
    min_value = image.min(axis=(1,2))
    image_normalized = (image - min_value[:,None,None])/(max_value - min_value)[:,None,None]

    rgb_image = reshape_patch_spectra(image_normalized)
    return rgb_image

def get_radar(time_series, t_show=-1):
    """Utility function to get a displayable rgb image
       from a Sentinel-1 RADAR time series.
    """
    image = time_series[t_show, [0,1,2]] #from the time series, we get the t_show element
                                         #and 3 bands [a,b,c] from the 10 available. [a,b,c]

    # Normalize image
    max_value = image.max(axis=(1,2))
    min_value = image.min(axis=(1,2))
    image_normalized = (image - min_value[:,None,None])/(max_value - min_value)[:,None,None]

    radar_image = reshape_patch_spectra(image_normalized)
    return radar_image
