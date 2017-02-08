import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def color_mask(hsv,img, min_thresh, max_thresh):

    mask = cv2.inRange(hsv, min_thresh, max_thresh)
    res = cv2.bitwise_and(img,img, mask= mask)
    return mask, res

def roi(img, width_percent=0.10, height_percent=0.65):

    mask = np.zeros_like(img)
    h, w = img.shape[0:2]
    y = int(h*height_percent)
    x_1, x_2, x_3, x_4 = int(w*width_percent), int(w*.4), int(w*.6), int(w*(1-width_percent)),
    vertices = np.array([[(x_1,h),(x_2, y), (x_3, y), (x_4,h)]], dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def lapacian_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):

    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    lapacian = cv2.Laplacian(gray, cv2.CV_64F,ksize=sobel_kernel)
    abs_lapacian = np.absolute(lapacian)
    scaled_lapacian = np.uint8(255*(abs_lapacian/np.max(abs_lapacian)))

    binary_output =  np.zeros_like(abs_lapacian)
    binary_output[(abs_lapacian >= mag_thresh[0]) & (abs_lapacian <= mag_thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

def abs_sobel_thresh(img, orient='x',sobel_kernel=3, thresh_min=0, thresh_max=255):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*(abs_sobel/np.max(abs_sobel)))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


if __name__ == '__main__':

    #img = mpimg.imread('test_images/0.jpg')
    #img = mpimg.imread('test_images/straight_lines2.jpg')
    img = mpimg.imread('test_images/test5.jpg')
    img = gaussian_blur(img, 7)


    HLS = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HLS)
    L = HLS[:,:,1]
    S = HLS[:,:,2]
    _, S_binary = cv2.threshold(S.astype('uint8'), 175, 255, cv2.THRESH_BINARY)

    k_size = 3
    s_lap_binary = lapacian_threshold(S, sobel_kernel=k_size, mag_thresh=(25, 255))
    l_lap_binary = lapacian_threshold(L, sobel_kernel=k_size, mag_thresh=(25, 255))

    combined = np.zeros_like(s_lap_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((gradx == 1) & (grady == 1))] = 1
    combined[(s_lap_binary == 1) & (l_lap_binary == 1)] = 1
    #combined[S_binary == 255] = 1

    h, w = img.shape[0:2]
    src = np.array([[570. /1280*w, 465./720*h],
                    [715. /1280*w, 465./720*h],
                    [1100./1280*w, 720./720*h],
                    [200. /1280*w, 720./720*h]], np.float32)

    dst = np.array([[300. /1280*w, 0./720*h],
                    [1000./1280*w, 0./720*h],
                    [1000./1280*w, 720./720*h],
                    [300. /1280*w, 720./720*h]], np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(roi(combined), M, (w,h))
    warped_img = cv2.warpPerspective(img, M, (w,h))
    warped[warped > 0 ] = 1
    warped = warped.astype('uint8')

    HSV = cv2.cvtColor(warped_img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    yellow_mask, yellow_img = color_mask(HSV,warped_img, np.array([0,100,100]), np.array([60,255,255]))
    white_mask, white_img = color_mask(HSV,warped_img, np.array([20, 0, 180]), np.array([255,80,255]))

    warped[(yellow_mask > 0) | (white_mask > 0)] = 1

    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((warped,warped,warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    nwindows = 9
    window_height = np.int(warped.shape[0]/nwindows)

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 75

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):

        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    y = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right_fitx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    draw_margin = 20
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-draw_margin, y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+draw_margin, y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-draw_margin, y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+draw_margin, y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    between_line_pts = np.hstack((left_line_window2, right_line_window1))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255, 255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,255, 255))
    cv2.fillPoly(window_img, np.int_([between_line_pts]), (0,100, 255))
    rewarped = cv2.warpPerspective(window_img, Minv, (w,h))
    result = cv2.addWeighted(img, 1, rewarped, 0.75, 0)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(white_img)
    ax2.set_title('Warped Image.', fontsize=50)
    ax3.imshow(warped, cmap='gray')
    ax3.set_title('Thresholded Image.', fontsize=50)
    ax4.imshow(result)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
