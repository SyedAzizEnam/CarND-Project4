import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.undetected = True
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

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

def perspective_change(img,inv=False):

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

    if inv:
        warped_img = cv2.warpPerspective(img, Minv, (w,h))
    else:
        warped_img = cv2.warpPerspective(img, M, (w,h))

    return warped_img

def find_lines(img):

    nwindows = 9
    window_height = np.int(img.shape[0]/nwindows)

    margin = 100
    minpix = 75

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_lane.undetected or right_lane.undetected:
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        left_lane.line_base_pos = np.argmax(histogram[:midpoint])
        right_lane.line_base_pos = np.argmax(histogram[midpoint:])+midpoint

    leftx_current = left_lane.line_base_pos
    rightx_current = right_lane.line_base_pos

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):

        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

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

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if left_lane.current_fit != None and right_lane.current_fit != None:
        res_left = np.abs(left_lane.current_fit[0] - left_fit[0]).sum()
        res_right = np.abs(right_lane.current_fit[0] - right_fit[0]).sum()

        if res_left > 0.0005 or res_right > 0.0005 or len(leftx) < 10 or len(rightx) < 10:
            left_lane.undetected, right_lane.undetected = True, True
        else:
            left_lane.undetected, right_lane.undetected = False, False
            left_lane.current_fit, right_lane.current_fit = left_fit, right_fit
    else:
        left_lane.undetected, right_lane.undetected = False, False
        left_lane.current_fit, right_lane.current_fit = left_fit, right_fit

    left_fit = left_lane.current_fit
    right_fit = right_lane.current_fit
    y = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right_fitx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

    draw_margin = 10
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-draw_margin, y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+draw_margin, y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-draw_margin, y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+draw_margin, y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    between_line_pts = np.hstack((left_line_window2, right_line_window1))

    return left_line_pts, right_line_pts, between_line_pts

def process_image(img):

    img = cv2.GaussianBlur(img, (7, 7), 0)

    HLS = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HLS)
    L = HLS[:,:,1]
    S = HLS[:,:,2]
    k_size = 3
    s_lap_binary = lapacian_threshold(S, sobel_kernel=k_size, mag_thresh=(25, 255))
    l_lap_binary = lapacian_threshold(L, sobel_kernel=k_size, mag_thresh=(25, 255))

    HSV = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    yellow_mask, yellow_img = color_mask(HSV,img, np.array([0,100,100]), np.array([60,255,255]))
    #white_mask, white_img = color_mask(HSV,img, np.array([20, 0, 180]), np.array([255,80,255]))

    combined = np.zeros_like(s_lap_binary)
    combined[(s_lap_binary == 1) & (l_lap_binary == 1)] = 1
    #combined[(yellow_mask > 0) | (white_mask > 0)] = 1
    combined[(yellow_mask > 0)] = 1

    birds_eye = perspective_change(roi(combined), inv=False)
    birds_eye[birds_eye > 0 ] = 1
    birds_eye = birds_eye.astype('uint8')

    out_img = np.dstack((birds_eye,birds_eye,birds_eye))*255
    window_img = np.zeros_like(out_img)

    left_line_pts, right_line_pts, between_line_pts = find_lines(birds_eye)
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255,255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,255,255))
    cv2.fillPoly(window_img, np.int_([between_line_pts]), (0,100, 255))
    unbirds_eye = perspective_change(window_img, inv=True)
    result = cv2.addWeighted(img, 1, unbirds_eye, 0.75, 0)

    return result

if __name__ == '__main__':

    left_lane = Line()
    right_lane = Line()

    output = 'out1.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_clip.write_videofile(output, audio=False)
