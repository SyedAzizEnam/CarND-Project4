import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import collections
import glob

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.undetected = True
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=10)
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
        self.allx = collections.deque(maxlen=5)
        #y values for detected line pixels
        self.ally = collections.deque(maxlen=5)

    def check_curv(self, R):
        if self.radius_of_curvature is None:
            return True

        return abs(R-R0)/R0 <= 0.5

def stack_arr(arr):
    # Stacks 1-channel array into 3-channel array to allow plotting
    return np.stack((arr, arr,arr), axis=2)

def rad_curv(xarray, yarray):
    fit = np.polyfit(yarray*ym_per_pix, xarray*xm_per_pix, 2)
    y_eval = np.max(yarray*ym_per_pix)
    curverad = (1 + (2*fit[0]*y_eval + fit[1])**2)**1.5/2/fit[0]
    return curverad

def reject_outliers(x_list,y_list):

    mu_x= np.mean(np.concatenate(x_list))
    sig_x= np.std(np.concatenate(x_list))

    for i in range(len(x_list)):
        index = abs(x_list[i] -mu_x) < 2*sig_x
        x_list[i] = x_list[i][index]
        y_list[i] = y_list[i][index]
    return x_list, y_list

def calibrate_camera():

    cal_images = glob.glob('camera_cal/calibration*.jpg')
    nx, ny = 9, 6

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    fname = cal_images[0]
    for fname in cal_images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return mtx, dist

def color_mask(hsv,img, min_thresh, max_thresh):

    mask = cv2.inRange(hsv, min_thresh, max_thresh)
    res = cv2.bitwise_and(img,img, mask= mask)
    return mask, res

def roi(img, width_percent=0, height_percent=0.65):

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
    src = np.array([[585./1280*w, 455./720*h],
                    [705./1280*w, 455./720*h],
                    [1130./1280*w, 720./720*h],
                    [190./1280*w, 720./720*h]], np.float32)

    dst = np.array([[300./1280*w, 100./720*h],
                    [1000./1280*w, 100./720*h],
                    [1000./1280*w, 720./720*h],
                    [300./1280*w, 720./720*h]], np.float32)

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
    #out_img = np.dstack((img,img,img))*255
    margin = 50
    minpix = 50

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_lane.undetected or right_lane.undetected:
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        leftx_current  = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:])+midpoint

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):

        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
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

    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Extract left and right line pixel positions
    left_lane.allx.append(nonzerox[left_lane_inds])
    left_lane.ally.append(nonzeroy[left_lane_inds])
    right_lane.allx.append(nonzerox[right_lane_inds])
    right_lane.ally.append(nonzeroy[right_lane_inds])

    left_lane.allx, left_lane.ally = reject_outliers(left_lane.allx,left_lane.ally)
    right_lane.allx, right_lane.ally = reject_outliers(right_lane.allx,right_lane.ally)

    y = np.linspace(0, img.shape[0]-1, img.shape[0] )

    left_lane.current_fit = np.polyfit(np.concatenate(left_lane.ally), np.concatenate(left_lane.allx), 2)
    right_lane.current_fit = np.polyfit(np.concatenate(right_lane.ally), np.concatenate(right_lane.allx), 2)

    y_eval = np.max(np.concatenate(left_lane.ally))
    left_lane.radius_of_curvature = rad_curv(np.concatenate(left_lane.allx), np.concatenate(left_lane.ally))
    right_lane.radius_of_curvature = rad_curv(np.concatenate(right_lane.allx), np.concatenate(right_lane.ally))
    lane_pos = (left_lane.current_fit[-1] + right_lane.current_fit[-1])//2

    left_fitx = left_lane.current_fit[0]*y**2 + left_lane.current_fit[1]*y + left_lane.current_fit[2]
    right_fitx = right_lane.current_fit[0]*y**2 + right_lane.current_fit[1]*y + right_lane.current_fit[2]

    left_lane.recent_xfitted.append(left_fitx)
    right_lane.recent_xfitted.append(right_fitx)

    left_lane.bestx = np.mean(left_lane.recent_xfitted, axis = 0)
    right_lane.bestx = np.mean(right_lane.recent_xfitted, axis = 0)

    draw_margin = 10
    left_line_window1 = np.array([np.transpose(np.vstack([left_lane.bestx-draw_margin, y]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane.bestx+draw_margin, y])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_lane.bestx-draw_margin, y]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx+draw_margin, y])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    between_line_pts = np.hstack((left_line_window2, right_line_window1))

    return left_line_pts, right_line_pts, between_line_pts,lane_pos

def process_image(img):

    img = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    HLS = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HLS)
    L = HLS[:,:,1]
    S = HLS[:,:,2]
    k_size = 3
    s_lap_binary = lapacian_threshold(S, sobel_kernel=k_size, mag_thresh=(100, 255))
    l_lap_binary = lapacian_threshold(L, sobel_kernel=k_size, mag_thresh=(100, 255))

    HSV = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    yellow_mask, yellow_img = color_mask(HSV,img, np.array([0,100,100]), np.array([60,255,255]))
    white_mask, white_img = color_mask(HSV,img, np.array([20, 0, 180]), np.array([255,80,255]))

    combined = np.zeros_like(s_lap_binary)
    combined[(s_lap_binary == 1) | (l_lap_binary == 1)] = 1
    combined[(yellow_mask > 0)|(white_mask > 0)] = 1

    birds_eye = perspective_change(combined, inv=False)
    birds_eye[birds_eye > 0 ] = 1
    birds_eye = birds_eye.astype('uint8')

    window_img = np.zeros_like(np.dstack((birds_eye,birds_eye,birds_eye)))

    left_line_pts, right_line_pts, between_line_pts,lane_pos = find_lines(birds_eye)

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255,255))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,255,255))
    cv2.fillPoly(window_img, np.int_([between_line_pts]), (0,100, 255))
    unbirds_eye = perspective_change(window_img, inv=True)
    result = cv2.addWeighted(img, 1, unbirds_eye, 0.75, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(result,'Left radius of curvature = %.2f m'%(left_lane.radius_of_curvature),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Right radius of curvature = %.2f m'%(right_lane.radius_of_curvature),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Vehicle position offset = %.2f m'%((lane_pos - (result.shape[1]//2))*xm_per_pix),(50,110), font, 1,(255,255,255),2,cv2.LINE_AA)

    if do_diagnosis == 1:
        font = cv2.FONT_HERSHEY_COMPLEX
        diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        diagScreen[0:720, 0:1280] = result
        diagScreen[0:240, 1280:1600] = cv2.resize(stack_arr(255*birds_eye), (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[0:240, 1600:1920,:] = cv2.resize(stack_arr(255*combined), (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1280:1600] = cv2.resize(yellow_img, (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[240:480, 1600:1920] = cv2.resize(window_img, (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[480:720, 1280:1600] = cv2.resize(white_img, (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[480:720, 1600:1920] = cv2.resize(stack_arr(255*s_lap_binary), (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 0:320] = cv2.resize(img, (320,240), interpolation=cv2.INTER_AREA)
        diagScreen[840:1080, 320:640] = cv2.resize(stack_arr(255*l_lap_binary), (320,240), interpolation=cv2.INTER_AREA)

        return diagScreen
    else:
        return result

if __name__ == '__main__':

    cam_mtx, cam_dist = calibrate_camera()

    do_diagnosis = 0

    left_lane = Line()
    right_lane = Line()
    """
    img = mpimg.imread('./test_images/straight_lines1.jpg')
    plt.imshow(process_image(img))
    plt.show()

    """
    output = 'out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_clip.write_videofile(output, audio=False)
