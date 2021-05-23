import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import matplotlib


#checkboards with 9x6 inside corners
nx = 9
ny = 6

#lists to store object points and image points
objpoints = []
imgpoints = []

images = glob.glob('camera_cal/calibration*.jpg')

for idx, fname in enumerate(images):
    img = mpimg.imread(fname)
    

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    if ret == True:
        
        imgpoints.append(corners)
        objpoints.append(objp) #this obj points will be the same for all calibration images, represent the real checkboard
        #draw coners found in the original images as save to disks with name corners_foundx.jpg
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        write_corners = 'camera_cal/corners_found' + str(idx) + '.jpg'
        cv2.imwrite(write_corners, img)

#dump pickle to save imgpoints and objpoints for later use      
points_dict = {'imgpoints': imgpoints, 'objpoints': objpoints}
points_pickle = open('camera_cal/points_pickle.p', 'wb')
pickle.dump(points_dict, points_pickle)
points_pickle.close()

#load pickle
dist_pickle = pickle.load(open("camera_cal/points_pickle.p", "rb"))
objpoints = dist_pickle['objpoints']
imgpoints = dist_pickle['imgpoints']

#undistort function for convenient use
def undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

#undistort images, using ojbpoints and imgpoints from previous steps
images = glob.glob('test_images/test*.jpg')     
for idx, fname in enumerate(images):   
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])
    #get mtx, dist for later use
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist = undistort(img, objpoints, imgpoints)
    write_undistorted = 'test_images/undistorted' + str(idx+1) + '.jpg'
    cv2.imwrite(write_undistorted, undist) 

#dump pickle to save mtx, dist for later use
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump(dist_pickle, open('camera_cal/calibration_pickle.p', 'wb'))

#print an example of original vs undistorted image (checkboard)
img = cv2.imread('camera_cal/calibration5.jpg')
undistorted = undistort(img, objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

def undistort_color(img, objpoints, imgpoints):
	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist

#print an example of original vs undistorted image (roadimage)
img = cv2.imread('test_images/test1.jpg')
undistorted = undistort_color(img, objpoints, imgpoints)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

####################This part for image transform methods######################

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = abs(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1
    return binary_output

def color_threshold(img, s_thresh=(0,255), v_thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    binary_output = np.zeros_like(v_binary)
    binary_output[(s_binary == 1) & (v_binary == 1)] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobelxy = np.sqrt(sobel_x**2 + sobel_y**2)
    scaled_mag_sobelxy = np.uint8(255*mag_sobelxy/np.max(mag_sobelxy))
    binary_output = np.zeros_like(scaled_mag_sobelxy)
    binary_output[(scaled_mag_sobelxy >= mag_thresh[0]) & (scaled_mag_sobelxy <= mag_thresh[1])] = 1
    return binary_output

def dir_thresh(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_dir = np.arctan2(abs(sobel_y), abs(sobel_x))
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= dir_thresh[0]) & (grad_dir <= dir_thresh[1])] = 1
    return binary_output

def color_and_gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = abs(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))*255
    return color_binary

 ###########################Tracker Class#############################

class tracker():

	def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym=1, My_xm=1, Mysmooth_factor=15):
		self.recent_centers = []
		self.window_width = Mywindow_width
		self.window_height = Mywindow_height
		self.margin = Mymargin
		self.ym_per_pix = My_ym
		self.xm_per_pix = My_xm
		self.smooth_factor = Mysmooth_factor

	def find_window_centroids(self, warped):
		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin
		window_centroids = []
		window = np.ones(window_width)

		l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2

		r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1]/2)

		window_centroids.append((l_center, r_center))

		for level in range(1, (int)(warped.shape[0]/window_height)):
			image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
			conv_signal = np.convolve(window, image_layer)

			offset = window_width/2
			l_min_index = int(max(l_center+offset-margin, 0))
			l_max_index = int(min(l_center+offset+margin, warped.shape[1]))
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

			r_min_index = int(max(r_center+offset-margin,0))
			r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

			window_centroids.append((l_center, r_center))
		self.recent_centers.append(window_centroids)

		return np.average(self.recent_centers[-self.smooth_factor:], axis=0)



# Polynomial fit values from the previous frame

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = np.nonzero((nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)&(nonzeroy>=win_y_low)&(nonzeroy<win_y_high))[0]
        good_right_inds = np.nonzero((nonzerox>=win_xright_low)&(nonzerox<win_xright_high)&(nonzeroy>=win_y_low)&(nonzeroy<win_y_high))[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) >  minpix :
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        #pass # Remove this when you add your function

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    

    return left_fitx, right_fitx, ploty

def search_around_poly(binary_warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    #nonzero = binary_warped.nonzero()
    nonzero = np.nonzero(binary_warped)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
  
    left_lane_inds = (((left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)>nonzerox)
                    & ((left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)<nonzerox))
                    
    right_lane_inds = (((right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)>nonzerox)
                    & ((right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)<nonzerox))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result

 #################REQUIREMENT 4 TEST OUT PUT#####################   
img_org = mpimg.imread('test_images/test5.jpg')
hls = cv2.cvtColor(img_org, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]

# Grayscale image
gray = cv2.cvtColor(img_org, cv2.COLOR_RGB2GRAY)

# Sobel x
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

# Threshold color channel
s_thresh_min = 170
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

img = combined_binary.copy()

w = img.shape[1]
h = img.shape[0]
img_size = (w, h)
bot_w = .76
mid_w = .1
h_pct = .62
bottom_trim = .935

offset = w*0.25

#src = np.float32([[w*(0.5-mid_w/2.3), h*h_pct], [w*(0.5+mid_w/1.5), h*h_pct], [w*(0.5+bot_w/1.8), h*bottom_trim], [w*(0.5-bot_w/2.2), h*bottom_trim]])
src = np.float32([[w*(0.5-mid_w/2), h*h_pct], [w*(0.5+mid_w/2), h*h_pct], [w*(0.5+bot_w/2), h*bottom_trim], [w*(0.5-bot_w/2), h*bottom_trim]])
dst = np.float32([[offset, 0], [w-offset, 0], [w-offset, h], [offset, h]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)

left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

result = search_around_poly(warped)

plt.imshow(result)
plt.show() 

#lood pickle

dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

#Final: all together into the final process_image method for lane line detection
def window_mask(width, height, img, center, level):
    output = np.zeros_like(img)
    output[int(img.shape[0]-(level+1)*height):int(img.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img.shape[1])] = 1
    return output

def process_image(img_in):
    img_org = cv2.undistort(img_in, mtx, dist, None, mtx)
    gradx = abs_sobel_thresh(img_org, orient='x', thresh_min=25, thresh_max=255)
    grady = abs_sobel_thresh(img_org, orient='y', thresh_min=25, thresh_max=255)
    mag_binary = mag_thresh(img_org, sobel_kernel=9, mag_thresh=(30, 100))
    c_binary = color_threshold(img_org, s_thresh=(100,255), v_thresh=(50,255))
    dir_binary = dir_thresh(img_org, sobel_kernel=9, dir_thresh=(0.7, 1.3))
    
    preprocessImage = np.zeros_like(grady)
    
    preprocessImage[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (c_binary == 1)] = 1

    img = preprocessImage.copy()
    w = img.shape[1]
    h = img.shape[0]
    img_size = (w, h)
    bot_w = .76
    mid_w = .1    
    h_pct = .62
    bottom_trim = .935

    offset = w*0.25

    src = np.float32([[w*(0.5-mid_w/2), h*h_pct], [w*(0.5+mid_w/2), h*h_pct], [w*(0.5+bot_w/2), h*bottom_trim], [w*(0.5-bot_w/2), h*bottom_trim]])
    dst = np.float32([[offset, 0], [w-offset, 0], [w-offset, h], [offset, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    
    window_width = 25
    window_height = 80

    curve_centers = tracker(window_width, window_height, Mymargin=25, My_ym=10/720, My_xm=4/384, Mysmooth_factor=15)
    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    rightx = []
    leftx = []

    for level in range(0, len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

        l_points[(l_points==255)|(l_mask==1)] = 255
        r_points[(r_points==255)|(r_mask==1)] = 255

    template = np.array(r_points+l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    result = cv2.addWeighted(warpage, 1, template, 1.0, 0.0)

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals, yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals, yvals[::-1]),axis=0))),np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals, yvals[::-1]),axis=0))),np.int32)

    road = np.zeros_like(img_org)
    road_bkg = np.zeros_like(img_org)
    cv2.fillPoly(road, [left_lane], color=[255,0,0])
    cv2.fillPoly(road, [inner_lane], color=[0,255,0])
    cv2.fillPoly(road, [right_lane], color=[0,0,255])
    
    cv2.fillPoly(road_bkg, [left_lane], color=[255,255,255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255,255,255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img_org, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix = curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
    curverad = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix+curve_fit_cr[1])**2)**1.5)/abs(2*curve_fit_cr[0])

    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, 'Radius of Curvature = '+str(round(curverad,3))+'(m)',(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'(m) '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return result

from moviepy.editor import VideoFileClip
Output_video_challenge = 'project2_output_05232021.mp4'
Input_video = 'project_video.mp4'
Input_video_challenge = 'challenge_video.mp4'

clip2 = VideoFileClip(Input_video)
video_clip = clip2.fl_image(process_image)
video_clip.write_videofile(Output_video_challenge, audio=False)