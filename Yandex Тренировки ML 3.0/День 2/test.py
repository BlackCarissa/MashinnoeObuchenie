# do not change the code in the block below
# __________start of block__________
import numpy as np
def compute_sobel_gradients_two_loops(image):
    # Get image dimensions
    height, width = image.shape

    # Initialize output gradients
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Pad the image with zeros to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
# __________end of block__________

    # Define the Sobel kernels for X and Y gradients
    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]]).reshape(-1,9).T # YOUR CODE HERE
    sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]]).reshape(-1,9).T # YOUR CODE HERE

    # Apply Sobel filter for X and Y gradients using convolution
    for i in range(1, height + 1):
        for j in range(1, width + 1):
            gradient_x[i-1][j-1]=np.dot(padded_image[i-1:i+2, j-1:j+2].reshape(-1,9),sobel_x)[0][0]
            gradient_y[i-1][j-1]=np.dot(padded_image[i-1:i+2, j-1:j+2].reshape(-1,9), sobel_y)[0][0]
    return gradient_x, gradient_y
import numpy as np # for your convenience when you copy the code to the contest
def compute_gradient_magnitude(sobel_x, sobel_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    # YOUR CODE HERE
    gradient_magnitude = np.zeros_like(sobel_x, dtype=np.float64)
    height, width = sobel_x.shape
    for i in range(0, height):
        for j in range(0, width):
            gradient_magnitude[i][j] = np.sqrt(sobel_x[i][j]**2+sobel_y[i][j]**2)
            
    return gradient_magnitude


def compute_gradient_direction(sobel_x, sobel_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    # YOUR CODE HERE
    gradient_direction = np.zeros_like(sobel_x, dtype=np.float64)
    height, width = sobel_x.shape
    for i in range(0, height):
        for j in range(0, width):
            gradient_direction[i][j] = (np.arctan2(sobel_y[i][j],sobel_x[i][j]))*180/np.pi
    return gradient_direction
cell_size = 7
def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))
def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale
    
    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image) # YOUR CODE HERE

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y) # YOUR CODE HERE
    direction = compute_gradient_direction(gradient_x, gradient_y) # YOUR CODE HERE
    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_x = image.shape[1] // cell_width
    n_cells_y = image.shape[0] // cell_height

    histograms = np.zeros((n_cells_y, n_cells_x, bins))
    magnitude = split(magnitude,cell_width,cell_height)
    direction = split(direction,cell_width,cell_height)
    k = 0
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            histograms[i][j]=np.histogram(direction[k], bins, (-180, 180), weights=magnitude[k], density=False)[0]
            a = histograms[i][j]
            b = a.sum()
            if b != 0:
                histograms[i][j] = a/b
            k+=1
    return histograms