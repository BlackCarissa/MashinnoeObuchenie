import numpy as np
# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________

def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # YOUR CODE HERE
    matches = []
    d1_sq = np.sum(des1**2,axis=1,keepdims=True)
    d2_sq = np.sum(des2**2,axis=1,keepdims=True).T
    L2 = np.sqrt(d1_sq+d2_sq-2*np.dot(des1,des2.T))
    for i in range(len(L2)):
        j = np.argmin(L2[i], axis=None)
        if i == np.argmin(L2.T[j],axis=None):
            r = DummyMatch(i,j,L2[i][j])
            matches.append(r)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches