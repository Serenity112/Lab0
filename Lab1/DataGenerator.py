import numpy as np

def norm_dataset(mu, sigma, n):
    mu0 = mu[0]
    mu1 = mu[1]
    sigma0 = sigma[0]
    sigma1 = sigma[1]
    col = len(mu0)

    class0 = np.random.normal(mu0[0], sigma0[0], [n, 1])
    class1 = np.random.normal(mu1[0], sigma1[0], [n, 1])

    for i in range(1, col):
        v0 = np.random.normal(mu0[i], sigma0[i], [n, 1])
        class0 = np.hstack((class0, v0))

        v1 = np.random.normal(mu1[i], sigma1[i], [n, 1])
        class1 = np.hstack((class1, v1))

    y0 = np.zeros((n, 1), dtype=bool)
    y1 = np.ones((n, 1), dtype=bool) 
    
    x = np.vstack((class0, class1))
    y = np.vstack((y0, y1)).ravel()
    
    rng = np.random.default_rng()
    arr = np.arange(2 * n) 
    rng.shuffle(arr)
    
    x = x[arr]
    y = y[arr]

    return x, y, class0, class1


def generate_points(N, ellipse_params, rectangle_params):
    class0 = np.zeros((N, 2))
    class1 = np.zeros((N, 2))
    counter0 = 0
    counter1 = 0

    while counter0 < N or counter1 < N:
        x = np.random.normal(loc=ellipse_params['center_x'], scale=ellipse_params['a']/3)
        y = np.random.normal(loc=ellipse_params['center_y'], scale=ellipse_params['b']/3)

        # ßéöî
        if ((x - ellipse_params['center_x']) / ellipse_params['a'])**2 + ((y - ellipse_params['center_y']) / ellipse_params['b'])**2 <= 1:
            if counter0 < N:
                class0[counter0, 0] = x
                class0[counter0, 1] = y
                counter0 += 1
            continue
        
        x = np.random.normal(loc=(rectangle_params['left'] + rectangle_params['right'])/2, scale=(rectangle_params['right'] - rectangle_params['left'])/6)
        y = np.random.normal(loc=rectangle_params['bottom'], scale=(rectangle_params['top'] - rectangle_params['bottom'])/3)

        # Ðþìêà
        if (
            rectangle_params['left'] < x < rectangle_params['right'] and
            rectangle_params['bottom'] < y < rectangle_params['top'] and
            ((x - rectangle_params['circle_center_x'])**2 + (y - rectangle_params['circle_center_y'])**2) > rectangle_params['circle_radius']**2
        ):
            if counter1 < N:
               class1[counter1, 0] = x
               class1[counter1, 1] = y
               counter1 += 1

    y0 = np.zeros((N, 1), dtype=bool)
    y1 = np.ones((N, 1), dtype=bool) 
    
    x = np.vstack((class0, class1))
    y = np.vstack((y0, y1)).ravel()
    
    rng = np.random.default_rng()
    arr = np.arange(2 * N) 
    rng.shuffle(arr)
    
    x = x[arr]
    y = y[arr]

    return x, y, class0, class1
