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



    return  class0, class1


def generate_points(N, ellipse_params, rectangle_params):
    class_0_points = []
    class_1_points = []

    while len(class_0_points) < N or len(class_1_points) < N:
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)

        # ßéöî
        if ((x - ellipse_params['center_x']) / ellipse_params['a'])**2 + ((y - ellipse_params['center_y']) / ellipse_params['b'])**2 <= 1:
            class_0_points.append([x, y])
            continue

        # Ðþìêà
        if (
            rectangle_params['left'] < x < rectangle_params['right'] and
            rectangle_params['bottom'] < y < rectangle_params['top'] and
            ((x - rectangle_params['circle_center_x'])**2 + (y - rectangle_params['circle_center_y'])**2) > rectangle_params['circle_radius']**2
        ):
            class_1_points.append([x, y])

    return np.array(class_0_points), np.array(class_1_points)
