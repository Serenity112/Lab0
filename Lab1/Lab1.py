import matplotlib.pyplot as plt
import DataGenerator as dataGen


def task1():
    mu0 = [0, 2, 3]
    mu1 = [3, 5, 1]
    mu = [mu0, mu1]
    sigma0 = [2, 1, 2]
    sigma1 = [1, 2, 1]
    sigma = [sigma0, sigma1]
    col = len(mu0)
    n = 1000

    class0, class1 = dataGen.norm_dataset(mu, sigma, n)

    for i in range(0, col):
        _ = plt.hist(class0[:, i], bins='auto', alpha=0.7)
        _ = plt.hist(class1[:, i], bins='auto', alpha=0.7)
        plt.savefig('output/' + str(i + 1) + '.png')

    plt.clf()

    _ = plt.scatter(class0[:, 0], class0[:, 2], marker=".", alpha=0.7)
    _ = plt.scatter(class1[:, 0], class1[:, 2], marker=".", alpha=0.7)
    plt.savefig('output/scatter.png')
    plt.show()

def task2():
    ellipse_params = {'center_x': 50, 'center_y': 50, 'a': 30, 'b': 40}
    rectangle_params = {'left': 0, 'right': 100, 'bottom': 0, 'top': 50, 'circle_center_x': 50, 'circle_center_y': 50, 'circle_radius': 8}
    N = 500
    
    class_0_points, class_1_points = dataGen.generate_points(N, ellipse_params, rectangle_params)
    _ = plt.scatter(class_0_points[:, 0], class_0_points[:, 1], marker=".", alpha=0.7)
    _ = plt.scatter(class_1_points[:, 0], class_1_points[:, 1], marker=".", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    task2()
