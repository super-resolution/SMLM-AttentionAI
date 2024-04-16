import numpy as np

class ApproachingLines():
    def points(self, width:int, n:int, size:int)->np.ndarray:
        """
        Returns points on a negative exponential distribution
        :param width: line width
        :param n: number of points
        :param size: size in x direction
        :return: array of x and y positions
        """
        width_dist = (np.random.rand(n))*width
        x = np.random.rand(n)*size
        #decrease x to achieve slower convergence
        y = 30*np.exp(-x/8)+width_dist
        return np.concatenate([x[:,None],y[:,None]],axis=1)
        #todo: exponential
    def approaching_lines(self):
        #todo create positive and negative points appraoching each other
        line1 = self.points(1,100000,60)
        line2 = self.points(1,100000,60)
        line1[:,1] += 30
        line2[:,1] = 30-line2[:,1]
        return np.concatenate([line1, line2],axis=0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    struct = ApproachingLines()
    data = struct.approaching_lines()
    plt.scatter(data[:,0],data[:,1])
    plt.show()

