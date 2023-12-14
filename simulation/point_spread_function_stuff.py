class PointSpreadFunction():
    def __init__(self, sigma=None, covariance=None):
        """
        provide either sigma or covariance matrix to generate PSF
        :param sigma:
        :param covariance:
        """
        if np.any(covariance):
            # if covariance use covariance
            self.covariance = torch.tensor(covariance)
        elif sigma:
            #if sigma build covariance matrix from sigma
            try:
                self.covariance = torch.eye(2)*sigma
            except:
                Exception("Nonbroadcastable shape")
        else:
            raise Exception("requires either Covariance or sigma")


    def gaussian(self, positions, probs=None):
        """
        Return Gaussian mixture distribution from positions with relative intensity as probs
        :param positions: Emitter positions
        :param probs: relative intensity
        :return:
        """
        positions = torch.tensor(positions, dtype=torch.float32)
        #test position shape for broadcasting capability with sigma

        if positions.shape[-1] !=2:
            raise Exception("Position should have 2 coordinates")


        #test positions for shape
        if torch.linalg.matrix_rank(positions) == 1:
            pdf = td.multivariate_normal.MultivariateNormal(
                positions,
                covariance_matrix=self.covariance,
            )
        elif torch.linalg.matrix_rank(positions) == 2:
            distributions = [td.multivariate_normal.MultivariateNormal(
                loc=position,
                covariance_matrix=self.covariance,
            ) for position in positions]
            if not probs:
                probs = [1/positions.shape[0]]*positions.shape[0]
            pdf = torch.Mixture(cat=torch.Categorical(probs=probs),
                        components=distributions
                        )

        return pdf
