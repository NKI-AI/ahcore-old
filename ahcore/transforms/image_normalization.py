# encoding: utf-8
"""
Image normalization functions
# TODO: Support `return_stains = True` for MacenkoNormalization()
# TODO: Use torch.linalg.lstsq to solve the linear system of equations in MacenkoNormalization().
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from kornia.constants import DataKey


def _transpose_channels(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.transpose(tensor, 1, 3)
    tensor = torch.transpose(tensor, 2, 3)
    return tensor


def covariance_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    e_x = tensor.mean(dim=1)
    tensor = tensor - e_x[:, None]
    return torch.mm(tensor, tensor.T) / (tensor.size(1) - 1)


def percentile(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """
    Original author: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    Parameters
    ----------
    tensor: torch.Tensor
        input tensor for which the percentile must be calculated.
    value: float
        The percentile value
    Returns
    -------
    ``value``-th percentile of the input tensor's data.
    Notes
    -----
     Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if value is a np.float32.
    k = 1 + round(0.01 * float(value) * (tensor.numel() - 1))
    return tensor.view(-1).kthvalue(k).values


class MacenkoNormalizer(nn.Module):
    """
    A torch implementation of the Macenko Normalization technique to learn optimal staining matrix during training.
    This implementation is derived from https://github.com/EIDOSLAB/torchstain
    The reference values from the orginal implementation are:
    >>> HE_REFERENCE = torch.tensor([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
    >>> MAX_CON_REFERENCE = torch.tensor([1.9705, 1.0308])
    """

    HE_REFERENCE = torch.Tensor([[0.5042, 0.1788], [0.7723, 0.8635], [0.3865, 0.4716]])
    MAX_CON_REFERENCE = torch.Tensor([1.3484, 1.0886])

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        transmitted_intensity: int = 240,
        return_stains: bool = False,
        probability: float = 1.0,
        learnable: bool = True,
    ):
        """
        Normalize staining appearence of hematoxylin & eosin stained images. Based on [1].
        Parameters
        ----------
        alpha : float
            Percentile
        beta : float
            Transparency threshold
        transmitted_intensity : int
            Transmitted light intensity
        return_stains : bool
            If true, the output will also include the H&E channels
        probability : bool
            Probability of applying the transform
        learnable : bool
            If true, the normalization matrix will be learned during training.
        References
        ----------
        [1] A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009
        """

        super().__init__()
        self._alpha = alpha
        self._beta = beta
        self._transmitted_intensity = transmitted_intensity
        self._return_stains = return_stains
        if self._return_stains:
            raise NotImplementedError("Return stains is not implemented yet.")
        self._probability = probability
        self._learnable = learnable
        self._he_reference = nn.Parameter(self.HE_REFERENCE, requires_grad=self._learnable)
        self._max_con_reference = nn.Parameter(self.MAX_CON_REFERENCE, requires_grad=self._learnable)

    def convert_rgb_to_optical_density(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        # calculate optical density
        optical_density = -torch.log(
            (image_tensor.reshape((image_tensor.shape[0], -1, image_tensor.shape[-1])).float() + 1) / self._transmitted_intensity
        )
        # remove transparent pixels
        optical_density_hat = [sample[~torch.any(sample < self._beta, dim=1)] for sample in optical_density]
        return optical_density, optical_density_hat

    def convert_optical_density_to_rgb(self, od_tensor: torch.Tensor) -> torch.Tensor:
        normalised_image_tensor = []
        for norm_conc in od_tensor:
            normalised_image_tensor.append(self._transmitted_intensity * torch.exp(-self._he_reference.to(norm_conc) @ norm_conc))
        normalised_image_tensor = torch.stack(normalised_image_tensor, dim=0)
        normalised_image_tensor[normalised_image_tensor > 255] = 255
        return normalised_image_tensor

    def find_he_components(self, optical_density_hat: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
        """
        This function -
        1. Computes the H&E staining vectors by projecting the OD values of the image pixels on the plane
        spanned by the eigenvectors corresponding to their two largest eigenvalues.
        2. Normalizes the staining vectors to unit length.
        3. Calculates the angle between each of the projected points and the first principal direction.
        Parameters:
        ----------
        optical_density_hat: torch.Tensor
            Optical density of the image
        eigvecs: torch.Tensor
            Eigenvectors of the covariance matrix
        Returns:
        -------
        he_components: torch.Tensor
            The H&E staining vectors
        """
        t_hat = torch.matmul(optical_density_hat, eigvecs)
        phi = torch.atan2(t_hat[:, 1], t_hat[:, 0])

        min_phi = percentile(phi, self._alpha)
        max_phi = percentile(phi, 100 - self._alpha)

        v_min = torch.matmul(eigvecs, torch.stack((torch.cos(min_phi), torch.sin(min_phi)))).unsqueeze(1)
        v_max = torch.matmul(eigvecs, torch.stack((torch.cos(max_phi), torch.sin(max_phi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the one corresponding to eosin second
        he_vector = torch.where(
            v_min[0] > v_max[0], torch.cat((v_min, v_max), dim=1), torch.cat((v_max, v_min), dim=1)
        )

        return he_vector

    def __compute_matrices(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the H&E staining vectors and their concentration values for every pixel in the image tensor.
        """
        batch_he_vecs = []
        batch_max_con = []
        batch_con_vecs = []
        # Convert RGB values in the image to optical density values following the Beer-Lambert's law.
        optical_density, optical_density_hat = self.convert_rgb_to_optical_density(image_tensor)
        # For every sample in the batch, calculate the eigenvectors of optical density matrix thresholded to remove transparent pixels.
        for i in range(len(optical_density_hat)):
            _, eigvecs = torch.linalg.eigh(covariance_matrix(optical_density_hat[i].T), UPLO="U")
            # choose the first two eigenvectors corresponding to the two largest eigenvalues.
            eigvecs = eigvecs[:, [1, 2]]
            # Find the H&E staining vectors and their concentration values for every pixel in the image tensor.
            # Note - The dependence of staining and their concentrations are linear in OD space
            he = self.find_he_components(optical_density_hat[i], eigvecs)
            # Calculate the concentrations of the H&E stains in each pixel.
            # We do this by solving a linear system of equations. (In this case, the system is overdetermined).
            # OD =   HE * C -> (1)
            # where:
            #   1. OD is the optical density of the pixels in the batch. The dimension is: (n x 3)
            #   2. HE is the H&E staining vectors (3 x 2). The dimension is: (3 x 2)
            #   3. C is the concentration of the H&E stains in each pixel. The dimension is: (2 x n)
            # The solution to this system of equation is unique and is computed in the following way:
            concentration = he.pinverse() @ optical_density[i].T
            max_concentration = torch.stack([percentile(concentration[0, :], 99), percentile(concentration[1, :], 99)])
            batch_he_vecs.append(he)
            batch_max_con.append(max_concentration)
            batch_con_vecs.append(concentration)
        return torch.stack(batch_he_vecs, dim=0), torch.stack(batch_con_vecs, dim=0), torch.stack(batch_max_con, dim=0)

    def fit(self, image_tensor: torch.Tensor) -> None:
        he_matrix, _, maximum_concentration = self.__compute_matrices(image_tensor=image_tensor)
        self._he_reference = nn.Parameter(he_matrix, requires_grad=self._learnable)
        self._max_con_reference = nn.Parameter(maximum_concentration, requires_grad=self._learnable)

    def __normalize_concentrations(
        self, concentrations: torch.Tensor, maximum_concentration: torch.Tensor
    ) -> torch.Tensor:
        output = []
        for conc, max_conc in zip(concentrations, maximum_concentration):
            norm_conc = conc * (self._max_con_reference.to(max_conc) / max_conc).unsqueeze(-1)
            output.append(norm_conc)
        return torch.stack(output, dim=0)

    def __create_normalized_images(
        self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor
    ) -> torch.Tensor:
        batch, classes, height, width = image_tensor.shape
        # recreate the image using reference mixing matrix
        normalised_image_tensor = self.convert_optical_density_to_rgb(od_tensor=normalized_concentrations)
        normalised_image_tensor = normalised_image_tensor.mT.reshape(batch, height, width, classes)
        normalised_image_tensor = _transpose_channels(normalised_image_tensor)
        return normalised_image_tensor

    # def __get_h_stain(self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
    #     batch, classes, height, width = image_tensor.shape
    #     hematoxylin_tensors = torch.mul(
    #         self._transmitted_intensity,
    #         torch.exp(
    #             torch.matmul(-self._he_reference[:, 0].unsqueeze(-1), normalized_concentrations[0, :].unsqueeze(0))
    #         ),
    #     )
    #     hematoxylin_tensors[hematoxylin_tensors > 255] = 255
    #     hematoxylin_tensors = hematoxylin_tensors.T.reshape(batch, height, width, classes)
    #     hematoxylin_tensors = _transpose_channels(hematoxylin_tensors)
    #     return hematoxylin_tensors
    #
    # def __get_e_stain(self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
    #     batch, classes, height, width = image_tensor.shape
    #     eosin_tensors = torch.mul(
    #         self._transmitted_intensity,
    #         torch.exp(
    #             torch.matmul(-self._he_reference[:, 1].unsqueeze(-1), normalized_concentrations[1, :].unsqueeze(0))
    #         ),
    #     )
    #     eosin_tensors[eosin_tensors > 255] = 255
    #     eosin_tensors = eosin_tensors.T.reshape(batch, height, width, classes)
    #     eosin_tensors = _transpose_channels(eosin_tensors)
    #     return eosin_tensors

    def forward(self, *args: tuple[torch.Tensor], data_keys: Optional[list]) -> list[torch.Tensor]:
        args = list(args)
        stains = {}
        image_tensor: torch.Tensor = args[0]

        # # TODO: Do random sampling from batch before augmentation
        # if np.random.rand() > self._probability:
        #     args[0] = image_tensor
        #     return args

        he_matrix, concentrations, maximum_concentration = self.__compute_matrices(image_tensor)

        normalized_concentrations = self.__normalize_concentrations(concentrations, maximum_concentration)
        normalised_image = self.__create_normalized_images(normalized_concentrations, image_tensor)
        args[0] = normalised_image
        # if self._return_stains:
        #     stains["image_hematoxylin"] = self.__get_h_stain(normalized_concentrations, image_tensor)
        #     stains["image_eosin"] = self.__get_e_stain(normalized_concentrations, image_tensor)

        return args

    def __repr__(self):
        return (
            f"{type(self).__name__}(alpha={self._alpha}, beta={self._beta}, "
            f"transmitted_intensity={self._transmitted_intensity}, probability={self._probability})"
        )
