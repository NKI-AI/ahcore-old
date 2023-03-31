# encoding: utf-8
"""
Histopathology stain specific image normalization functions
# TODO: Support `return_stains = True` for MacenkoNormalization()
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


def _transpose_channels(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.transpose(tensor, 1, 3)
    tensor = torch.transpose(tensor, 2, 3)
    return tensor


def _compute_concentrations(
    he_vector: torch.Tensor, optical_density: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the concentrations of the individual stains.
    Parameters
    ----------
    he_vector : torch.Tensor
        The H&E staining vectors
    optical_density: torch.Tensor
        Optical density of the image

    Returns
    -------
    he_concentrations: torch.Tensor
        Concentrations of the individual stains
    max_concentrations: torch.Tensor
        Maximum concentrations of the individual stains
    """
    he_concentrations = he_vector.pinverse() @ optical_density.T
    max_concentration = torch.stack([percentile(he_concentrations[0, :], 99), percentile(he_concentrations[1, :], 99)])
    return he_concentrations, max_concentration


def covariance_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """
    https://en.wikipedia.org/wiki/Covariance_matrix
    """
    e_x = tensor.mean(dim=1)
    tensor = tensor - e_x[:, None]
    return torch.mm(tensor, tensor.T) / (tensor.size(1) - 1)


def _compute_eigenvecs(optical_density_hat: torch.Tensor) -> torch.Tensor:
    """
    This function computes the eigenvectors of the covariance matrix of the optical density values.
    Parameters:
    ----------
    optical_density_hat: list[torch.Tensor]
        Optical density of the image

    Returns:
    -------
    eigvecs: torch.Tensor
        Eigenvectors of the covariance matrix
    """
    _, eigvecs = torch.linalg.eigh(covariance_matrix(optical_density_hat.T), UPLO="U")
    # choose the first two eigenvectors corresponding to the two largest eigenvalues.
    eigvecs = eigvecs[:, [1, 2]]
    return eigvecs


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
        self._he_reference = self.HE_REFERENCE
        self._max_con_reference = self.MAX_CON_REFERENCE

    def convert_rgb_to_optical_density(self, image_tensor: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        This function converts an RGB image to optical density values following the Beer-Lambert's law.
        Parameters
        ----------
        image_tensor: torch.Tensor
            RGB image tensor, shape (B, 3, H, W)

        Returns
        -------
        optical_density: torch.Tensor
            Optical density of the image tensor, shape (B, H*W, 3)

        optical_density_hat: list[torch.Tensor]
            Optical density of the image tensor, shape (B, num_foreground_pixels, 3)
        """
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        num_tiles, height, width, channels = image_tensor.shape
        # calculate optical density
        optical_density = -torch.log(
            (image_tensor.reshape((num_tiles, -1, channels)).float() + 1) / self._transmitted_intensity
        )
        # remove transparent pixels
        optical_density_hat = [sample[~torch.any(sample < self._beta, dim=1)] for sample in optical_density]
        # Remove lone pixels (i.e. if non-transparent pixels in a tile do not exceed one-fifth of the tile width)
        # This should be done to avoid ill-conditioned covariance matrices in downstream processing.
        optical_density_hat = [
            tile_pixels for tile_pixels in optical_density_hat if tile_pixels.shape[0] > (height / 5)
        ]
        return optical_density, optical_density_hat

    def convert_optical_density_to_rgb(self, od_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converts optical density values to RGB

        Parameters
        ----------
        od_tensor: torch.Tensor
            Optical density of the image.

        Returns
        -------
        rgb_tensor: torch.Tensor
            RGB image.
        """
        normalised_image_tensor = []
        for norm_conc in od_tensor:
            normalised_image_tensor.append(
                self._transmitted_intensity * torch.exp(-self._he_reference.to(norm_conc) @ norm_conc)
            )
        normalised_image_tensor = torch.stack(normalised_image_tensor, dim=0)
        normalised_image_tensor[normalised_image_tensor > 255] = 255
        return normalised_image_tensor

    def _find_he_components(self, optical_density_hat: torch.Tensor, eigvecs: torch.Tensor) -> torch.Tensor:
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

    def __compute_matrices(
        self, image_tensor: torch.Tensor, staining_parameters: Optional[dict[str : torch.Tensor] | None] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the H&E staining vectors and their concentration values for every pixel in the image tensor.
        """
        batch_he_vecs = []
        batch_max_con = []
        batch_con_vecs = []
        wsi_eigenvectors = staining_parameters["wsi_eigenvectors"] if staining_parameters is not None else None
        wsi_staining_vectors = staining_parameters["wsi_staining_vectors"] if staining_parameters is not None else None
        # Convert RGB values in the image to optical density values following the Beer-Lambert's law.
        # Note - The dependence of staining and their concentrations are linear in OD space.
        optical_density, optical_density_hat = self.convert_rgb_to_optical_density(image_tensor)
        # For tile in the optical density (OD) vector:
        #  Step 1. calculate the eigenvectors of thresholded OD vector (optical_density_hat).
        #  Step 2. find the H&E staining vectors by projecting the OD vector on the plane spanned by the eigenvectors.
        #  Step 3. calculate the concentration of the H&E staining vectors for each pixel in the OD vector.
        #  Step 4. Also return the maximum concentration of the H&E staining within the tile.
        for i in range(len(optical_density_hat)):
            # Performing Step 1:
            eigvecs = _compute_eigenvecs(optical_density_hat[i]) if wsi_eigenvectors is None else wsi_eigenvectors[i]
            # Performing Step 2:
            he = (
                self._find_he_components(optical_density_hat[i], eigvecs)
                if wsi_staining_vectors is None
                else wsi_staining_vectors[i]
            )
            # Performing Step 3, 4:
            #   Calculate the concentrations of the H&E stains in each pixel.
            #   We do this by solving a linear system of equations. (In this case, the system is overdetermined).
            #   OD =   HE * C -> (1)
            #   where:
            #       1. OD is the optical density of the pixels in the batch. The dimension is: (n x 3)
            #       2. HE is the H&E staining vectors (3 x 2). The dimension is: (3 x 2)
            #       3. C is the concentration of the H&E stains in each pixel. The dimension is: (2 x n)
            he_concentrations, max_concentrations = _compute_concentrations(he, optical_density[i])
            batch_he_vecs.append(he)
            batch_max_con.append(max_concentrations)
            batch_con_vecs.append(he_concentrations)
        return torch.stack(batch_he_vecs, dim=0), torch.stack(batch_con_vecs, dim=0), torch.stack(batch_max_con, dim=0)

    def fit(self, wsi: torch.Tensor) -> dict[str : torch.Tensor]:
        """
        Compress a WSI to a single matrix of eigenvectors and return staining parameters.

        Parameters:
        ----------
        wsi: torch.tensor
            A tensor containing a whole slide image of shape (1, channels, height, width)

        Returns:
        -------
        staining_parameters: dict[str: torch.Tensor, str: torch.Tensor]
            The eigenvectors of the optical density values of the pixels in the image.
        """
        optical_density, optical_density_hat = self.convert_rgb_to_optical_density(wsi)
        wsi_eigenvectors = _compute_eigenvecs(optical_density_hat[0])
        wsi_level_he = self._find_he_components(optical_density_hat[0], wsi_eigenvectors)
        _, wsi_level_max_concentrations = _compute_concentrations(wsi_level_he, optical_density[0])
        staining_parameters = {
            "wsi_staining_vectors": wsi_level_he,
            "wsi_eigenvectors": wsi_eigenvectors,
            "max_wsi_concentration": wsi_level_max_concentrations,
        }
        return staining_parameters

    def set(self, target_image: torch.Tensor) -> None:
        """
        Set the reference image for the stain normaliser.
        Parameters:
        ----------
        target_image: torch.Tensor
            The reference image for the stain normaliser.
        """
        he_matrix, _, maximum_concentration = self.__compute_matrices(image_tensor=target_image)
        self._he_reference = he_matrix
        self._max_con_reference = maximum_concentration

    def __normalize_concentrations(
        self, concentrations: torch.Tensor, maximum_concentration: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize the concentrations of the H&E stains in each pixel against the reference concentration.

        Parameters
        ----------
        concentrations: torch.Tensor
            The concentration of the H&E stains in each pixel.
        maximum_concentration
            The maximum concentration of the H&E stains in each pixel.
        Returns
        -------
        normalized_concentrations: torch.Tensor
            The normalized concentration of the H&E stains in each pixel.
        """
        output = []
        for conc, max_conc in zip(concentrations, maximum_concentration):
            norm_conc = conc * (self._max_con_reference.to(max_conc) / max_conc).unsqueeze(-1)
            output.append(norm_conc)
        return torch.stack(output, dim=0)

    def __create_normalized_images(
        self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Create the normalized images from the normalized concentrations.

        Parameters
        ----------
        normalized_concentrations: torch.Tensor
            The normalized concentrations of the H&E stains in the image.
        image_tensor: torch.Tensor
            The image tensor to be normalized.

        Returns
        -------
        normalized_images: torch.Tensor
            The normalized images.
        """
        batch, classes, height, width = image_tensor.shape
        # recreate the image using reference mixing matrix
        normalised_image_tensor = self.convert_optical_density_to_rgb(od_tensor=normalized_concentrations)
        normalised_image_tensor = normalised_image_tensor.mT.reshape(batch, height, width, classes)
        normalised_image_tensor = _transpose_channels(normalised_image_tensor)
        return normalised_image_tensor

    def __get_h_stain(self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the H-stain from the normalized concentrations.

        Parameters
        ----------
        normalized_concentrations: torch.Tensor
            The normalized concentrations of the H&E stains in the image.
        image_tensor: torch.Tensor
            The image tensor to be normalized.

        Returns
        -------
        h_stain: torch.Tensor
            The H-stain.
        """
        batch, classes, height, width = image_tensor.shape
        hematoxylin_tensors = torch.mul(
            self._transmitted_intensity,
            torch.exp(
                torch.matmul(-self._he_reference[:, 0].unsqueeze(-1), normalized_concentrations[0, :].unsqueeze(0))
            ),
        )
        hematoxylin_tensors[hematoxylin_tensors > 255] = 255
        hematoxylin_tensors = hematoxylin_tensors.T.reshape(batch, height, width, classes)
        hematoxylin_tensors = _transpose_channels(hematoxylin_tensors)
        return hematoxylin_tensors

    def __get_e_stain(self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get the E-stain from the normalized concentrations.

        Parameters
        ----------
        normalized_concentrations: torch.Tensor
            The normalized concentrations of the H&E stains in the image.
        image_tensor: torch.Tensor
            The image tensor to be normalized.

        Returns
        -------
        e_stain: torch.Tensor
            The E-stain.
        """
        batch, classes, height, width = image_tensor.shape
        eosin_tensors = torch.mul(
            self._transmitted_intensity,
            torch.exp(
                torch.matmul(-self._he_reference[:, 1].unsqueeze(-1), normalized_concentrations[1, :].unsqueeze(0))
            ),
        )
        eosin_tensors[eosin_tensors > 255] = 255
        eosin_tensors = eosin_tensors.T.reshape(batch, height, width, classes)
        eosin_tensors = _transpose_channels(eosin_tensors)
        return eosin_tensors

    def forward(
        self,
        *args: tuple[torch.Tensor],
        data_keys: Optional[list],
        staining_parameters: Optional[dict[str : torch.Tensor, str : torch.Tensor] | None] = None,
    ) -> list[torch.Tensor]:
        output = []
        for sample, data_key in zip(args, data_keys):
            if data_key in ["image"]:
                image_tensor: torch.Tensor = sample[data_key]
                _, concentrations, maximum_concentration = self.__compute_matrices(
                    image_tensor, staining_parameters=staining_parameters
                )
                if staining_parameters is not None:
                    maximum_concentration = staining_parameters["max_wsi_concentration"]

                normalized_concentrations = self.__normalize_concentrations(concentrations, maximum_concentration)
                normalised_image = self.__create_normalized_images(normalized_concentrations, image_tensor)
                output.append(normalised_image)
            # if self._return_stains:
            #     stains["image_hematoxylin"] = self.__get_h_stain(normalized_concentrations, image_tensor)
            #     stains["image_eosin"] = self.__get_e_stain(normalized_concentrations, image_tensor)
        if len(output) == 1:
            return output[0]

        return output

    def __repr__(self):
        return (
            f"{type(self).__name__}(alpha={self._alpha}, beta={self._beta}, "
            f"transmitted_intensity={self._transmitted_intensity}, probability={self._probability})"
        )
