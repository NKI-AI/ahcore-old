# encoding: utf-8
"""
Histopathology stain specific image normalization functions
# TODO: Support `return_stains = True` for MacenkoNormalization()
"""
from __future__ import annotations

from typing import Optional

import torch
import os
import torch.nn as nn
from functools import lru_cache
from pathlib import Path
import numpy as np
import h5py

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


def dump_staining_parameters(staining_parameters: dict, path_to_folder: Path) -> None:
    """
    This function dumps the staining parameters to a h5 file.
    Parameters
    ----------
    staining_parameters: dict
        Staining parameters
    path_to_folder: Path
        Path to the folder where the h5 file must be saved.
    """
    if not path_to_folder.exists():
        path_to_folder.mkdir(parents=True)

    with h5py.File(path_to_folder / str(staining_parameters["wsi_name"] + ".h5"), "w") as hf:
        for key, value in staining_parameters.items():
            hf.create_dataset(key, data=value)


@lru_cache(maxsize=None)
def _load_vector_from_h5_file(filename):
    path_to_stains = (
            Path(os.environ.get("SCRATCH", "/tmp")) / "ahcore_cache" / "staining_parameters" / (filename + ".h5"))
    if not path_to_stains.exists():
        raise FileNotFoundError(f"Staining parameters not found for {filename}")
    staining_vectors = h5py.File(path_to_stains, 'r')
    he_staining_vectors = torch.tensor(np.array(staining_vectors["wsi_staining_vectors"]))
    max_concentrations = torch.tensor(np.array(staining_vectors["max_wsi_concentration"]))
    return he_staining_vectors, max_concentrations


def load_stainings_from_cache(filenames: list[str]) -> dict:
    """
    This function loads the staining vectors for a given WSI from the cache.
    Parameters
    ----------
    filenames: list[str]
        List of filenames for which the staining vectors must be loaded.
    Returns
    -------
    staining_vectors: dict
        Dictionary containing the staining vectors for the given filenames.
    """
    staining_parameters = {}
    hes = []
    max_concentrations = []
    for filename in filenames:
        he, max_con = _load_vector_from_h5_file(Path(filename).stem)
        hes.append(he)
        max_concentrations.append(max_con)
    staining_parameters["wsi_staining_vectors"] = torch.stack(hes)
    staining_parameters["max_wsi_concentration"] = torch.stack(max_concentrations)
    return staining_parameters


def _handle_stain_tensors(stain_tensor: torch.tensor, shape) -> tuple[torch.Tensor,torch.Tensor]:
    """
    This function post-processes the individual staining channels and returns them.
    Parameters
    ----------
    stain_tensor: torch.Tensor
        Tensor containing the H and E vectors
    shape: tuple
        Shape of the image
    Returns
    -------
    h_stain: torch.Tensor
        Tensor containing the H vector
    e_stain: torch.Tensor
        Tensor containing the E vector
    """
    individual_stains = []
    for stain in stain_tensor:
        batch, channels, height, width = shape
        stain = torch.clamp(stain, 0.0, 255.0)
        stain = stain.view(batch, channels, height, width)
        individual_stains.append(stain)
    h_stain = individual_stains[0]
    e_stain = individual_stains[1]
    return h_stain, e_stain


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
    he_concentrations = he_vector.to(optical_density).pinverse() @ optical_density.T
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
    >>> HE_REFERENCE = torch.tensor([[[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]])
    >>> MAX_CON_REFERENCE = torch.tensor([[1.9705, 1.0308]])
    """

    HE_REFERENCE = torch.Tensor([[[0.5042, 0.1788], [0.7723, 0.8635], [0.3865, 0.4716]]])
    MAX_CON_REFERENCE = torch.Tensor([[1.3484, 1.0886]])

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
        self._alpha = torch.tensor(alpha)
        self._beta = torch.tensor(beta)
        self._transmitted_intensity = torch.tensor(transmitted_intensity)
        self._return_stains = return_stains
        if self._return_stains:
            raise NotImplementedError("Return stains is not implemented yet.")
        self._probability = probability
        self._he_reference = self.HE_REFERENCE
        self._max_con_reference = self.MAX_CON_REFERENCE

    def __compute_matrices(self, image_tensor: torch.Tensor, staining_parameters: dict[str: torch.Tensor]) -> torch.Tensor:
        """
        Compute the H&E staining vectors and their concentration values for every pixel in the image tensor.
        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor
        staining_parameters : dict[str: torch.Tensor]
            The staining parameters
        Returns
        -------
        he_concentrations : torch.Tensor
            Concentrations of the individual stains.
        """
        batch_con_vecs = []
        # Convert RGB values in the image to optical density values following the Beer-Lambert's law.
        # Note - The dependence of staining and their concentrations are linear in OD space.
        optical_density, _ = self.convert_rgb_to_optical_density(image_tensor)
        for sample_idx in range(len(optical_density)):
            od_tensor = optical_density[sample_idx].view(-1, 3)
            he = staining_parameters["wsi_staining_vectors"][sample_idx]
            # Calculate the concentrations of the H&E stains in each pixel.
            # We do this by solving a linear system of equations. (In this case, the system is overdetermined).
            # OD =   HE * C -> (1)
            # where:
            #     1. OD is the optical density of the pixels in the batch. The dimension is: (n x 3)
            #     2. HE is the H&E staining vectors (3 x 2). The dimension is: (3 x 2)
            #     3. C is the concentration of the H&E stains in each pixel. The dimension is: (2 x n)
            he_concentrations, _ = _compute_concentrations(he, od_tensor)
            batch_con_vecs.append(he_concentrations)
        return torch.stack(batch_con_vecs, dim=0)

    def __normalize_concentrations(
            self, concentrations: torch.Tensor, maximum_concentration: torch.tensor
    ) -> torch.Tensor:
        """
        Normalize the concentrations of the H&E stains in each pixel against the reference concentration.
        Parameters
        ----------
        concentrations: torch.Tensor
            The concentration of the H&E stains in each pixel.
        maximum_concentration: torch.Tensor
            The maximum concentration of the H&E stains in each pixel.
        Returns
        -------
        normalized_concentrations: torch.Tensor
            The normalized concentration of the H&E stains in each pixel.
        """
        scaled_reference = torch.div(self._max_con_reference, maximum_concentration)
        scaled_reference = scaled_reference.unsqueeze(1).permute(0, 2, 1).contiguous()
        normalised_concentration = torch.mul(concentrations, scaled_reference.to(concentrations))
        return normalised_concentration

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
        normalised_image_tensor = self.convert_optical_density_to_rgb(od_tensor=normalized_concentrations)
        normalised_image_tensor = normalised_image_tensor.view(batch, classes, height, width)
        return normalised_image_tensor

    def __get_stains(
            self, normalized_concentrations: torch.Tensor, image_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the H-stain and the E-stain from the normalized concentrations.
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
        e_stain: torch.Tensor
            The E-stain.
        """
        stains = []
        for i in range(0, 1):
            stains.append(torch.mul(
                self._transmitted_intensity,
                torch.exp(
                    torch.matmul(-self._he_reference[:, i].unsqueeze(-1), normalized_concentrations[i, :].unsqueeze(0))
                ),
            ))
        h_stain, e_stain = _handle_stain_tensors(stains, image_tensor.shape)
        return h_stain, e_stain

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
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous()
        num_tiles, height, width, channels = image_tensor.shape
        # calculate optical density
        optical_density = -torch.log((image_tensor.float() + 1) / self._transmitted_intensity)
        # remove transparent pixels
        mask = optical_density.min(dim=-1).values > self._beta
        optical_density_hat = [optical_density[i][mask[i]] for i in range(num_tiles)]
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
        # recreate the image using reference mixing matrix
        projection_to_reference_stains = -self._he_reference.to(od_tensor) @ od_tensor
        optical_density_to_rgb = torch.exp(projection_to_reference_stains)
        normalised_image_tensor = optical_density_to_rgb * self._transmitted_intensity
        normalised_image_tensor = torch.clamp(normalised_image_tensor, 0.0, 255.)
        return normalised_image_tensor

    def fit(self, wsi: torch.Tensor, wsi_name: str, dump_to_folder: Optional[Path] = None) -> dict[str: torch.Tensor]:
        """
        Compress a WSI to a single matrix of eigenvectors and return staining parameters.
        Parameters:
        ----------
        wsi: torch.tensor
            A tensor containing a whole slide image of shape (1, channels, height, width)
        name: Path
            Path to the WSI file
        Returns:
        -------
        staining_parameters: dict[str: torch.Tensor, str: torch.Tensor]
            The eigenvectors of the optical density values of the pixels in the image.
        Note:
            Dimensions of HE_vector are: (3 x 2)
            Dimensions of max concentration vector are: (2)
            Dimensions of wsi_eigenvectors are: (3 x 2)
        """
        logger.info("Fitting stain matrix for WSI: %s", wsi_name)
        optical_density, optical_density_hat = self.convert_rgb_to_optical_density(wsi)
        optical_density = optical_density.squeeze(0).view(-1, 3)
        wsi_eigenvectors = _compute_eigenvecs(optical_density_hat[0])
        wsi_level_he = self._find_he_components(optical_density_hat[0], wsi_eigenvectors)
        wsi_level_concentrations, wsi_level_max_concentrations = _compute_concentrations(wsi_level_he, optical_density)
        staining_parameters = {
            "wsi_name": wsi_name,
            "wsi_staining_vectors": wsi_level_he,
            "max_wsi_concentration": wsi_level_max_concentrations,
        }
        if dump_to_folder:
            dump_staining_parameters(staining_parameters, dump_to_folder)
        return staining_parameters

    def set(self, target_image: torch.Tensor) -> None:
        """
        Set the reference image for the stain normaliser.
        Parameters:
        ----------
        target_image: torch.Tensor
            The reference image for the stain normaliser.
        """
        logger.info("Setting image for reference stainings...")
        staining_parameters = self.fit(wsi=target_image, wsi_name="target image")
        self._he_reference = staining_parameters["wsi_staining_vectors"]
        self._max_con_reference = staining_parameters["max_wsi_concentration"]

    def forward(self, *args: tuple[torch.Tensor], **kwargs) -> tuple[torch.Tensor]:
        args = list(args)
        sample = args[0]
        if "staining_parameters" in kwargs.keys():
            staining_parameters = kwargs["staining_parameters"]
        else:
            filenames = kwargs["filenames"]
            staining_parameters = load_stainings_from_cache(filenames)
        tile_concentrations = self.__compute_matrices(sample, staining_parameters=staining_parameters)
        wsi_maximum_concentration = staining_parameters["max_wsi_concentration"]
        normalized_concentrations = self.__normalize_concentrations(tile_concentrations, wsi_maximum_concentration)
        normalised_image_tensor = self.__create_normalized_images(normalized_concentrations, sample)
        args[0] = normalised_image_tensor
        # if self._return_stains:
        #     stains["image_hematoxylin"] = self.__get_h_stain(normalized_concentrations, image_tensor)
        #     stains["image_eosin"] = self.__get_e_stain(normalized_concentrations, image_tensor)
        return tuple(args)

    def __repr__(self):
        return (
            f"{type(self).__name__}(alpha={self._alpha}, beta={self._beta}, "
            f"transmitted_intensity={self._transmitted_intensity}, probability={self._probability})"
        )
