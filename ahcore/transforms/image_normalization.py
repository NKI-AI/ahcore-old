# encoding: utf-8
"""
Histopathology stain specific image normalization functions
# TODO: Support `return_stains = True` for MacenkoNormalization()
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from pathlib import Path
import h5py
from kornia.constants import DataKey

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


def dump_staining_parameters(staining_parameters: dict, path_to_folder: Path) -> None:
    """
    This function dumps the staining parameters to a h5 file.
    Parameters
    ----------
    staining_parameters: dict
        Staining parameters
    """
    if not path_to_folder.exists():
        path_to_folder.mkdir(parents=True)

    with h5py.File(path_to_folder / str(staining_parameters["wsi_name"] + ".h5"), "w") as hf:
        for key, value in staining_parameters.items():
            hf.create_dataset(key, data=value)


def _handle_stain_tensors(stain_tensor: torch.tensor, shape) -> torch.Tensor:
    """
    Parameters
    ----------
    stain_tensor
    Returns
    -------
    """
    batch, channels, height, width = shape
    stain_tensor[stain_tensor > 255] = 255
    stain_tensor = stain_tensor.reshape(batch, channels, height, width)
    return stain_tensor


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
        normalised_image_tensor = []
        if len(od_tensor.shape) == 2:
            normalised_image_tensor = self._transmitted_intensity * torch.exp(-self._he_reference.to(od_tensor) @ od_tensor)
        else:
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
            self, image_tensor: torch.Tensor, staining_parameters: dict[str: torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the H&E staining vectors and their concentration values for every pixel in the image tensor.
        """
        batch_con_vecs = []
        # Convert RGB values in the image to optical density values following the Beer-Lambert's law.
        # Note - The dependence of staining and their concentrations are linear in OD space.
        optical_density, _ = self.convert_rgb_to_optical_density(image_tensor)
        for i in range(len(optical_density)):
            od_tensor = optical_density[i].reshape(-1, 3)
            he = staining_parameters["wsi_staining_vectors"][i]
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
            Dimensions of HE vector are: (3 x 2)
            Dimensions of max cancentration vector are: (2)
            Dimensions of wsi_eigenvectors are: (3 x 2)
        """
        logger.info("Fitting stain matrix for WSI: %s", wsi_name)
        optical_density, optical_density_hat = self.convert_rgb_to_optical_density(wsi)
        optical_density = optical_density.squeeze(0).reshape(-1, 3)
        wsi_eigenvectors = _compute_eigenvecs(optical_density_hat[0])
        wsi_level_he = self._find_he_components(optical_density_hat[0], wsi_eigenvectors)
        _, wsi_level_max_concentrations = _compute_concentrations(wsi_level_he, optical_density)
        staining_parameters = {
            "wsi_name": wsi_name,
            "wsi_staining_vectors": wsi_level_he.unsqueeze(0),
            "max_wsi_concentration": wsi_level_max_concentrations.unsqueeze(0),
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
        staining_parameters = self.fit(wsi=target_image, wsi_name="target image")
        self._he_reference = staining_parameters["wsi_staining_vectors"]
        self._max_con_reference = staining_parameters["max_wsi_concentration"]

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
        if len(concentrations.shape) == 2:
            normalised_concentration = concentrations * (self._max_con_reference.to(maximum_concentration) / maximum_concentration).unsqueeze(-1)
        else:
            for conc, max_conc in zip(concentrations, maximum_concentration):
                norm_conc = conc * (self._max_con_reference.to(max_conc) / max_conc).unsqueeze(-1)
                output.append(norm_conc)
            normalised_concentration = torch.stack(output, dim=0)
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
        # recreate the image using reference mixing matrix
        normalised_image_tensor = self.convert_optical_density_to_rgb(od_tensor=normalized_concentrations)
        normalised_image_tensor = normalised_image_tensor.reshape(batch, classes, height, width)
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
        """

        h_stain = torch.mul(
            self._transmitted_intensity,
            torch.exp(
                torch.matmul(-self._he_reference[:, 0].unsqueeze(-1), normalized_concentrations[0, :].unsqueeze(0))
            ),
        )
        e_stain = torch.mul(
            self._transmitted_intensity,
            torch.exp(
                torch.matmul(-self._he_reference[:, 1].unsqueeze(-1), normalized_concentrations[1, :].unsqueeze(0))
            ),
        )
        h_stain = _handle_stain_tensors(h_stain, image_tensor.shape)
        e_stain = _handle_stain_tensors(e_stain, image_tensor.shape)
        return h_stain, e_stain

    def forward(self, *args: tuple[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        output = []
        data_keys = kwargs["data_keys"]
        if "staining_parameters" in kwargs.keys():
            staining_parameters = kwargs["staining_parameters"]
        else:
            staining_parameters = None
        for sample, data_key in zip(args, data_keys):
            if data_key in [DataKey.INPUT, 0, "INPUT"]:
                concentrations = self.__compute_matrices(sample, staining_parameters=staining_parameters)
                maximum_concentration = staining_parameters["max_wsi_concentration"]
                normalized_concentrations = self.__normalize_concentrations(concentrations, maximum_concentration)
                normalised_image = self.__create_normalized_images(normalized_concentrations, sample)
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
