import matplotlib.pyplot as plt
import numpy as np

from utils import util
from config import AFF

class Element(object):
    """Element

    Objects represent an element from the periodic table and contain methods to assist with
    the determination of atomic form factors used in x-ray diffraction modeling.

    Attributes:  
        name: (str) The element name, e.g. 'Au'
    """  
    def __init__(self, name):
        self.name = name
        self._get_aff_parameters()

    def _get_aff_parameters(self):
        """Helper function that parses config for the given element.
        
        Sets parameters used in AFF calculation.

        Returns: 
            None
        """
        aff_data = [record for record in AFF if record[u'element'] == self.name]
        if not aff_data:
            raise NotImplementedError(u'element {} not supported'.format(e))
        else:
            record = aff_data[0]
            self._a1 = float(record[u'a1'])
            self._a2 = float(record[u'a2'])
            self._a3 = float(record[u'a3'])
            self._a4 = float(record[u'a4'])
            self._b1 = float(record[u'b1'])
            self._b2 = float(record[u'b2'])
            self._b3 = float(record[u'b3'])
            self._b4 = float(record[u'b4'])
            self._c = float(record[u'c'])

    def atomic_form_factor(self, q, weight=1.0):
        """Calculates the atomic form factor evaluated at q.

        Args:  
            q:  Either a list or single momentum transfer value
            weight:  (float, optional) A multiplicative factor to apply to the form factor

        Returns:  
            The AFF evaluated at all given q values.
        """
        constants = [(self._a1, self._b1),
                     (self._a2, self._b2),
                     (self._a3, self._b3),
                     (self._a4, self._b4)]
        return weight * (np.sum([a * np.exp(-b * (q / 4.0 * np.pi)**2.0)
                            for a, b in constants], axis=0) + self._c)


class SingleCrystal(object):
    """Single Crystal

    Objects represent single solid crystal and contain methods to generate crystallographic 
    properties related to x-ray diffraction, such as the atomic form factor of constituent atoms,
    the structure factor, scattering intensity, etc.  

    Attributes: 
        elements:  (list) The elements in the basis, e.g. H2O = ['H', 'H', 'O'].  Order should match basis.
        lattice_constant:  (float) The primary lattice spacing for this crystal.
        lattice_vectors:  (list) The lattice vectors that determine the crystalline structure.
        basis_vectors:  (list) The basis vectors for each atom in direct coordinates.
    """
    def __init__(self, elements, lattice_constant, lattice_vectors, basis_vectors): 
        self.elements = elements
        self.lattice_constant = lattice_constant
        self.lattice_vectors = lattice_vectors
        self.basis_vectors = basis_vectors # Assuming basis vector in direct coordinates

    @property
    def elements(self):
        return self._elements

    @property
    def lattice_constant(self):
        return self._lattice_constant

    @property
    def lattice_vectors(self):
        return self._lattice_vectors

    @property
    def basis_vectors(self):
        return self._basis_vectors

    @elements.setter
    def elements(self, element_names):
        if type(element_names) == list:
            self._elements = [Element(name) for name in element_names]
        elif isinstance(element_names, basestring):
            self._elements = [Element(element_names)] 
        else:
            raise ValueError(u'elements must be strings or list of strings.')

    @lattice_constant.setter
    def lattice_constant(self, lc):
        if not isinstance(lc, float):
            raise ValueError(u'Lattice constant must be a float.')
        else:
            self._lattice_constant = lc

    @lattice_vectors.setter
    def lattice_vectors(self, lv):
        self._lattice_vectors = np.array(lv)

    @basis_vectors.setter
    def basis_vectors(self, bv):
        _basis_vectors = np.array(bv)
        if len(_basis_vectors.shape) == 1:
            _basis_vectors = np.array([bv])
        self._basis_vectors = _basis_vectors

        assert self._basis_vectors.shape[0] == len(self.elements), u'Length of basis vectors and elements must be equal.'

    def reciprocal_vectors(self):
        """Form the reciprocal lattice vectors.

        b_i = 2pi * (a_j x a_k) / volume

        Returns:
            A numpy array.
        """
        a1 = self.lattice_constant * self.lattice_vectors[0]
        a2 = self.lattice_constant * self.lattice_vectors[1]
        a3 = self.lattice_constant * self.lattice_vectors[2]
        vol = np.dot(a1, np.cross(a2, a3))

        b1 = 2 * np.pi * (np.cross(a2, a3) / vol)
        b2 = 2 * np.pi * (np.cross(a3, a1) / vol)
        b3 = 2 * np.pi * (np.cross(a1, a2) / vol)
        return np.array([b1, b2, b3], dtype='f8')

    def structure_factor(self, q_vect, weights=None):
        """Calculature the structure factor.

        S(q) = basis_sum(atomic_form_factor_i * exp(-j q.r_i))

        Args:  
            q_vect:  A 1 or 2D array of momentum transfer vectors
            weights:  An array of multiplicative factors to apply to each atom's AFF

        Returns:   
            The structure factor evaluated at each given vector.
        """
        partial_factors = []
        for i, element in enumerate(self.elements):
            if not weights:
                aff = element.atomic_form_factor(np.linalg.norm(q_vect, axis=-1))
            else:
                assert len(weights) == len(self.elements), u'Must include a weight for each element.'
                aff = element.atomic_form_factor(np.linalg.norm(q_vect, axis=-1), weight=weights[i])

            partial_factors.append(aff * np.exp(-1j * np.dot(q_vect, self.basis_vectors[i])))

        return np.sum(partial_factors, axis=0)

    def get_xrd_pattern(self, hkl_min=-2, hkl_max=2, decimals=1, **kwargs):
        """Calculates the scattering intensity seen in x-ray diffraction

        Args: 
            hkl_min, hkl_max: (int, optional) min and max values for the Miller indices
            decimals: (int, optional) The number of decimals used to set the bin size
            **kwargs:
                weights:  (optional) An array of multiplicative factors to apply to each atom's AFF
                debye_waller:  (float, optional) The scattering center displacement used in the Debye-Waller factor
                energy: (float, optional) An energy in keV, converts the q-axis to angle
                lorentz: (boolean, optional) Specifies whether or not to apply a Lorentz factor, default True
                gaussian_peaks:  (boolean, optional) If True, applies gaussians centered at each peak
                gaussian_width:  (float, optional) The std of the gaussians
                gaussian_step:  (float, optional) The q-axis step size
                gaussian_q_min:  (float, optional) The first q-axis value to use in sampling
                gaussian_q_max:  (float, optional) the last q-axis value to use in sampling

        Returns:  
            A tuple (q, I(q))
        """
        q_vect = self.get_reciprocal_points(hkl_min, hkl_max, return_vectors=True)
        sf_sqd = np.absolute(self.structure_factor(q_vect, weights=kwargs.get(u'weights')))
        q_norm, idx, wts = self.get_reciprocal_points(hkl_min, hkl_max, decimals=decimals)
        I_q = sf_sqd[idx] * wts

        debye_waller = kwargs.get(u'debye_waller')
        if debye_waller:
            I_q = np.array([I_q[i] * np.exp(((-q**2) * (debye_waller**2)) / 3.0) for i, q in enumerate(q_norm)])

        energy = kwargs.get(u'energy')
        if energy: 
            q_norm = util.angle_convert(q_norm, energy)
            if kwargs.get(u'lorentz', True):
                deg_to_rad = np.pi / 180.0
                I_q = np.array([I_q[i] / (np.cos(q * deg_to_rad) * np.sin(q * deg_to_rad)**2) for i, q in enumerate(q_norm)])

        if kwargs.get(u'gaussian_peaks'):
            step_q = kwargs.get(u'gaussian_step', 0.01)
            std = kwargs.get(u'gaussian_width', 0.1)
            min_q = kwargs.get(u'gaussian_q_min', q_norm.min() - 3.0 * std)
            max_q = kwargs.get(u'gaussian_q_max', q_norm.max() + 3.0 * std)
            
            q_range = np.arange(min_q, max_q, step_q)
            gaussians = [I_q[i] * util.norm(q_range, mu, std=std) for i, mu in enumerate(q_norm)]

            q_norm = q_range
            I_q = np.sum(gaussians, axis=0)

        return (q_norm, I_q)

    def plot_xrd_pattern(self, hkl_min=-2, hkl_max=2, decimals=1, **kwargs):
        """Plots the scattering intensity seen in x-ray diffraction

        Args: 
            hkl_min, hkl_max: (int, optional) min and max values for the Miller indices
            decimals: (int, optional) The number of decimals used to set the bin size
            **kwargs:
                weights:  (optional) An array of multiplicative factors to apply to each atom's AFF
                debye_waller:  (float, optional) The scattering center displacement used in the Debye-Waller factor
                energy: (float, optional) An energy in keV, converts the q-axis to angle
                lorentz: (boolean, optional) Specifies whether or not to apply a Lorentz factor, default True
                gaussian_peaks:  (boolean, optional) If True, applies gaussians centered at each peak
                gaussian_width:  (float, optional) The std of the gaussians
                gaussian_step:  (float, optional) The q-axis step size
                gaussian_q_min:  (float, optional) The first q-axis value to use in sampling
                gaussian_q_max:  (float, optional) the last q-axis value to use in sampling

        Returns:  
            A figure.
        """
        X, Y = self.get_xrd_pattern(hkl_min=hkl_min, hkl_max=hkl_max, **kwargs)

        fig = plt.figure()
        plt.ylim(min(Y) - 0.1 * (max(Y) - min(Y)), max(Y) + 0.1 * (max(Y) - min(Y)))

        if not kwargs.get(u'gaussian_peaks'):
            plt.scatter(X, Y)
            y_min, y_max = plt.ylim()
            for i, x in enumerate(X):
                plt.axvline(x=x, ymax=(Y[i] - y_min) / (y_max - y_min), linestyle='--')
        else:
            plt.plot(X, Y)

        return fig

    def get_miller_indices(self, hkl_min, hkl_max):
        """Helper function that forms a grid of Miller indices.

        Args:  
            hkl_min, hkl_max:  (int) min and max values for the Miller indices
        
        Returns:  
            A numpy array of Miller indices.
        """
        return np.mgrid[hkl_min:hkl_max+1:1,
                        hkl_min:hkl_max+1:1,
                        hkl_min:hkl_max+1:1].reshape(3, -1).T

    def get_reciprocal_points(self, hkl_min, hkl_max, return_vectors=False, decimals=1):
        """Get points in the reciprocal lattice space.

        Args:  
            hkl_min, hkl_max:  (int) min and max values for the Miller indices
            return_vectors:  (boolean, optional)  If True, returns vectors instead of magnitudes
            decimals:  (int, optional)  The number of decimals used to set the bin size
        
        Returns:  
            A list of reciprocal lattice points, or vectors
        """
        hkls = self.get_miller_indices(hkl_min, hkl_max)
        if return_vectors:
            q_vects = np.array([np.dot(self.reciprocal_vectors(), hkl) for hkl in hkls])
            return q_vects[np.argsort(np.linalg.norm(q_vects, axis=1))][1:]
        else:
            return np.unique(np.around(sorted([np.linalg.norm(np.dot(self.reciprocal_vectors(), hkl))
                                               for hkl in hkls])[1:], decimals=decimals),
                             return_index=True,
                             return_counts=True)
