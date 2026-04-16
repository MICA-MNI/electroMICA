import ants
import glob
import json
import nibabel as nib
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp   
import scipy.ndimage as ndi
import shutil
from pathlib import Path
from scipy.io import loadmat, savemat
from scipy.linalg import solve
from scipy.spatial.distance import pdist, squareform
from sklearn import base


def electroMICA_iEEG(output_folder, feature, electrodes=None, micapipe_derivatives=None, hippunfold_derivatives=None):
    """
    Project intracranial EEG features to the cortical and hippocampal surfaces produced by MICApipe,
    in order to easily incorporet electrophysiological data to the MICA structural and functional 
    analyses ecosystem.
    An homogeneous singe layer is used to model the brain, the depth electrodes are modeled as line
    segments, and the source of electric activity is assumed to be distributed on the cortical
    surface (instead of dipoles). The Boundadry Elements Method is used to compute the sensitivity
    profile of each contact, and used in assigning the feature value at each cortical vertex.   
    Hippocampal surfaces obtained with hippunfold can also be included in the analysis.
    If the sensitivity profiles have been computed previously, only the output folder and feature
    labels need to be specified, then the computation of the features map will be much faster.


    Parameters:
    -----------
    output_folder : str
        electroMICA derivatives folder, where the output will be stored
    feature : str
        feature label, or feature.tsv file
    electrodes : str, optional
        BIDS_iEEG folder (or electrodes.tsv file), optional
    micapipe_derivatives : str, optional
        MICA derivatives folder
    hippunfold_derivatives : str, optional
        hippunfold derivatives folder
    """

    # Constants
    ChanTresh = 0.05  
    GlobalTresh = 0.001
    ContactLength = 2  # mm
    ContactSize = 1.0053  # mm^2
    cond_brain = 0.33  # S.m

    # ------------------------------------------------------------
    # Handle input conditions
    # ------------------------------------------------------------
    if electrodes is not None and micapipe_derivatives is None:
        print("If electrodes are specified, cortical surfaces should also be.")
        return

    if electrodes is None and micapipe_derivatives is not None:
        print("If cortical surfeces are specified, electrodes should also be.")
        return

    if hippunfold_derivatives is not None and micapipe_derivatives is None:
        print("If hippocampal surfaces are specified, cortical surfaces should also be.")
        return

    # ------------------------------------------------------------
    # Resolve sub/ses paths
    # ------------------------------------------------------------
    sub, ses = os.path.split(output_folder)
    if ses.startswith("sub-"):   
        sub = output_folder
        ses = ""
        fol = os.path.join(sub, "")
    else:
        fol = os.path.join(sub, ses, "")
    se = f"{ses}_" if ses else ""

    # If building a new dataset, create directory structure
    if micapipe_derivatives is not None:
        os.makedirs(sub, exist_ok=True)
        os.makedirs(fol, exist_ok=True)
        for subf in ["anat", "feat", "maps", "model", "surf", "xfm"]:
            os.makedirs(os.path.join(fol, subf), exist_ok=True)

    _, sub = os.path.split(sub)

    # ------------------------------------------------------------
    # Handle feature inputs
    # ------------------------------------------------------------
    ff, fn = os.path.split(feature)

    if ff:
        # Feature specified with path → copy into feat folder
        for f in glob.glob(feature):
            dest = os.path.join(fol, "feat", os.path.basename(f))
            shutil.copy2(f, dest)
        feature = fn  # use just filename

    if not feature.endswith(".tsv"):
        feature = feature + ".tsv"

    if micapipe_derivatives is not None:
        # delete existing files from surf
        surf_files = list((Path(fol) / 'surf').glob(f"{sub}_*.gii*"))
        for mf in surf_files:
            os.remove(mf) 
    
    # ------------------------------------------------------------
    # Copy hippocampal surfaces if requested
    # ------------------------------------------------------------
    if hippunfold_derivatives is not None:
        _, seh = os.path.split(hippunfold_derivatives)

        # Ensure correct sub/ses navigation
        if not (seh.startswith("sub-") or seh.startswith("ses-")):
            hippunfold_derivatives = os.path.join(hippunfold_derivatives, sub)
            if os.path.isdir(os.path.join(hippunfold_derivatives, "ses-01")):
                hippunfold_derivatives = os.path.join(hippunfold_derivatives, "ses-01")

        src = os.path.join(hippunfold_derivatives, "surf",
                           f"{sub}_*space-T1w_den-*_label-hipp_midthickness.surf.gii*")
        for f in glob.glob(src):
            shutil.copy2(f, os.path.join(fol, "surf"))

    # ------------------------------------------------------------
    # If surfaces given, copy micapipe content
    # ------------------------------------------------------------
    if micapipe_derivatives is not None:

        _, seh = os.path.split(micapipe_derivatives)
        if not (seh.startswith("sub-") or seh.startswith("ses-")):
            micapipe_derivatives = os.path.join(micapipe_derivatives, sub)
            if os.path.isdir(os.path.join(micapipe_derivatives, "ses-01")):
                micapipe_derivatives = os.path.join(micapipe_derivatives, "ses-01")

        # Copy T1w + surfaces
        for pattern in [
            f"{sub}*_space-nativepro_T1w.nii*",
            f"{sub}*_space-nativepro_T1w_brain.nii*",
            f"{sub}*_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii*",
            f"{sub}_*space-nativepro_surf-fsnative_label-midthickness.surf.gii*", 
            f"{sub}_*space-nativepro_surf-fsaverage5_label-midthickness.surf.gii*"
        ]:
            for f in glob.glob(os.path.join(micapipe_derivatives, "anat", pattern)):
                shutil.copy2(f, os.path.join(fol, "anat"))
            for f in glob.glob(os.path.join(micapipe_derivatives, "surf", pattern)):
                shutil.copy2(f, os.path.join(fol, "surf"))

    # Handle electrodes
    if electrodes is not None:
        elec_path = Path(electrodes)
        if elec_path.suffix:  # It's a file
            electrode_file = str(elec_path)
            iEEG_folder = elec_path.parent
        else:  # It's a folder
            iEEG_folder = elec_path
            fie = elec_path.name
            if not (fie.startswith('ses-') or fie.startswith('sub-')):
                iEEG_folder = iEEG_folder / sub / ses
            # Find electrode file
            electrode_files = list(iEEG_folder.glob('*electrodes.tsv'))
            if electrode_files:
                electrode_file = str(electrode_files[0])
            else:
                raise FileNotFoundError("No electrodes.tsv file found")

    #####################################
    ####### Deal with electrodes ########
    #####################################
    fol=Path(fol)
    if electrodes is not None:
        # Compute transformation between electrode image and nativepro 
        tx_file = fol / 'xfm' / f"{sub}_{se}iEEG-volume_to_nativepro.mat"
        get_transform(sub, se, fol, iEEG_folder)
        # Determine contact positions in nativepro
        ContactName, ContactPosition = GetContactPositions(fol, sub, se, electrode_file, tx_file, ContactSize)
        # Determine contacts outside brain
        p1, t1, di, ContactOutsideBrain = ContactProperties(fol, sub, ContactPosition, ContactLength)

        ###################################################
        ######## Compute leadfield if necessary ###########
        ###################################################

        # delte existing files from model
        model_files = list((fol / 'model').glob(f"{sub}_{se}leadfield_hemi-*.mat"))
        for mf in model_files:
            os.remove(mf)
        # Compute leadfield
        ComputeSensitivityProfile(sub, se, fol, p1, t1, di, cond_brain, ContactLength, ContactName, ContactPosition, ContactOutsideBrain)

    ########################################################
    ############## Compute feature maps ####################
    ########################################################

    ComputeFeatureMaps(fol, feature, ChanTresh, GlobalTresh)


def electroMICA_ScalpEEG(output_folder, feature, electrodes=None,
                         micapipe_derivatives=None, hippunfold_derivatives=None):
    """
    Project scalp EEG features to the cortical and hippocampal surfaces produced by MICApipe,
    in order to easily incorporet electrophysiological data to the MICA structural and functional 
    analyses ecosystem.
    An three-layer model derived from the T1w anatomical MRI is used to model the head, and the 
    source of electric activity is modeled as a dipolar sheet over the cortical surface. 
    The Boundadry Elements Method is used to solve the fprward problem and compute the leadfield
    matrix. Five feature maps are computed on each surface, corresponding to very high, high, 
    medium, low, and very low signal-to-noise ratios for the features (in steps of 7 dB). A modified
    eLORETA algorithm with a non-diagonal weighting matrix to favor spatial correlation of the 
    source profile is used to estimate the inverse problem solution.
    profile of each contact, and used in assigning the feature value at each cortical vertex.   
    Hippocampal surfaces obtained with hippunfold can also be included in the analysis.
    If the leadfields have been computed previously, only the output folder and feature labels
    need to be specified, then the computation of the features map will be much faster.


    Parameters:
    -----------
    output_folder : str
        electroMICA derivatives folder, where the output will be stored
    feature : str
        feature label, or feature.tsv file
    electrodes : str, optional
        BIDS_EEG folder (or electrodes.tsv file), optional
    micapipe_derivatives : str, optional
        MICA derivatives folder
    hippunfold_derivatives : str, optional
        hippunfold derivatives folder
    """

    # ------------------------------------------------------------
    # Handle input conditions
    # ------------------------------------------------------------
    if electrodes is not None and micapipe_derivatives is None:
        print("If electrodes are specified, cortical surfaces should also be.")
        return

    if hippunfold_derivatives is not None and micapipe_derivatives is None:
        print("If hippocampal surfaces are specified, cortical surfaces should also be.")
        return

    # ------------------------------------------------------------
    # Resolve sub/ses paths
    # ------------------------------------------------------------
    sub, ses = os.path.split(output_folder)

    if ses.startswith("sub-"):   # output_folder = sub-XX
        sub = output_folder
        ses = ""
        fol = os.path.join(sub, "")
    else:
        fol = os.path.join(sub, ses, "")

    se = f"{ses}_" if ses else ""

    # If building a new dataset, create directory structure
    if micapipe_derivatives is not None:
        os.makedirs(sub, exist_ok=True)
        os.makedirs(fol, exist_ok=True)
        for subf in ["anat", "feat", "maps", "model", "surf", "xfm"]:
            os.makedirs(os.path.join(fol, subf), exist_ok=True)

    _, sub = os.path.split(sub)

    # ------------------------------------------------------------
    # Handle feature inputs
    # ------------------------------------------------------------
    ff, fn = os.path.split(feature)

    if ff:
        # Feature specified with path → copy into feat folder
        for f in glob.glob(feature):
            dest = os.path.join(fol, "feat", os.path.basename(f))
            shutil.copy2(f, dest)
        feature = fn  # use just filename

    if not feature.endswith(".tsv"):
        feature = feature + ".tsv"

    if micapipe_derivatives is not None:
        # delete existing files from surf
        surf_files = list((Path(fol) / 'surf').glob(f"{sub}_*.gii*"))
        for mf in surf_files:
            os.remove(mf) 
    # ------------------------------------------------------------
    # Copy hippocampal surfaces if requested
    # ------------------------------------------------------------
    if hippunfold_derivatives is not None:
        _, seh = os.path.split(hippunfold_derivatives)

        # Ensure correct sub/ses navigation
        if not (seh.startswith("sub-") or seh.startswith("ses-")):
            hippunfold_derivatives = os.path.join(hippunfold_derivatives, sub)
            if os.path.isdir(os.path.join(hippunfold_derivatives, "ses-01")):
                hippunfold_derivatives = os.path.join(hippunfold_derivatives, "ses-01")

        src = os.path.join(hippunfold_derivatives, "surf",
                           f"{sub}_*space-T1w_den-2*_label-hipp_midthickness.surf.gii*")
        if not glob.glob(src):
            src = os.path.join(hippunfold_derivatives, "surf",
                           f"{sub}_*space-T1w_den-8*_label-hipp_midthickness.surf.gii*")
        for f in glob.glob(src):
            shutil.copy2(f, os.path.join(fol, "surf"))

    # ------------------------------------------------------------
    # If surfaces given, copy micapipe content
    # ------------------------------------------------------------
    if micapipe_derivatives is not None:

        _, seh = os.path.split(micapipe_derivatives)
        if not (seh.startswith("sub-") or seh.startswith("ses-")):
            micapipe_derivatives = os.path.join(micapipe_derivatives, sub)
            if os.path.isdir(os.path.join(micapipe_derivatives, "ses-01")):
                micapipe_derivatives = os.path.join(micapipe_derivatives, "ses-01")

        # Copy T1w + surfaces
        for pattern in [
            f"{sub}*_space-nativepro_T1w.nii*",
            f"{sub}*_space-nativepro_T1w_brain.nii*",
            f"{sub}*_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii*"
        ]:
            for f in glob.glob(os.path.join(micapipe_derivatives, "anat", pattern)):
                shutil.copy2(f, os.path.join(fol, "anat"))
            for f in glob.glob(os.path.join(micapipe_derivatives, "surf", pattern)):
                shutil.copy2(f, os.path.join(fol, "surf"))

        # Copy transform
        for f in glob.glob(os.path.join(
                micapipe_derivatives, "xfm",
                f"{sub}_*_from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN*")):
            shutil.copy2(f, os.path.join(fol, "xfm"))

        # --------------------------------------------------------
        # Determine electrode file
        # --------------------------------------------------------
        if electrodes in [None, ""]:
            no_electrode_file = True
            electrode_file = None
        else:
            no_electrode_file = False

            if electrodes.endswith(".tsv"):
                electrode_file = electrodes

            else:
                # Navigate BIDS EEG folder
                EEG_BIDS_folder, sube = os.path.split(electrodes)
                sese = ""
                if sube.startswith("ses-"):
                    sese = sube
                    EEG_BIDS_folder, sube = os.path.split(EEG_BIDS_folder)

                EEG_BIDS_folder = os.path.join(EEG_BIDS_folder, sube, sese)

                matches = glob.glob(os.path.join(EEG_BIDS_folder, "*_electrodes.tsv"))
                electrode_file = matches[0] if matches else None

        # --------------------------------------------------------
        # Build BEM model
        # --------------------------------------------------------
        print("Constructing BEM surfaces...")
        p1, p2, p3, t1, t2, t3, c3, pes, rs = build_BEM_model(fol, sub, se)

        print("Determining electrode position on scalp...")
        pe, ContactName = place_electrodes(fol, sub, se,
                                           electrode_file, c3, pes, rs)

        print("Computing BEM linear system...")
        compute_BEM_linear_system(fol, sub, se,
            p1, p2, p3, t1, t2, t3, pe, ContactName, no_electrode_file)

        # Compute leadfields
        compute_leadfield(fol, sub, se)

    # ------------------------------------------------------------
    # Solve inverse problem (always done)
    # ------------------------------------------------------------
    solve_inverse_problem(fol, sub, se, feature)


def int_lp(Y, T, X=None, Ft=None):
    """
    INT = int_lp(Y, T[, X[, Ft]]);
    
    This function computes the integral
    Ft_m * ∫_{S_m} h_n(y) ∇(1 / |x_k - y|)·n_m ds(y)
    where S_m is the m-th triangular element of the surface mesh defined by
    Y (Nx3 nodes in 3D) and T (Mx3 node indices of the M triangular elements
    of the mesh), n_m is the unit normal vector of the m-th triangle, and
    h_n(y) is a linear function that takes value 1 in the n-th vertex of the
    triangular element, and 0 on the other two vertices. The normals to the
    triangles are obtained following the right hand rule for the vertices,
    i.e. the vertices in T must be ordered accordingly. All the integrals
    associated to the same node are grouped. 
    X is a Kx3 matrix of the 3D points where the result is desired. If X is
    omitted, it is assumed that X=Y.
    Ft is a vector of size Mx1. Each element is a constant associated to a
    triangle (i.e. a piecewise constant function defined on the surface). If
    omitted it is assumed to be 1 for all the triangles. It is useful when
    computing the electric potential with BEM. For more details see: 
    de Munck (1992) A linear discretization of the volume conductor boundary integral equation 
    using analytically integrated elements. IEEE Trans Biomed Eng doi: 10.1109/10.256433
    The output INT is of size KxN
    """

    if X is None:
        X = Y
    if Ft is None:
        Ft = np.ones(T.shape[0])
    np.seterr(invalid='ignore') 
    np.seterr(divide='ignore')
    INT = np.zeros((X.shape[0], np.max(T) + 1))

    for ii in range(T.shape[0]):
        # Calculate the distance vectors from X to the vertices of the triangle
        x1 = X[:, 0] - Y[T[ii, 0], 0]
        x2 = X[:, 0] - Y[T[ii, 1], 0]
        x3 = X[:, 0] - Y[T[ii, 2], 0]
        
        y1 = X[:, 1] - Y[T[ii, 0], 1]
        y2 = X[:, 1] - Y[T[ii, 1], 1]
        y3 = X[:, 1] - Y[T[ii, 2], 1]
        
        z1 = X[:, 2] - Y[T[ii, 0], 2]
        z2 = X[:, 2] - Y[T[ii, 1], 2]
        z3 = X[:, 2] - Y[T[ii, 2], 2]
        
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        r2 = np.sqrt(x2**2 + y2**2 + z2**2)
        r3 = np.sqrt(x3**2 + y3**2 + z3**2)

        # Cross product of vectors to compute the area and normal vector
        d = x1 * (y2 * z3 - z2 * y3) + y1 * (z2 * x3 - x2 * z3) + z1 * (x2 * y3 - y2 * x3)
        AS = 2 * np.arctan2(d, r1 * r2 * r3 + (x1 * x2 + y1 * y2 + z1 * z2) * r3 + (x1 * x3 + y1 * y3 + z1 * z3) * r2 + (x2 * x3 + y2 * y3 + z2 * z3) * r1)

        # Calculate g1, g2, g3 based on the distances between triangle vertices
        d12 = np.sqrt((x2[0] - x1[0])**2 + (y2[0] - y1[0])**2 + (z2[0] - z1[0])**2)
        g1 = -1 / d12 * np.log((r1 * d12 + x1 * (x2 - x1) + y1 * (y2 - y1) + z1 * (z2 - z1)) / (r2 * d12 + x2 * (x2 - x1) + y2 * (y2 - y1) + z2 * (z2 - z1)))
        
        d23 = np.sqrt((x3[0] - x2[0])**2 + (y3[0] - y2[0])**2 + (z3[0] - z2[0])**2)
        g2 = -1 / d23 * np.log((r2 * d23 + x2 * (x3 - x2) + y2 * (y3 - y2) + z2 * (z3 - z2)) / (r3 * d23 + x3 * (x3 - x2) + y3 * (y3 - y2) + z3 * (z3 - z2)))
        
        d31 = np.sqrt((x1[0] - x3[0])**2 + (y1[0] - y3[0])**2 + (z1[0] - z3[0])**2)
        g3 = -1 / d31 * np.log((r3 * d31 + x3 * (x1 - x3) + y3 * (y1 - y3) + z3 * (z1 - z3)) / (r1 * d31 + x1 * (x1 - x3) + y1 * (y1 - y3) + z1 * (z1 - z3)))
        
        # Compute the integrals over each vertex of the triangle
        d12 = (x1 - x2) * g1 + (x2 - x3) * g2 + (x3 - x1) * g3
        d23 = (y1 - y2) * g1 + (y2 - y3) * g2 + (y3 - y1) * g3
        d31 = (z1 - z2) * g1 + (z2 - z3) * g2 + (z3 - z1) * g3
        
        g1 = d * ((x2 - x3) * d12 + (y2 - y3) * d23 + (z2 - z3) * d31)
        g2 = d * ((x3 - x1) * d12 + (y3 - y1) * d23 + (z3 - z1) * d31)
        g3 = d * ((x1 - x2) * d12 + (y1 - y2) * d23 + (z1 - z2) * d31)

        # Compute the components of the normal vector
        mx1 = y2 * z3 - z2 * y3
        my1 = z2 * x3 - z3 * x2
        mz1 = x2 * y3 - y2 * x3
        mx2 = y3 * z1 - z3 * y1
        my2 = z3 * x1 - z1 * x3
        mz2 = x3 * y1 - y3 * x1
        mx3 = y1 * z2 - z1 * y2
        my3 = z1 * x2 - z2 * x1
        mz3 = x1 * y2 - y1 * x2

        # Calculate the normal vector
        nx = mx1 + mx2 + mx3
        ny = my1 + my2 + my3
        nz = mz1 + mz2 + mz3

        nm = (nx**2 + ny**2 + nz**2) / Ft[ii]

        # Calculate h1, h2, h3
        h1 = np.real((mx1 * nx + my1 * ny + mz1 * nz) * AS + g1) / nm
        h2 = np.real((mx2 * nx + my2 * ny + mz2 * nz) * AS + g2) / nm
        h3 = np.real((mx3 * nx + my3 * ny + mz3 * nz) * AS + g3) / nm
        
        # Handle NaN and Inf values
        h1[np.isnan(h1) | np.isinf(h1)] = 0
        h2[np.isnan(h2) | np.isinf(h2)] = 0
        h3[np.isnan(h3) | np.isinf(h3)] = 0

        # Add the contributions to the final result
        INT[:, T[ii, 0]] += h1
        INT[:, T[ii, 1]] += h2
        INT[:, T[ii, 2]] += h3

    return INT

def bem_fsl(P, T, p, H):
    """
    MG = bem_fsl(P, T, p, H)
    
    This function computes the source term for the EEG forward problem, for generators modelled
    as dipolar sheets with intensity varying linearly between nodes of a tessellated surface mesh,
    and the resulting Boundary Elements Method leadfield. It computes the contribution of each
    vertex of the surface mesh (with vertices P and faces T) to the electric potential at the 
    points in p. Since this can typically have prohibitive memory requirements, it is not 
    computed explicity, but pre-multiplied by the BEM linear system H. A detailed description of
    the method can be found in: von Ellenrieder et al., "On the EEG/MEG forward problem solution
    for distributed cortical sources," Med Biol Eng Comput, 2009. doi: 10.1007/s11517-009-0529-x
    """

    # Initialize MG as a zero matrix of appropriate shape
    np.seterr(invalid='ignore') 
    np.seterr(divide='ignore')
    MG = np.zeros((H.shape[0], P.shape[0]))

    for ii in range(T.shape[0]):
        # Calculate the distance vectors from each point to the triangle vertices
        x1 = p[:, 0] - P[T[ii, 0], 0]
        x2 = p[:, 0] - P[T[ii, 1], 0]
        x3 = p[:, 0] - P[T[ii, 2], 0]

        y1 = p[:, 1] - P[T[ii, 0], 1]
        y2 = p[:, 1] - P[T[ii, 1], 1]
        y3 = p[:, 1] - P[T[ii, 2], 1]

        z1 = p[:, 2] - P[T[ii, 0], 2]
        z2 = p[:, 2] - P[T[ii, 1], 2]
        z3 = p[:, 2] - P[T[ii, 2], 2]

        # Calculate the distances to each vertex of the triangle
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        r2 = np.sqrt(x2**2 + y2**2 + z2**2)
        r3 = np.sqrt(x3**2 + y3**2 + z3**2)

        # Cross product to compute the area and normal vector of the triangle
        d = x1 * (y2 * z3 - z2 * y3) + y1 * (z2 * x3 - x2 * z3) + z1 * (x2 * y3 - y2 * x3)
        AS = 2 * np.arctan2(d, r1 * r2 * r3 + (x1 * x2 + y1 * y2 + z1 * z2) * r3 +
                            (x1 * x3 + y1 * y3 + z1 * z3) * r2 + (x2 * x3 + y2 * y3 + z2 * z3) * r1)

        # Compute g1, g2, g3 based on distances between triangle vertices
        d12 = np.sqrt((x2[0] - x1[0])**2 + (y2[0] - y1[0])**2 + (z2[0] - z1[0])**2)
        g1 = -1 / d12 * np.log((r1 * d12 + x1 * (x2 - x1) + y1 * (y2 - y1) + z1 * (z2 - z1)) /
                               (r2 * d12 + x2 * (x2 - x1) + y2 * (y2 - y1) + z2 * (z2 - z1)))

        d23 = np.sqrt((x3[0] - x2[0])**2 + (y3[0] - y2[0])**2 + (z3[0] - z2[0])**2)
        g2 = -1 / d23 * np.log((r2 * d23 + x2 * (x3 - x2) + y2 * (y3 - y2) + z2 * (z3 - z2)) /
                               (r3 * d23 + x3 * (x3 - x2) + y3 * (y3 - y2) + z3 * (z3 - z2)))

        d31 = np.sqrt((x1[0] - x3[0])**2 + (y1[0] - y3[0])**2 + (z1[0] - z3[0])**2)
        g3 = -1 / d31 * np.log((r3 * d31 + x3 * (x1 - x3) + y3 * (y1 - y3) + z3 * (z1 - z3)) /
                               (r1 * d31 + x1 * (x1 - x3) + y1 * (y1 - y3) + z1 * (z1 - z3)))

        # Compute d12, d23, d31 for the triangle based on g1, g2, g3
        d12 = (x1 - x2) * g1 + (x2 - x3) * g2 + (x3 - x1) * g3
        d23 = (y1 - y2) * g1 + (y2 - y3) * g2 + (y3 - y1) * g3
        d31 = (z1 - z2) * g1 + (z2 - z3) * g2 + (z3 - z1) * g3

        # Compute the final g1, g2, g3 contributions
        g1 = d * ((x2 - x3) * d12 + (y2 - y3) * d23 + (z2 - z3) * d31)
        g2 = d * ((x3 - x1) * d12 + (y3 - y1) * d23 + (z3 - z1) * d31)
        g3 = d * ((x1 - x2) * d12 + (y1 - y2) * d23 + (z1 - z2) * d31)

        # Compute the components of the normal vector
        mx1 = y2 * z3 - z2 * y3
        my1 = z2 * x3 - z3 * x2
        mz1 = x2 * y3 - y2 * x3
        mx2 = y3 * z1 - z3 * y1
        my2 = z3 * x1 - z1 * x3
        mz2 = x3 * y1 - y3 * x1
        mx3 = y1 * z2 - z1 * y2
        my3 = z1 * x2 - z2 * x1
        mz3 = x1 * y2 - y1 * x2

        # Calculate the normal vector
        nx = mx1 + mx2 + mx3
        ny = my1 + my2 + my3
        nz = mz1 + mz2 + mz3

        # Compute the magnitude of the normal vector squared
        nm = nx**2 + ny**2 + nz**2

        # Compute h1, h2, h3 contributions
        h1 = ((mx1 * nx + my1 * ny + mz1 * nz) * AS + g1) / nm
        h2 = ((mx2 * nx + my2 * ny + mz2 * nz) * AS + g2) / nm
        h3 = ((mx3 * nx + my3 * ny + mz3 * nz) * AS + g3) / nm

        # Update the MG matrix with the computed contributions
        MG[:, T[ii, 0]] -= H @ h1
        MG[:, T[ii, 1]] -= H @ h2
        MG[:, T[ii, 2]] -= H @ h3

    return MG


def areasv(P, T):
    """
    Calculate surface areas at vertices of a triangular mesh. The are of each vertex is defined
    as the integral over the surface of the hat function that takes value 1 at the vertex and 0 
    at the neighbouring vertices.
    
    Parameters:
    -----------
    P : array_like, shape (N, 3)
        Surface vertices
    T : array_like, shape (M, 3)
        Triangle faces
    
    Returns:
    --------
    fv : ndarray, shape (N,)
        Area per vertex
    """
    nd = P.shape[0]
    fv = np.zeros(nd)
    
    # Calculate edge vectors and lengths
    v12 = P[T[:, 0], :] - P[T[:, 1], :]
    b3 = np.sqrt(np.sum(v12**2, axis=1))
    
    v23 = P[T[:, 1], :] - P[T[:, 2], :]
    b1 = np.sqrt(np.sum(v23**2, axis=1))
    
    v31 = P[T[:, 2], :] - P[T[:, 0], :]
    b2 = np.sqrt(np.sum(v31**2, axis=1))
    
    # Calculate heights 
    h1 = np.sqrt(np.maximum(0, b3**2 - (np.sum(v12*v23, axis=1)/b1)**2)) * b1 / 6
    h2 = np.sqrt(np.maximum(0, b1**2 - (np.sum(v23*v31, axis=1)/b2)**2)) * b2 / 6
    h3 = np.sqrt(np.maximum(0, b2**2 - (np.sum(v31*v12, axis=1)/b3)**2)) * b3 / 6
    
    # Accumulate areas at vertices
    for ii in range(T.shape[0]):
        fv[T[ii, 0]] += h1[ii]
        fv[T[ii, 1]] += h2[ii]
        fv[T[ii, 2]] += h3[ii]
    
    return fv


def armesf(P, n=None):
    """
    This function computes the spherical harmonics values in azimuth and elevation
    corresponding to the points 'P' up to order 'n-1' (i.e., n^2 terms).

    Parameters:
    P (ndarray): An array of points with shape (num_points, 3), where num_points is the number of points.
    n (int, optional): The order of the spherical harmonics (default is calculated based on the number of points).

    Returns:
    U (ndarray): The spherical harmonic values for each point, with shape (num_points, n^2).
    """
    num_points = P.shape[0]
    
    # Calculate n if not provided
    if n is None:
        n = int(np.floor(num_points**0.5 + 1))
    
    # Normalization matrix N
    N = np.ones((n, n)).astype(float)
    for jj in range(n):
        for kk in range(jj, n):
            k=np.float64(kk + 1)
            j=np.float64(jj + 1)
            N[kk,jj] = np.sqrt((2*k-1)*np.prod(np.arange(1,k-j+1))/(4*np.pi*np.prod(np.arange(1,k+j-1))))
    
    # Initialize result array U
    U = np.zeros((num_points, n**2)).astype(float)
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    
    # Compute spherical harmonics for each point
    for ii in range(1, num_points+1):
        r = np.linalg.norm(P[ii-1, :])
        ph = np.arctan2(P[ii-1, 1], P[ii-1, 0])  # Azimuth angle
        x = P[ii-1, 2] / r  # cos(elevation)
        sc = np.sqrt(1 - x**2)  # sin(elevation)
        
        # Precompute d1 and d2 for each n
        for jj in range(1, n+1):
            d1[jj-1] = (-sc)**(jj-1) * np.prod(np.float64(np.arange(1, 2*jj, 2)))
            d2[jj-1] = d1[jj-1] * x * (2 * jj - 1)
        
        # Recursion for computing L
        L = np.diag(d1) + np.diag(d2[0:n-1], -1)
        for jj in range(1,n+1):
            for kk in range(jj + 2, n+1):
                L[kk-1, jj-1] = (x * (2 * kk - 3) * L[kk - 2, jj-1] - (jj+kk-3) * L[kk - 3, jj-1]) / (kk-jj)
        
        # Spherical harmonics computation for each point
        sp = np.sqrt(2) * np.sin(np.arange(0, n) * ph)
        cp = np.sqrt(2) * np.cos(np.arange(0, n) * ph)
        cp[0] = cp[0] / np.sqrt(2)
        
        L = L * N
        La = L * np.transpose(np.repeat(sp[:,np.newaxis],n,1))
        Lb = L * np.transpose(np.repeat(cp[:,np.newaxis],n,1))
        
        # Fill U array
        U[ii-1, 0] = Lb[0, 0]
        for jj in range(2, n+1):
            U[ii-1, (jj-1)**2:jj**2] = np.concatenate([Lb[jj-1, :jj], La[jj-1, 1:jj]])

    return U


def esfera(r, N, m=0):
    """
    Creates a tesselated spherical surface with radius r. The number of vertices is 20*(4^N) if m=0
    (icosahedron base) or 40*(4^N) if m=1.
    """

    if m == 0:
        # ICOSAEDRO
        v = np.array([1, -1, -1])
        tau = np.max(np.roots(v))
        P = np.array([[0, tau, 1],
                      [0, tau, -1],
                      [0, -tau, 1],
                      [0, -tau, -1],
                      [1, 0, tau],
                      [-1, 0, tau],
                      [1, 0, -tau],
                      [-1, 0, -tau],
                      [tau, 1, 0],
                      [tau, -1, 0],
                      [-tau, 1, 0],
                      [-tau, -1, 0]]) / 2
        T = np.array([[2, 1, 9], [9, 7, 2], [11, 1, 2], [2, 8, 11], 
                      [6, 1, 11], [11, 12, 6], [5, 1, 6], [6, 3, 5], 
                      [9, 1, 5], [5, 10, 9], [10, 4, 7], [7, 9, 10], 
                      [7, 4, 8], [8, 2, 7], [8, 4, 12], [12, 11, 8], 
                      [12, 4, 3], [3, 6, 12], [3, 4, 10], [10, 5, 3]])
    else:
        # Cuerpo de 40 triangulos.
        o = 48 * np.pi / 180
        P = np.array([[0, 0, 1],
                      [np.cos(o), 0, np.sin(o)],
                      [np.cos(o) * np.cos(2 * np.pi / 5), np.cos(o) * np.sin(2 * np.pi / 5), np.sin(o)],
                      [np.cos(o) * np.cos(4 * np.pi / 5), np.cos(o) * np.sin(4 * np.pi / 5), np.sin(o)],
                      [np.cos(o) * np.cos(6 * np.pi / 5), np.cos(o) * np.sin(6 * np.pi / 5), np.sin(o)],
                      [np.cos(o) * np.cos(8 * np.pi / 5), np.cos(o) * np.sin(8 * np.pi / 5), np.sin(o)],
                      [1, 0, 0],
                      [np.cos(np.pi / 5), np.sin(np.pi / 5), 0],
                      [np.cos(2 * np.pi / 5), np.sin(2 * np.pi / 5), 0],
                      [np.cos(3 * np.pi / 5), np.sin(3 * np.pi / 5), 0],
                      [np.cos(4 * np.pi / 5), np.sin(4 * np.pi / 5), 0],
                      [-1, 0, 0],
                      [np.cos(6 * np.pi / 5), np.sin(6 * np.pi / 5), 0],
                      [np.cos(7 * np.pi / 5), np.sin(7 * np.pi / 5), 0],
                      [np.cos(8 * np.pi / 5), np.sin(8 * np.pi / 5), 0],
                      [np.cos(9 * np.pi / 5), np.sin(9 * np.pi / 5), 0],
                      [np.cos(o), 0, -np.sin(o)],
                      [np.cos(o) * np.cos(2 * np.pi / 5), np.cos(o) * np.sin(2 * np.pi / 5), -np.sin(o)],
                      [np.cos(o) * np.cos(4 * np.pi / 5), np.cos(o) * np.sin(4 * np.pi / 5), -np.sin(o)],
                      [np.cos(o) * np.cos(6 * np.pi / 5), np.cos(o) * np.sin(6 * np.pi / 5), -np.sin(o)],
                      [np.cos(o) * np.cos(8 * np.pi / 5), np.cos(o) * np.sin(8 * np.pi / 5), -np.sin(o)],
                      [0, 0, -1]])
        
        T = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6], 
                      [1, 6, 2], [2, 7, 8], [2, 8, 3], [3, 8, 9], 
                      [3, 9, 10], [4, 3, 10], [4, 10, 11], [4, 11, 12], 
                      [5, 4, 12], [5, 12, 13], [5, 13, 14], [6, 5, 14], 
                      [6, 14, 15], [6, 15, 16], [2, 6, 16], [2, 16, 7], 
                      [8, 7, 17], [8, 17, 18], [8, 18, 9], [10, 9, 18], 
                      [10, 18, 19], [10, 19, 11], [12, 11, 19], [12, 19, 20], 
                      [12, 20, 13], [14, 13, 20], [14, 20, 21], [14, 21, 15], 
                      [16, 15, 21], [16, 21, 17], [16, 17, 7], [22, 18, 17], 
                      [22, 19, 18], [22, 20, 19], [22, 21, 20], [22, 17, 21]])
    
    
    T = T - 1
    for j in range(N):
        P, T = triang4(P, T)  # Cuadriplica el número de triángulos.
        P = P * (r / np.sqrt(np.sum(P**2, axis=1, keepdims=True)))  # Ubica los puntos a distancia r del origen.
    
    return P, T


def triang4(P, T):
    """
    This function splits all triangles in 3D space (vertices P and faces T) in 4 equal parts.
    """

    m = T.shape[0]
    n = P.shape[0]
    
    T = T + 1
    B = np.column_stack((T[:, 1:3], T[:, 0]))  # equivalent to MATLAB's B = [T(:,2:3) T(:,1)]
    A = np.maximum(T, B) + n * np.minimum(T, B)
    A = A - 1
    T = T - 1

    a = np.zeros(n**2, dtype=int)
    a[A.flatten()] = 1
    a_indices = np.where(a)[0]
    
    Tn = np.zeros((4 * m, 3), dtype=int)
    P = np.concatenate((P , np.zeros((np.shape(a_indices)[0] , 3), dtype=float)))

    for ii in range(m):
        p1 = (P[T[ii, 0], :] + P[T[ii, 1], :]) / 2
        a1 = n + np.where(a_indices == A[ii, 0])[0][0]
        P[a1, :] = p1
        
        p2 = (P[T[ii, 1], :] + P[T[ii, 2], :]) / 2
        a2 = n + np.where(a_indices == A[ii, 1])[0][0]
        P[a2, :] = p2
        
        p3 = (P[T[ii, 2], :] + P[T[ii, 0], :]) / 2
        a3 = n + np.where(a_indices == A[ii, 2])[0][0]
        P[a3, :] = p3
        
        Tn[4 * ii, :] = [a1, T[ii, 1], a2]
        Tn[4 * ii + 1, :] = [a2, a3, a1]
        Tn[4 * ii + 2, :] = [a3, a2, T[ii, 2]]
        Tn[4 * ii + 3, :] = [T[ii, 0], a1, a3]
    
    return P, Tn


def coregant(iEEG, MICA, tx_file, out_img):
    """
    Coregisters T1-weighted images from same subject using ANTs.

    Parameters:
    - spect_path: str, path to SPECT image (moving)
    - t1_path: str, path to T1 image (fixed)
    - output_prefix: str or None, path prefix for saving outputs

    Returns:
    - dict with keys: 'warped_image', 'transform', 'inverse_transform'
    """
    # Load images
    iEEG_img = ants.image_read(iEEG)
    MICA_img = ants.image_read(MICA)

    # Perform registration
    registration = ants.registration(
        fixed=MICA_img,
        moving=iEEG_img,
        type_of_transform='Rigid'
    )

    # Save results
    ants.image_write(registration['warpedmovout'], out_img)
    tx=ants.read_transform(registration['fwdtransforms'][0])
    ants.write_transform(tx, tx_file)


def get_transform(sub, se, fol, iEEG_folder):
    """
    Locates necessary files and performs coregistration of iEEG anatomical image to MICA T1w image.
    """

    tx_file = fol / 'xfm' / f"{sub}_{se}iEEG-volume_to_nativepro.mat"
    # Find MICA file
    mica_files = list((fol / 'anat').glob(f"{sub}_*space-nativepro_T1w.nii*"))
    if not mica_files:
        raise FileNotFoundError("No MICA T1w file found")
    MICA = str(mica_files[0])
    
    out_img = fol / 'anat' / f"{sub}_{se}space-nativepro_iEEG.nii.gz"
    
    # Find iEEG anatomical file
    search_patterns = [
        f"{sub}_{se}*presurgical*.nii*",
        f"{sub}_{se}*T1w.nii*",
        f"{sub}_{se}*T1map.nii*",
        f"{sub}_{se}*.nii*"
    ]
    
    iEEG_file = None
    for pattern in search_patterns:
        files = list((iEEG_folder / 'anat').glob(pattern))
        if files:
            # Filter for specific types if multiple files
            if len(files) > 1:
                if 'presurgical' in pattern:
                    t1_files = [f for f in files if 'T1w' in f.name or 'T1map' in f.name]
                    if t1_files:
                        iEEG_file = str(t1_files[0])
                    else:
                        iEEG_file = str(files[0])
                elif 'T1w' in pattern:
                    mag_files = [f for f in files if 'part-mag' in f.name]
                    if mag_files:
                        iEEG_file = str(mag_files[0])
                    else:
                        iEEG_file = str(files[0])
                else:
                    iEEG_file = str(files[0])
            else:
                iEEG_file = str(files[0])
            break
    
    if iEEG_file is None:
        raise FileNotFoundError("No anatomical file found in BIDS/iEEG/.../anat/")
    
    # Coregister
    print("Coregistering iEEG space to MICA prospace...")
    coregant(iEEG_file, MICA, str(tx_file), str(out_img))


def ComputeSensitivityProfile(sub, se, fol, p1, t1, di, cond_brain, ContactLength,
                               ContactName, ContactPosition, ContactOutsideBrain):
    """
    This function computes the sensitivity profile on MICAs cortical surfaces for the iEEG
    contacts. It models the head as a single homogeneous conductor with conductivity cond_brain
    defined by the tessellated inner skull surface with vertices p1 and faces t1. The generator 
    is modelled as a set of dipolar sheets of linearly varying intensity along the cortical surface 
    tessellation. The contacts are modelled as line segment of length ContactLength, oriented
    along the directions di. The Boundary Elements Method is used to compute the sensitivity maps.
    Contacts located outside the brain (ContactOutsideBrain=True) are assigned null sensitivity.
    """

    surfaces = [
        'L_space-nativepro_surf-fsLR-32k_label-',
        'R_space-nativepro_surf-fsLR-32k_label-',
        'L_space-nativepro_surf-fsnative_label-',
        'R_space-nativepro_surf-fsnative_label-',
        'L_space-nativepro_surf-fsaverage5_label-',
        'R_space-nativepro_surf-fsaverage5_label-',
        'L_space-T1w_den-0p5mm_label-hipp_',
        'r_space-T1w_den-0p5mm_label-hipp_',
        'L_space-T1w_den-2mm_label-hipp_',
        'r_space-T1w_den-2mm_label-hipp_']
    
    delta = np.array([-0.90618, -0.538469, 0, 0.538469, 0.90618]) * ContactLength
    w = np.array([0.236927, 0.478629, 0.568889, 0.478629, 0.236927])
    H = int_lp(p1, t1)
    H = H - sp.spdiags(np.sum(H, axis=1), 0, H.shape[0], H.shape[1]) - 2*np.pi/H.shape[0]
    V = np.linalg.solve(H.T, w[0] * int_lp(p1, t1, ContactPosition + di * delta[0]).T).T
    for j in range(1, len(delta)):
        V += np.linalg.solve(H.T, w[j] * int_lp(p1, t1, ContactPosition + di * delta[j]).T).T
    V = V - np.mean(V, axis=1, keepdims=True)

    for i in range(len(surfaces)):
        print(f'Computing sensitivity map for {surfaces[i][:-1]}')
        
        surf_file = list((fol / 'surf').glob(f"*{surfaces[i]}midthickness.surf.gii"))
        if surf_file:
            g = nib.load(surf_file[0])
            if (g.darrays[0].data).shape[0] > (g.darrays[1].data).shape[0]:
                ts = g.darrays[0].data
                ps = g.darrays[1].data
            else:
                ps = g.darrays[0].data
                ts = g.darrays[1].data
            ps = ps.astype(np.float64)
            U = w[0] * int_lp(ps, ts, ContactPosition + di * delta[0])
            
            for j in range(1, len(delta)):
                U += w[j] * int_lp(ps, ts, ContactPosition + di * delta[j])
            
            ContactSensitivityMap = (U - bem_fsl(ps, ts, p1, V)) / (4*np.pi*cond_brain)
            
            ts+=1  # Convert to 1-based indexing
            # Save results
            results = {
                'ContactName': ContactName,
                'ContactPosition': ContactPosition,
                'ContactOutsideBrain': ContactOutsideBrain,
                'ContactSensitivityMap': ContactSensitivityMap,
                'Vertices': ps,
                'Faces': ts}
            savemat(
                fol / 'model' / f"{sub}_{se}leadfield_hemi-{surfaces[i]}midthickness.mat",
                results)


def ComputeFeatureMaps(fol, feature, ChanTresh, GlobalTresh):
    """
    This function assembles the cortical maps for the features of interest. The feature associated 
    to each intracranial channel is projected to the cortical surface based on the sensitivity 
    profiles. To define the cortical region associated to each channel, the sensitivity maps are
    thresholded using a channel-specific threshold defined as a fraction ChanTresh of the maximum 
    sensitivity of the channel, and a global threshold GlobalTresh applied to all channels. The 
    feature value at each vertex is then defined as the value of the channel with maximum 
    sensitivity at that vertex.
    """

    leadfield_files = list((fol / 'model').glob(f"*leadfield*.mat"))
    leadfield_data = loadmat(str(leadfield_files[0]))
    ContactName = leadfield_data['ContactName']
    ContactOutsideBrain = leadfield_data['ContactOutsideBrain']

    if not feature.startswith("*"):
        feature = "*" + feature
    feature_files = [f for f in glob.glob(os.path.join(fol, "feat", feature))]
    for nf, feature_file in enumerate(feature_files):
        T = pd.read_csv(feature_file, sep='\t')
        ChannelName = T['ChannelName'].tolist()
        FeatureName = T.columns[1:].tolist()
        Data = T.iloc[:, 1:].values    
        Data=Data.astype(float)

        M = np.zeros((len(ChannelName), len(ContactName)))
        ChannelOutsideBrain = np.full(len(ChannelName), False)
        for i, channel in enumerate(ChannelName):
            if '-' in channel:
                # Bipolar channels
                parts = channel.split('-')
                id1 = [j for j, name in enumerate(ContactName) if name.rstrip().lower() == parts[0].lower()]
                id2 = [j for j, name in enumerate(ContactName) if name.rstrip().lower() == parts[1].lower()]
            
                if id1:
                    M[i, id1[0]] = 1
                    ChannelOutsideBrain[i] = ContactOutsideBrain[0,id1[0]]
                    if ContactOutsideBrain[0,id1[0]]:
                        M[i, id1[0]] = 0
                if id2:
                    M[i, id2[0]] = -1
                    ChannelOutsideBrain[i] = ChannelOutsideBrain[i] and ContactOutsideBrain[0,id2[0]]
                    if ContactOutsideBrain[0,id2[0]]:
                        M[i, id2[0]] = 0
            else:
                # Referential channels
                id = [j for j, name in enumerate(ContactName) if name.lower() == channel.lower()]
                if id:
                    M[i, id[0]] = 1
                    ChannelOutsideBrain[i] = ContactOutsideBrain[0,id[0]]
                    if ContactOutsideBrain[0,id[0]]:
                        M[i, id[0]] = 0

        D=Data[~ChannelOutsideBrain,:]
        Data[ChannelOutsideBrain, :] = np.nan

        for i, leadfield_file in enumerate(leadfield_files):
            surface = str(leadfield_file.name).split('leadfield')[1]
            print(f'Computing feature map {Path(feature_file).stem}{surface}')
        
            leadfield_data = loadmat(str(leadfield_file))
            ContactSensitivityMap = leadfield_data['ContactSensitivityMap']
            Vertices = leadfield_data['Vertices']
            Faces = leadfield_data['Faces']
        
            areas = areasv(Vertices, Faces-1)
            ChannelSens = M @ ContactSensitivityMap
            so = np.sort(np.abs(ChannelSens) / areas.T, axis=1)[:, ::-1]
            # Apply thresholds
            for j in range(len(ChannelName)):
                mask = np.abs(ChannelSens[j, :]) < so[j, 1] * areas * ChanTresh
                ChannelSens[j, mask] = 0
                mask = np.abs(ChannelSens[j, :]) < areas * GlobalTresh
                ChannelSens[j, mask] = 0
            # Linear interpolation
            CSens= np.abs(ChannelSens[~ChannelOutsideBrain,:]) 
            Intr = np.full((CSens.shape[1], D.shape[0]), 0.0)
            sm=np.sum(CSens, axis=0)
            for j in range(len(sm)):
                if sm[j] > 0:
                    Intr[j,:]=CSens[:, j].T /sm[j]
            FeatureMap = Intr @ D
            mask = np.sum(Intr, axis=1) == 0
            FeatureMap[mask, :] = np.nan

            # out_path = fol / 'maps' / f"{Path(feature_file).stem}{surface}" '_smooth.mat'
            # savemat(out_path, {
            # "FeatureMap": FeatureMap,
            # "FeatureName": FeatureName,
            # "Faces": Faces,
            # "Vertices": Vertices,
            # "FeatureValue": Data,
            # "ChannelName": ChannelName,
            # "ChannelOutsideBrain": ChannelOutsideBrain})        

            # Save results as GIFTI surface metric file(s)
            # Each column in FeatureMap corresponds to one feature (FeatureName[j])
            gif = nib.gifti.GiftiImage()
            for j, fname in enumerate(FeatureName):
                data = np.asarray(FeatureMap[:, j], dtype=np.float32)
                # Create a GiftiDataArray for this feature and attach metadata
                da = nib.gifti.GiftiDataArray(data=data, 
                    intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                    datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                    meta={"Feature": str(fname)})
                gif.add_gifti_data_array(da)
            out_path = fol / 'maps' / f"{Path(feature_file).stem}{surface}"
            out_path =  str(out_path)[:-4] + "_smooth.gii"
            nib.save(gif, out_path)

            # Nearest neighbor interpolation
            ref_row = np.full((1, ChannelSens.shape[1]), 
                             np.min(areas) * GlobalTresh / 100000)
            ChannelSens = np.vstack([ChannelSens, ref_row])
            ChannelMap = np.argmax(np.abs(ChannelSens), axis=0)
        
            # Create feature map
            FeatureMap = np.full((len(ChannelMap), Data.shape[1]), np.nan)
            Data_padded = np.vstack([Data, np.full((1, Data.shape[1]), np.nan)])
            for j in range(Data.shape[1]):
                FeatureMap[:, j] = Data_padded[ChannelMap, j]

            out_path = fol / 'maps' / f"{Path(feature_file).stem}{surface}"
            # savemat(out_path, {
            # "FeatureMap": FeatureMap,
            # "FeatureName": FeatureName,
            # "Faces": Faces,
            # "Vertices": Vertices,
            # "FeatureValue": Data,
            # "ChannelName": ChannelName,
            # "ChannelOutsideBrain": ChannelOutsideBrain})        

            # Save results as GIFTI surface metric file(s)
            # Each column in FeatureMap corresponds to one feature (FeatureName[j])
            gif = nib.gifti.GiftiImage()
            for j, fname in enumerate(FeatureName):
                data = np.asarray(FeatureMap[:, j], dtype=np.float32)
                # Create a GiftiDataArray for this feature and attach metadata
                da = nib.gifti.GiftiDataArray(data=data, 
                    intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                    datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                    meta={"Feature": str(fname)})
                gif.add_gifti_data_array(da)
            out_path =  str(out_path)[:-4] + ".gii"
            nib.save(gif, out_path)


def ContactProperties(fol, sub, ContactPosition, ContactLength):

    """
    This function computes properties of the iEEG contacts including their orientation
    and whether they are located inside or outside the brain. Integration points are also defined
    for each contact along its main axis. The brain boundary is derived from the brain mask and a
    tessellated surface is created using spherical harmonics fitting.
    """

    di = ContactPosition * 0
    np.seterr(invalid='ignore') 
    np.seterr(divide='ignore')
    # Determine contact orientations based on contacts on a line
    for i in range(ContactPosition.shape[0]):
        p = ContactPosition - ContactPosition[i]
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        # Compute distance matrix
        px=p[:,0][:, np.newaxis] / p_norm
        py=p[:,1][:, np.newaxis] / p_norm
        pz=p[:,2][:, np.newaxis] / p_norm
        d = (px - px.T)**2 + (py - py.T)**2 + (pz - pz.T)**2
        d += np.eye(d.shape[0]) * 2
        d[i,:] = 2
        d[:,i] = 2 
        i1, _ = np.where(d < 1e-6)
        u1, counts = np.unique(i1, return_counts=True)
        u1 = u1[counts > 1]
        di[i, :] = np.mean(p[u1, :], axis=0)   
        
        indices = np.concatenate(([i], u1))
        pei = ContactPosition[indices, :]
        dpe = np.linalg.norm(pei, axis=1)
        ima = np.argmax(dpe)
        imi = np.argmin(dpe)
        di[i, :] = pei[ima, :] - pei[imi, :]
        norm_di = np.linalg.norm(di[i])
        di[i] = di[i] / norm_di

    # Integration points per contact
    delta = np.array([-0.90618, -0.538469, 0, 0.538469, 0.90618]) * ContactLength
    # Determine which contacts are inside the brain
    brain_files = list((fol / 'anat').glob(f"{sub}*_space-nativepro_T1w_brain.nii*"))
    brain_img = nib.load(str(brain_files[0]))
    im0_img = brain_img.get_fdata() > 0
    # Get boundary points
    ins = im0_img[::2, ::2, ::2]
    ins2 = ndi.binary_erosion(ins, structure=ndi.generate_binary_structure(3, 1), border_value=1)  
    boundary = ins & ~ins2
    coords = np.array(np.where(boundary)).T   
    # Transform to world coordinates
    pixdim = brain_img.header.get_zooms()
    qoffset = brain_img.affine[:3, 3]
    x_coords = 2 * pixdim[0] * coords[:, 0] + qoffset[0]
    y_coords = 2 * pixdim[1] * coords[:, 1] + qoffset[1]
    z_coords = 2 * pixdim[2] * coords[:, 2] + qoffset[2]
    boundary_points = np.column_stack([x_coords, y_coords, z_coords])
    c = np.median(boundary_points, axis=0)
    # Spherical harmonics fitting  
    ps = boundary_points - c
    rs = np.linalg.norm(ps, axis=1)
    rs[rs < 50] = 50
    coef = np.linalg.lstsq(armesf(ps, 30), rs)[0]
    p, t1 = esfera(1, 4)
    p = p[:, [1, 2, 0]] 
    r = armesf(p, 30) @ coef
    r[r < 50] = 50
    p1 = p * r[:, np.newaxis] + c   
    # Determine contacts outside brain
    result = int_lp(p1, t1, ContactPosition + di * np.min(delta))
    ContactOutsideBrain = np.sum(result, 1) > -0.1
    return p1, t1, di, ContactOutsideBrain


def GetContactPositions(fol, sub, se, electrode_file, tx_file, ContactSize):
    """
    Determines the position of the iEEG contacts in MICAs nativepro space.
    """

    # Read contact positions and names
    T = pd.read_csv(electrode_file, sep='\t')
    ContactName = T['name'].tolist()
    # Read transformation file
    tx_data = loadmat(str(tx_file))
    x = T[['x', 'y', 'z']].values
    # Determine units
    coord_file = Path(electrode_file).parent / (Path(electrode_file).stem[:-10] + 'coordinate_system.json')
    done = False
    if coord_file.exists():
        with open(coord_file, 'r') as f:
            coord_data = json.load(f)  
        if 'iEEGCoordinateUnits' in coord_data:
            units = coord_data['iEEGCoordinateUnits']
            if units == 'cm':
                x = x * 10
                done = True
            elif units == 'm':
                x = x * 1000
                done = True
    if not done:
        # Estimate units from inter-electrode distances
        distances = squareform(pdist(x))
        np.fill_diagonal(distances, 300)  # Large diagonal values
        dmin = np.median(np.min(distances, axis=1))
        if dmin < 0.0105:
            x = x * 1000
        elif dmin < 1.05:
            x = x * 10
    # Transform coordinates
    A = tx_data['AffineTransform_double_3_3']
    f = tx_data['fixed']
    Am = np.array([[A[0,0], A[3,0], A[6,0]],
                   [A[1,0], A[4,0], A[7,0]],
                   [A[2,0], A[5,0], A[8,0]]])
    t=f-np.linalg.solve(Am, f)
    offset = np.array([-A[9]-t[0], -A[10]-t[1], A[11]+t[2]])
    transform_matrix = np.array([[A[0,0], A[1,0], -A[2,0]],
                                 [A[3,0], A[4,0], -A[5,0]],
                                 [-A[6,0], -A[7,0], A[8,0]]])
    ContactPosition = np.dot(x - offset.T, transform_matrix)
    ContactPosition = np.round(ContactPosition * 100) / 100
    # Write electrodes.tsv
    electrode_df = pd.DataFrame({
        'name': ContactName,
        'x': ContactPosition[:, 0].tolist(),
        'y': ContactPosition[:, 1].tolist(), 
        'z': ContactPosition[:, 2].tolist(),
        'size': [ContactSize] * len(ContactName)})
    electrode_df.to_csv(
        fol / 'anat' / f"{sub}_{se}space-nativepro_T1w_electrodes.tsv",
        sep='\t', index=False)
    # Write coordinate system JSON
    sed = se if se else ""
    if sed:
        sed = sed[:-1] + "/"  # Replace _ with /
    coord_json = {
        "IntendedFor": f"bids::{sub}/{sed}anat/{sub}_{se}space-nativepro_iEEG.nii.gz",
        "iEEGCoordinateSystem": "ACPC",
        "iEEGCoordinateUnits": "mm"}
    with open(fol / 'anat' / f"{sub}_{se}space-nativepro_T1w_coordsystem.json", 'w') as f:
        json.dump(coord_json, f, indent=4)
        return ContactName, ContactPosition


def solve_inverse_problem(fol, sub, se, feature):

    """
    This function computes the source imaging results for the features of interest. A variation of 
    eLORETA is used to estiamte the inverse problem solution. The elements of the diagonal of the
    weight matrix are computed as in eLORETA (Pascual-Marqui (2007) Discrete, 3D distributed, 
    linear imaging methods of electric neuronal activity. Part 1: exact, zero error localization.
    http://arxiv.org/pdf/0710.3341 ), but non-zero off-diagonal elements are included for
    the neighboring nodes, with value decreassing with distance. The non-diagonal weight matrix 
    leads to a loss of the zero-error localization property of eLORETA for point sources, but it
    produces smoother source estimates which may be more appropriate for distributed sources.
    Five different solutions with different levels of regularization are computed, corresponding to
    signal-to-noise ratios from very low to very high, in steps of 7 dB (five times).
    """

    # -------------------------------------------------------
    # Load and concatenate leadfields (Left hemisphere)
    # -------------------------------------------------------
    mat = loadmat(
        os.path.join(fol, "model", 
                     f"{sub}_{se}hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness_leadfield.mat"),
        simplify_cells=True
    )
    L = mat["U"]
    dd = mat["d"]
    ii1 = mat["i1"]
    ii2 = mat["i2"]
    psl = mat["ps"]
    tsl = mat["ts"]
    pe = mat["pe"]

    # Right hemisphere
    mat = loadmat(
        os.path.join(fol, "model", 
                     f"{sub}_{se}hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness_leadfield.mat"),
        simplify_cells=True
    )
    L = np.concatenate((L, mat["U"]), axis=1)
    dd = np.concatenate((dd, mat["d"]), axis=0)
    ii1 = np.concatenate((ii1, mat["i1"] + len(psl)), axis=0)
    ii2 = np.concatenate((ii2, mat["i2"] + len(psl)), axis=0)
    psr = mat["ps"]
    tsr = mat["ts"]
    ContactName = mat["ContactName"]

    # -------------------------------------------------------
    # Optional Hippocampus Leadfields
    # -------------------------------------------------------
    include_hipp = 2
    lh_hipp = os.path.join(
        fol, "model", f"{sub}_{se}hemi-L_space-T1w_den-2mm_label-hipp_midthickness_leadfield.mat"
    )
    if not os.path.exists(lh_hipp):
        lh_hipp = os.path.join(
            fol, "model", f"{sub}_{se}hemi-L_space-T1w_den-8k_label-hipp_midthickness_leadfield.mat"
        )
    rh_hipp = os.path.join(
        fol, "model", f"{sub}_{se}hemi-R_space-T1w_den-2mm_label-hipp_midthickness_leadfield.mat"
    )
    if not os.path.exists(rh_hipp):
        rh_hipp = os.path.join(
            fol, "model", f"{sub}_{se}hemi-R_space-T1w_den-8k_label-hipp_midthickness_leadfield.mat"
        )
        include_hipp=8

    if os.path.exists(lh_hipp) and os.path.exists(rh_hipp):

        mat = loadmat(lh_hipp, simplify_cells=True)
        L = np.concatenate((L, mat["U"]), axis=1)
        dd = np.concatenate((dd, mat["d"]), axis=0)
        ii1 = np.concatenate(
            (ii1, mat["i1"] + len(psl) + len(psr)), axis=0
        )
        ii2 = np.concatenate(
            (ii2, mat["i2"] + len(psl) + len(psr)), axis=0
        )
        phl = mat["ps"]
        thl = mat["ts"]

        mat = loadmat(rh_hipp, simplify_cells=True)
        L = np.concatenate((L, mat["U"]), axis=1)
        dd = np.concatenate((dd, mat["d"]), axis=0)
        ii1 = np.concatenate(
            (ii1, mat["i1"] + len(psl) + len(psr) + len(phl)), axis=0
        )
        ii2 = np.concatenate(
            (ii2, mat["i2"] + len(psl) + len(psr) + len(phl)), axis=0
        )
        phr = mat["ps"]
        thr = mat["ts"]
    else:
        include_hipp = 0
        

    # -------------------------------------------------------
    # Locate feature files
    # -------------------------------------------------------
    if not feature.startswith("*"):
        feature = "*" + feature

    all_files = [f for f in glob.glob(os.path.join(os.path.join(fol, "feat"), feature))]

    bem = loadmat(os.path.join(fol, "model", f"{sub}_{se}BEMsys.mat"), simplify_cells=True)
    no_electrode_file = bem["no_electrode_file"]

    print("Computing source imaging results...")

    # -------------------------------------------------------
    # Process each feature file
    # -------------------------------------------------------
    for fname in all_files:
        print(fname)

        T = pd.read_table(os.path.join(fol, "feat", fname), sep=None, engine="python")
        ChannelName = T["ChannelName"].astype(str).tolist()
        FeatureName = list(T.columns[1:])
        Data = T.drop(columns=["ChannelName"]).to_numpy()

        # Map channel names to indices
        ind = np.zeros(len(ChannelName), dtype=int)

        for i, ch in enumerate(ChannelName):
            matches = [k for k, v in enumerate(ContactName) if v.lower().rstrip() == ch.lower()]
            if matches:
                ind[i] = matches[0]
            else:
                # Handle fallback electrode naming
                if no_electrode_file:
                    fallback = dict(T5=63, T3=41, T4=49, T6=71)
                    if ch.upper() in fallback:
                        ind[i] = fallback[ch.upper()]
                        continue
                print(f"Unknown electrode {ch}. Ignoring.")
                Data[i, :] = np.nan
                ind[i] = -1

        # keep only valid channels
        keep = ~np.any(np.isnan(Data), axis=1)
        Data = Data[keep, :]
        ind = ind[keep]

        U = L[ind, :]
        pe_sub = pe[ind, :]
        ChannelName = [ContactName[i] for i in ind]

        Ne, Nj = U.shape

        # Centering matrix
        H = np.eye(Ne) - np.ones((Ne, Ne)) / Ne
        U = H @ U
        Data = H @ Data

        a = -np.median(dd) / np.log(0.995)
        alpha = np.array([0.008, 0.04, 0.2, 1, 5])

        SourceImaging = np.zeros((Nj, Data.shape[1], len(alpha)))

        w = np.ones(Nj)

        # -------------------------------------------------------
        # Main iterative solver
        # -------------------------------------------------------
        for li, alp in enumerate(alpha):

            for _ in range(30):
                M = U @ sp.spdiags(1 / w, 0, Nj, Nj) @ U.T + alp * H
                wn = np.array([np.sqrt(U[:, j].T @ M @ U[:, j]) for j in range(Nj)])
                w = wn

            W = sp.csr_matrix(
                (np.exp(-dd / a) * np.sqrt(1 / (w[ii1] * w[ii2])), (ii1, ii2)),
                shape=(Nj, Nj)
            ) + sp.spdiags(1 / w, 0, Nj, Nj)

            V = W @ U.T
            M = U @ V + alp * H
            K = V @ np.linalg.inv(M + 2 * np.pi / Nj * np.eye(M.shape[0]))

            SourceImaging[:, :, li] = K @ Data

        # -------------------------------------------------------
        # Save separate maps
        # -------------------------------------------------------
        tsl +=1
        tsr +=1

        _,base = os.path.split(fname)
        base=base[:-4]
        
        # Left hemisphere    
        # savemat(os.path.join(
        #    fol, "maps", base+"_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness.mat"
        # ), {
        #     "FeatureMap": SourceImaging[:len(psl), :, :],
        #     "FeatureName": FeatureName,
        #     "alpha": alpha,
        #     "Faces": tsl,
        #     "Vertices": psl,
        #     "FeatureValue": Data,
        #     "ChannelPos": pe_sub        
        # })
        
        # Save results as GIFTI surface metric file(s)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[:len(psl), :, 4], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness_VeryLowSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[:len(psl), :, 3], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness_LowSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[:len(psl), :, 2], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness_MediumSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[:len(psl), :, 1], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness_HighSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[:len(psl), :, 0], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness_VeryHighSNR.gii")
        nib.save(gif, out_path)


        # Right hemisphere
        # savemat(os.path.join(
        #     fol, "maps", base+"_hemi-Rspace-nativepro_surf-fsLR-32k_label-midthickness.mat"
        # ), {
        #     "FeatureMap": SourceImaging[len(psl):len(psl)+len(psr), :, :],
        #     "ChannelName": ChannelName,
        #     "alpha": alpha,
        #     "Faces": tsr,
        #     "Vertices": psr,
        #     "FeatureValue": Data,
        #     "ChannelPos": pe_sub
        # })

        # Save results as GIFTI surface metric file(s)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[len(psl):len(psl)+len(psr), :, 4], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness_VeryLowSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[len(psl):len(psl)+len(psr), :, 3], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness_LowSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[len(psl):len(psl)+len(psr), :, 2], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness_MediumSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[len(psl):len(psl)+len(psr), :, 1], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness_HighSNR.gii")
        nib.save(gif, out_path)
        gif = nib.gifti.GiftiImage()
        data = np.asarray(SourceImaging[len(psl):len(psl)+len(psr), :, 0], dtype=np.float32)
        da = nib.gifti.GiftiDataArray(data=data, 
            intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
            datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
            meta={"Feature": str(fname)})
        gif.add_gifti_data_array(da)
        out_path =  os.path.join(fol, "maps", base+"_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness_VeryHighSNR.gii")
        nib.save(gif, out_path)


        # Hippocampus optional
        if include_hipp > 0:
            start = len(psl) + len(psr)
            thl +=1
            thr +=1
            hte='2mm'
            if include_hipp == 8:
                hte='8k'
            # Save results as GIFTI surface metric file(s)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start:start+len(phl), :, 4], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-L_space-T1w_den-"+hte+"_label-hipp_midthickness_VeryLowSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start:start+len(phl), :, 3], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-L_space-T1w_den-"+hte+"_label-hipp_midthickness_LowSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start:start+len(phl), :, 2], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-L_space-T1w_den-"+hte+"_label-hipp_midthickness_MediumSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start:start+len(phl), :, 1], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-L_space-T1w_den-"+hte+"_label-hipp_midthickness_HighSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start:start+len(phl), :, 0], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-L_space-T1w_den-"+hte+"_label-hipp_midthickness_VeryHighSNR.gii")
            nib.save(gif, out_path)

            # savemat(os.path.join(
            #     fol, "maps", base+"_hemi-L_space-T1w_den-2mm_label-hipp_midthickness.mat"
            # ), {
            #     "FeatureMap": SourceImaging[start:start+len(phl), :, :],
            #     "FeatureName": FeatureName,
            #     "ChannelName": ChannelName,
            #     "alpha": alpha,
            #     "Faces": thl,
            #     "Vertices": phl,
            #     "FeatureValue": Data,
            #     "ChannelPos": pe_sub
            # })

            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start+len(phl):start+len(phl)+len(phr), :, 4], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-R_space-T1w_den-"+hte+"_label-hipp_midthickness_VeryLowSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start+len(phl):start+len(phl)+len(phr), :, 3], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-R_space-T1w_den-"+hte+"_label-hipp_midthickness_LowSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start+len(phl):start+len(phl)+len(phr), :, 2], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-R_space-T1w_den-"+hte+"_label-hipp_midthickness_MediumSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start+len(phl):start+len(phl)+len(phr), :, 1], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-R_space-T1w_den-"+hte+"_label-hipp_midthickness_HighSNR.gii")
            nib.save(gif, out_path)
            gif = nib.gifti.GiftiImage()
            data = np.asarray(SourceImaging[start+len(phl):start+len(phl)+len(phr), :, 0], dtype=np.float32)
            da = nib.gifti.GiftiDataArray(data=data, 
                intent=nib.nifti1.intent_codes['NIFTI_INTENT_NONE'],
                datatype=nib.nifti1.data_type_codes['NIFTI_TYPE_FLOAT32'],
                meta={"Feature": str(fname)})
            gif.add_gifti_data_array(da)
            out_path =  os.path.join(
                fol, "maps", base+"_hemi-R_space-T1w_den-"+hte+"_label-hipp_midthickness_VeryHighSNR.gii")
            nib.save(gif, out_path)

            # savemat(os.path.join(
            #     fol, "maps", base+"_hemi-R_space-T1w_den-2mm_label-hipp_midthickness.mat"
            # ), {
            #     "FeatureMap": SourceImaging[start+len(phl):start+len(phl)+len(phr), :, :],
            #     "FeatureName": FeatureName,
            #     "ChannelName": ChannelName,
            #     "alpha": alpha,
            #     "Faces": thr,
            #     "Vertices": phr,
            #     "FeatureValue": Data,
            #     "ChannelPos": pe_sub
            # })


def place_electrodes(fol, sub, se, electrode_file, c3, pes, rs):
    """
    This function determines the position of the electrodes on the scalp. If an electrode tsv file
    is provided, it uses the coordinates in that file. In order to achieve this, the coordinates of
    the anatomical landmarks nasion and left/right pre-auricular points must be defined in either
    the electrodes.tsv file or the CoordianteSystem.json file. Otherwise, it only uses elctrodes 
    that are part of the standard 10-10 system (and the positions on the scalp are computed by 
    applying the non-linear transformation between the nativepor space and MNI space). Also, an
    electrode and coordimate system files are created in MICAs nativepro space.   
        pe : (Nx3) electrode positions on scalp
        ContactName : list of electrode names
    """

    # --------------------------------------------------------------------------------
    # 10–20 template electrode coordinates
    # --------------------------------------------------------------------------------
    elec = np.array([
        [-86.1, -20, -48], [85.8, -20, -48], [0, 86.8, -40], [-29.4, 83.9, -7],
        [0.1, 88.2, -1.7], [29.9, 84.9, -7.1], [-49, 64.1, -47.7], [-54.8, 68.6, -10.6],
        [-45.4, 72.9, 6], [-33.7, 76.8, 21.2], [-18.5, 79.9, 32.8], [0.2, 80.8, 35.4],
        [19.8, 80., 32.8], [35.7, 77.7, 22], [46.6, 73.8, 6], [55.7, 69.7, -10.8],
        [50.4, 63.9, -48], [-70.1, 41.7, -50], [-70.3, 42.5, -11.4], [-64.5, 48, 16.9],
        [-50.2, 53.1, 42.2], [-27.5, 56.9, 60.3], [0.3, 58.5, 66.5], [29.5, 57.6, 59.5],
        [51.8, 54.3, 40.8], [67.9, 49.8, 16.4], [73, 44.4, -12], [72.1, 42.1, -50.5],
        [-84.1, 14.6, -50.4], [-80.8, 14.1, -11.1], [-77.2, 18.6, 24.5], [-60.2, 22.7, 55.5],
        [-34.1, 26, 80], [0.4, 27.4, 88.7], [34.8, 26.4, 78.8], [62.3, 23.7, 55.6],
        [79.5, 19.9, 24.4], [81.8, 15.4, -11.3], [84.1, 14.4, -50.5], [-85.9, -15.8, -48.3],
        [-84.2, -16, -9.3], [-80.3, -13.8, 29.2], [-65.4, -11.6, 64.4], [-36.2, -10, 89.8],
        [0.4, -9.2, 100.2], [37.7, -9.6, 88.4], [67.1, -10.9, 63.6], [83.5, -12.8, 29.2],
        [85.2, -15, -9.5], [85.6, -16.4, -48.3], [-85.6, -46.5, -45.7], [-84.8, -46, -7.1],
        [-79.6, -46.6, 30.9], [-63.6, -47, 65.6], [-35.5, -47.9, 91.3], [0.4, -47.3, 99.4],
        [38.4, -47.1, 90.7], [66.6, -46.6, 65.6], [83.3, -46.1, 31.2], [85.5, -45.5, -7.1],
        [86.2, -47, -45.9], [-73, -73.8, -41], [-72.4, -73.5, -2.5], [-67.3, -76.3, 28.4],
        [-53, -78.8, 55.9], [-28.6, -80.5, 75.4], [0.3, -81.1, 82.6], [31.9, -80.5, 76.7],
        [55.7, -78.6, 56.6], [67.9, -75.9, 28.1], [73, -73.1, -2.5], [73.9, -74.4, -41.2],
        [-54.9, -98, -35.5], [-54.9, -97.5, 2.8], [-48.4, -99.3, 21.6], [-36.5, -100.9, 37.2],
        [-19, -101.8, 46.5], [0.2, -102.2, 50.6], [19.9, -101.8, 46.4], [36.8, -100.8, 36.4],
        [49.8, -99.4, 21.7], [55.7, -97.6, 2.7], [55, -98.1, -35.5], [-29.4, -112.4, 8.8],
        [0.1, -114.9, 14.7], [29.8, -112.2, 8.8], [-29.8, -114.6, -29.2], [0, -118.6, -23.1],
        [29.7, -114.3, -29.3]
    ])

    ContactName = [
        'LPA','RPA','Nz','Fp1','Fpz','Fp2','AF9','AF7','AF5','AF3','AF1','AFz',
        'AF2','AF4','AF6','AF8','AF10','F9','F7','F5','F3','F1','Fz','F2','F4','F6','F8',
        'F10','FT9','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','FT10','T9','T7',
        'C5','C3','C1','Cz','C2','C4','C6','T8','T10','TP9','TP7','CP5','CP3','CP1','CPz',
        'CP2','CP4','CP6','TP8','TP10','P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8',
        'P10','PO9','PO7','PO5','PO3','PO1','POz','PO2','PO4','PO6','PO8','PO10','O1','Oz',
        'O2','I1','Iz','I2'
    ]

    # --------------------------------------------------------------------------------
    # Apply ANTs transforms to canonical electrode set
    # --------------------------------------------------------------------------------
    transformlist = [
        os.path.join(fol, "xfm", f"{sub}_{se}from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_0GenericAffine.mat")
    ]
    warp = glob.glob(os.path.join(fol, "xfm",
                                  f"{sub}_{se}from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_1Warp.nii*"))
    transformlist.append(warp[0])

    points = np.column_stack([-elec[:, 0], -elec[:, 1], elec[:, 2]])
    pet = ants.apply_transforms_to_points(
        dim=3,
        transformlist=transformlist,
        points=pd.DataFrame(points, columns=['x', 'y', 'z']),
        whichtoinvert=[False, False]
    )
    pet=pet.to_numpy()
    pet = np.column_stack([-pet[:, 0], -pet[:, 1], pet[:, 2]])

    # --------------------------------------------------------------------------------
    # Project to scalp surface
    # --------------------------------------------------------------------------------
    pe = pet - c3
    pe = pe / np.linalg.norm(pe, axis=1, keepdims=True)

    r = np.zeros(pe.shape[0])
    for i in range(len(r)):
        r[i] = np.mean(rs[pes @ pe[i, :] > 0.9962])

    pe = pe * r[:, None] + c3

    print(f"Average distance to scalp: {np.mean(np.linalg.norm(pe - pet, axis=1))}")
    print(f"Max distance to scalp: {np.max(np.linalg.norm(pe - pet, axis=1))}")

    # Normal vector for hemisphere exclusion
    v = pe[0:2, :] - pe[2][None, :]
    nz = pe[2, :]

    dv = np.cross(v[0], v[1])
    dv = dv / np.linalg.norm(dv)

    # --------------------------------------------------------------------------------
    # If electrodes.tsv exists → use its positions
    # --------------------------------------------------------------------------------
    no_electrode_file = electrode_file is None or electrode_file == ""

    if not no_electrode_file:

        EEG_folder = os.path.dirname(electrode_file)
        T = pd.read_csv(electrode_file, sep='\t')

        ContactName = list(T['name'])
        pos = T[['x', 'y', 'z']].to_numpy()

        if np.any(pos != 0):   # The file contains electrode coordinates
            # Find fiducials
            def find(ch):
                chs = [i for i, n in enumerate(ContactName) if n.lower() == ch.lower()]
                return chs[0] if chs else None

            lpa = find('LPA') or find('T9')
            rpa = find('RPA') or find('T10')
            nas = find('NAS') or find('Nz')

            if lpa is None or rpa is None or nas is None:
                # Try JSON metadata
                jsons = []

                # coordinate_system.json
                csys = glob.glob(electrode_file[:-14] + 'coordinate_system.json')
                for f in csys:
                    jsons.append(f)

                # IntendedFor files
                for f in jsons:
                    js = json.load(open(f))
                    if 'IntendedFor' in js:
                        intended = js['IntendedFor']
                        if isinstance(intended, str):
                            intended = [intended]
                        for item in intended:
                            base = os.path.basename(item)
                            guess = os.path.join(EEG_folder, base)
                            jsons.append(guess)

                # T1w json
                jsons += glob.glob(os.path.join(EEG_folder, "*_T1w.json"))

                done = False
                for jf in jsons:
                    try:
                        js = json.load(open(jf))
                        if 'FiducialsCoordinates' in js:
                            fid = js['FiducialsCoordinates']
                            clpa = fid.get('LPA', None)
                            crpa = fid.get('RPA', None)
                            cnas = fid.get('NAS', None)
                            if clpa and crpa and cnas:
                                f = np.vstack([clpa, crpa, cnas])
                                done = True
                                break
                    except:
                        pass

                if not done:
                    print("Unknown position of anatomical landmarks...")
                    print("...attempting to continue with only 10-10 electrodes")
                    no_electrode_file = True
            else:
                f = pos[[lpa, rpa, nas], :]

    # --------------------------------------------------------------------------------
    # Align electrodes to scalp using fiducials if available
    # --------------------------------------------------------------------------------
    if not no_electrode_file:
        F = pe[:3, :]
        tF = np.mean(F, axis=0)
        F = F - tF

        tf = np.mean(f, axis=0)
        f = f - tf

        # scale to mm
        scales = np.sqrt(np.sum(F**2, axis=1) / np.sum(f**2, axis=1))
        f = f * (10**np.round(np.log10(scales)))

        s = np.linalg.norm(F) / np.linalg.norm(f)

        M = f.T @ F
        R = M @ np.linalg.inv((M.T @ M) ** 0.5)

        pec = s * pos @ R

        # Project to scalp
        pe = pec - c3
        pe = pe / np.linalg.norm(pe, axis=1, keepdims=True)

        r = np.zeros(pe.shape[0])
        for i in range(len(r)):
            r[i] = np.mean(rs[pes @ pe[i, :] > 0.9962])

        pe = pe * r[:, None] + c3

        print(f"Scale factor: {s}")
        print(f"Average distance to scalp: {np.mean(np.linalg.norm(pe - pec, axis=1))}")
        print(f"Max distance to scalp: {np.max(np.linalg.norm(pe - pec, axis=1))}")

        # Reject electrodes too low
        del_idx = ((pe - nz) @ dv) < -15
        if np.any(del_idx):
            print("The following electrodes are too low and will not be included:")
            dropped = [ContactName[i] for i in np.where(del_idx)[0]]
            print(dropped)
            ContactName = [ContactName[i] for i in range(len(ContactName)) if not del_idx[i]]
            pe = pe[~del_idx]

    # --------------------------------------------------------------------------------
    # Write electrodes.tsv
    # --------------------------------------------------------------------------------
    pe = np.round(pe, 2)
    df = pd.DataFrame({
        "name": ContactName,
        "x": pe[:, 0],
        "y": pe[:, 1],
        "z": pe[:, 2]
    })

    out_tsv = os.path.join(fol, "anat", f"{sub}_{se}space-nativepro_T1w_electrodes.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    # --------------------------------------------------------------------------------
    # Write coordsystem JSON
    # --------------------------------------------------------------------------------
    sed = se[:-1] + "/" if se else ""
    nii = glob.glob(os.path.join(fol, "anat", f"{sub}*space-nativepro_T1w.nii*"))[0]

    js = {
        "IntendedFor": f"bids::{sub}{sed}/anat/{os.path.basename(nii)}",
        "iEEGCoordinateSystem": "ACPC",
        "iEEGCoordinateUnits": "mm"
    }

    out_json = os.path.join(
        fol, "anat", f"{sub}_{se}space-nativepro_T1w_coordsystem.json"
    )
    with open(out_json, "w") as f:
        json.dump(js, f, indent=4)

    return pe, ContactName


def compute_BEM_linear_system(
    fol, sub, se,
    p1, p2, p3,
    t1, t2, t3,
    pe, ContactName,
    no_electrode_file):
    
    """
    This function computes the linear system of the EEG forward model for a three-layer realistic-
    shape head model using the Boundary Element Method (BEM) with linear basis functions, employing
    the isolated skull approach (Meijs et al. (1989) On the numerical accuracy of the boundary 
    element method. IEEE Trans Biomed Eng. doi: 10.1109/10.40805) to deal with the large 
    conductivity difference between the brain and skull.

    """

    # --------------------------------------------------------------
    # Conductivities
    # --------------------------------------------------------------
    cond_brain = 0.33
    cond_skull = 0.015
    cond_scalp = 0.33

    # --------------------------------------------------------------
    # Convert mm → m
    # --------------------------------------------------------------
    pe = pe / 1000
    p1 = p1 / 1000
    p2 = p2 / 1000
    p3 = p3 / 1000

    # --------------------------------------------------------------
    # Build combined vertices & triangles
    # --------------------------------------------------------------
    P = np.vstack([p1, p2, p3])

    nt1 = t1.shape[0]
    nt2 = t2.shape[0]
    nt3 = t3.shape[0]

    nv1 = p1.shape[0]
    nv2 = p2.shape[0]
    nv3 = p3.shape[0]

    # Triangle indexing with offsets
    T = np.vstack([
        t1,
        t2 + nv1,
        t3 + nv1 + nv2
    ])

    nt = np.array([nt1, nt2, nt3])
    nv = np.array([nv1, nv2, nv3])

    # Conductivity contrasts
    dc = np.concatenate([
        (cond_brain - cond_skull) * np.ones(nt1),
        (cond_skull - cond_scalp) * np.ones(nt2),
        cond_scalp * np.ones(nt3)
    ])

    # Compute interpolation matrices
    e = bem_interp(p3 / np.sqrt(np.sum(p3 ** 2, axis=1))[:, None] / 10, t3, pe / np.sqrt(np.sum(pe ** 2, axis=1))[:, None] / 10)
    E = np.hstack([np.zeros((pe.shape[0], p1.shape[0] + p2.shape[0])), e])
    # Compute linear system using ISA
    H = int_lp(P, T, P, dc) 
    H = solve((H - sp.spdiags(np.sum(H, axis=1), 0, H.shape[0], H.shape[1]) + 2 * np.pi / H.shape[0]).T, E.T).T
    del E
    H = H - np.mean(H, axis=1, keepdims=True)
    H = H * cond_skull * (int_lp(p1, t1, P) - (2 * np.pi) * sp.eye(P.shape[0], nv1))
    M = int_lp(p1, t1, p1, cond_brain * np.ones(nt1))
    M = M - np.diag(M.sum(axis=1)) + (2 * np.pi / M.shape[0])
    H = solve(M.T, H.T).T
    del M
    H = H - np.mean(H, axis=1, keepdims=True)


    # --------------------------------------------------------------
    # Save (MATLAB-style)
    # --------------------------------------------------------------
    out_path = os.path.join(fol, "model", f"{sub}_{se}BEMsys.mat")

    savemat(out_path, {
        "pe": pe,
        "p1": p1, "t1": t1,
        "p2": p2, "t2": t2,
        "p3": p3, "t3": t3,
        "ContactName": ContactName,
        "H": H,
        "cond_brain": cond_brain,
        "cond_skull": cond_skull,
        "cond_scalp": cond_scalp,
        "no_electrode_file": no_electrode_file
    })

    return H


def compute_leadfield(fol, sub, se):
    """
    Computes the scalp EEG leadfield matrices for cortical (and possibly hippocampal) surfaces.
    It also computes adjacency and distance matrices of surface vertices that will be used in the
    inverse problem solution.  
    """

    # ------------------------------------------------------------
    # Surface name patterns
    # ------------------------------------------------------------
    surface = [
        'L_space-nativepro_surf-fsLR-32k_label-',
        'R_space-nativepro_surf-fsLR-32k_label-',
        'L_space-T1w_den-2mm_label-hipp_',
        'R_space-T1w_den-2mm_label-hipp_'
    ]

    # If hippocampal surfaces missing, drop them
    surf_L = os.path.join(
        fol, 'surf',
        f"{sub}_{se}hemi-{surface[2]}midthickness.surf.gii"
    )
    surf_R = os.path.join(
        fol, 'surf',
        f"{sub}_{se}hemi-{surface[3]}midthickness.surf.gii"
    )

    if not (os.path.exists(surf_L) and os.path.exists(surf_R)):
        surface = [
            'L_space-nativepro_surf-fsLR-32k_label-',
            'R_space-nativepro_surf-fsLR-32k_label-',
            'L_space-T1w_den-8k_label-hipp_',
            'R_space-T1w_den-8k_label-hipp_'
        ]

        # If hippocampal surfaces missing, drop them
        surf_L = os.path.join(
            fol, 'surf',
            f"{sub}_{se}hemi-{surface[2]}midthickness.surf.gii"
        )
        surf_R = os.path.join(
            fol, 'surf',
            f"{sub}_{se}hemi-{surface[3]}midthickness.surf.gii"
        )

    if not (os.path.exists(surf_L) and os.path.exists(surf_R)):
        surface = surface[:2]
        print("No hippocampal surface")

    # ------------------------------------------------------------
    # Load BEM system
    # ------------------------------------------------------------
    bem_path = os.path.join(
        fol, 'model', f"{sub}_{se}BEMsys.mat"
    )
    bem = loadmat(bem_path)

    pe = bem['pe']
    p1 = bem['p1']
    ContactName = bem['ContactName']
    H = bem['H']

    # ------------------------------------------------------------
    # Loop over surfaces
    # ------------------------------------------------------------
    for surf in surface:
        print(f"Computing leadfield for hemi-{surf}...")

        # --------------------------------------------------------
        # Load midthickness surface
        # --------------------------------------------------------
        surf_path = os.path.join(
            fol, 'surf',
            f"{sub}_{se}hemi-{surf}midthickness.surf.gii"
        )
        g = nib.load(surf_path)
        if (g.darrays[0].data).shape[0] > (g.darrays[1].data).shape[0]:
            ts = g.darrays[0].data
            ps = g.darrays[1].data
        else:
            ps = g.darrays[0].data
            ts = g.darrays[1].data
        ps = ps.astype(np.float64)/1000

        # --------------------------------------------------------
        # Compute leadfield U
        # --------------------------------------------------------
        U = bem_fsl(ps, ts, p1, H)

        # Remove column mean
        U = U - U.mean(axis=0, keepdims=True)

        Nj = U.shape[1]

        # --------------------------------------------------------
        # Compute adjacency i1, i2
        # --------------------------------------------------------
        i1_list = []
        i2_list = []

        # For each vertex ii, find neighbors via faces
        for ii in range(Nj):
            ix = np.any(ts == ii, axis=1)
            neighbor_faces = ts[ix, :]

            neighbors = np.unique(neighbor_faces.reshape(-1))
            neighbors = neighbors[neighbors != ii]

            # Append repeated ii values and neighbor list
            i1_list.append(np.full(len(neighbors), ii, dtype=int))
            i2_list.append(neighbors)

        i1 = np.concatenate(i1_list)
        i2 = np.concatenate(i2_list)

        # --------------------------------------------------------
        # Compute distances d
        # --------------------------------------------------------
        dvals = np.sqrt(((ps[i1] - ps[i2]) ** 2).sum(axis=1))

        # --------------------------------------------------------
        # Save result
        # --------------------------------------------------------
        out_path = os.path.join(
            fol, "model",
            f"{sub}_{se}hemi-{surf}midthickness_leadfield.mat"
        )

        savemat(out_path, {
            "U": U,
            "ps": ps,
            "ts": ts,
            "pe": pe,
            "ContactName": ContactName,
            "d": dvals,
            "i1": i1,
            "i2": i2
        })


def bem_interp(P, T, v):
    """
    A = bem_interp(P, T, v)
    
    This function computes linear interpolation matrices for scalp EEG contact positions from the
    neighboring triangle vertices of a surface mesh.
    
    P: The mesh node coordinates (Nx3)
    T: The triangle connectivity (Mx3, indices into P)
    v: The points to interpolate (Nx3)

    Returns:
    A: Linear interpolation matrix (ns x N)

    """
    ns = v.shape[0]  # Number of points
    n = T.shape[0]   # Number of triangles
    
    # Initialize sparse matrices
    A = np.zeros((ns, P.shape[0]))  # Matrix for linear interpolation

    for ii in range(ns):
        # Compute vectors for triangle edges and point projection
        v1 = P[T[:, 0], :] - P[T[:, 2], :]
        v2 = P[T[:, 1], :] - P[T[:, 2], :]
        vii=v[ii,:]
        vt = np.repeat(vii[None,:], n, axis=0) - P[T[:, 2], :]

        # Compute cross products
        v1xv2 = np.cross(v1, v2)
        A2 = np.linalg.norm(v1xv2, axis=1)
        
        # Project point onto the plane of the triangle
        vp = vt - np.sum(vt * v1xv2, axis=1)[:, None] / A2[:, None] ** 2 * v1xv2
        v1xvp = np.cross(v1, vp)
        vpxv2 = np.cross(vp, v2)

        # Further computations for triangle areas and projections
        v3 = v1 - v2
        vpp = vp - v2
        vppxv3 = np.cross(vpp, v3)

        # Calculate the distances of projected points
        vp1 = np.linalg.norm(v1xvp, axis=1)
        vp2 = np.linalg.norm(vpxv2, axis=1)
        vp3 = np.linalg.norm(vppxv3, axis=1)

        # Find the triangle that the point projects onto
        ind = np.where((vp1 + vp2 + vp3 - A2) < np.min(A2) * 1e-3)[0]
        d, in_idx = np.min(np.sum((vt[ind, :] - vp[ind, :])**2, axis=1)), np.argmin(np.sum((vt[ind, :] - vp[ind, :])**2, axis=1))

        li = np.min(A2) * 5e-3

        # Recursive refinement if numerical errors are larger than expected
        while len(ind) == 0:
            li += np.min(A2) * 2e-3
            ind = np.where((vp1 + vp2 + vp3 - A2) < li)[0]
            d, in_idx = np.min(np.sum((vt[ind, :] - vp[ind, :])**2, axis=1)), np.argmin(np.sum((vt[ind, :] - vp[ind, :])**2, axis=1))

        while d > 1e-4:
            li += np.min(A2) * 2e-3
            ind = np.where((vp1 + vp2 + vp3 - A2) < li)[0]
            d, in_idx = np.min(np.sum((vt[ind, :] - vp[ind, :])**2, axis=1)), np.argmin(np.sum((vt[ind, :] - vp[ind, :])**2, axis=1))

        nt = ind[in_idx]

        A[ii, T[nt, 0]] = vp2[nt] / A2[nt]
        A[ii, T[nt, 1]] = vp1[nt] / A2[nt]
        A[ii, T[nt, 2]] = 1 - (vp1[nt] + vp2[nt]) / A2[nt]

    return A



def build_BEM_model(fol, sub, se):

    """
    This function builds a three-layer Boundary Element Method (BEM) head model from an anatomical
    T1w MRI. It segments the brain, skull, and scalp tissues using morphological operations, 
    and generates surface meshes for each tissue layer.
    """
    
    # ------------------------------------------------------------
    # Load anatomical images
    # ------------------------------------------------------------
    nii_path = glob.glob(f"{fol}anat/{sub}_{se}space-nativepro_T1w.nii*")[0]
    brain_path = glob.glob(f"{fol}anat/{sub}_{se}space-nativepro_T1w_brain.nii*")[0]

    mr = nib.load(nii_path)
    im0 = nib.load(brain_path)
    pixdim = mr.header.get_zooms()
    pxz = np.prod(pixdim) ** (1/3)
    rz = pixdim[2]

    # Connected components labeling
    mr_img = mr.get_fdata()
    iseg = im0.get_fdata()
    ins= iseg > 0
    cc, num_features = ndi.label(ins) 
    sizes = np.array([np.sum(cc == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(sizes) + 1  # The index of the largest component
    ins = (cc == largest_region)
    # Define spheres for morphological operations
    gx, gy, gz = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6), np.arange(-5, 6))
    r = gx**2 + gy**2 + gz**2
    sph5 = r<=25
    gx, gy, gz = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), np.arange(-4, 5))
    r = gx**2 + gy**2 + gz**2
    sph4 = r<=16
    imas = ndi.binary_dilation(ins, structure=sph5, iterations=round(20 / 5 / pxz))

    # Threshold calculation
    m = mr_img[:, :, int(round(96/rz))-1 :].max(axis=0)
    m = np.sort(m.flatten())
    mm = np.convolve(np.diff(m), np.ones(int(round(80/pxz)))/round(80/pxz), mode="same")
    m = np.sort(m.flatten())
    idx = np.argmax(mm[int(round(8000/pxz)) : int(round(44000/pxz))])
    thr = m[idx + int(round(8000/pxz))]

    # Apply threshold to image
    sca = ((mr_img > thr) & imas) | ins
    cc, num_features = ndi.label(sca)
    sizes = np.array([np.sum(cc == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(sizes) + 1
    sca = (cc == largest_region)

    # Morphological closing
    sca = ndi.binary_dilation(sca, structure=sph5, iterations=round(8 / 5 / pxz))
    sca = ndi.binary_erosion(sca, structure=sph5, iterations=round(8 / 5 / pxz), border_value=1)
    imax = ndi.binary_dilation(sca, structure=sph5, iterations=round(40 / 5 / pxz))
    sca2 = ((mr_img > thr) & imax) | sca
    cc, num_features = ndi.label(sca2)
    sizes = np.array([np.sum(cc == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(sizes) + 1
    sca2 = (cc == largest_region)

    # ------------------------------------------------------------
    # Remove ears using ANTs transforms
    # ------------------------------------------------------------
    aff = f"{fol}xfm/{sub}_{se}from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_0GenericAffine.mat"
    warp = glob.glob(f"{fol}xfm/{sub}_{se}from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_1Warp.nii*")[0]
    transformlist = [warp, aff]
 
    pet = ants.apply_transforms_to_points(
        dim=3,
        transformlist=transformlist,
        points=pd.DataFrame(np.array([[86.1,20,-48],[-85.8,20,-48]]),columns=['x','y','z']),
        whichtoinvert=[False, False]
    )
    # Convert MNI pts → voxel coords
    x = np.round((-pet.x.to_numpy() - mr.header["qoffset_x"]) / pixdim[0]).astype(int)
    # Remove top/bottom slices based on ear height
    zn = np.zeros(50, dtype=int)
    for ii in range(50):
        _, z = np.where(sca2[ii,:,:])
        if len(z): zn[ii] = z.max()
    h = min(np.where(zn > 149)[0][0], x[0] - 2)
    sca2[:h,:,:] = False

    zn = np.zeros(50, dtype=int)
    for ii in range(50):
        _, z = np.where(sca2[-ii-1,:,:])
        if len(z): zn[ii] = z.max()
    h = max(sca2.shape[0]-np.where(zn > 149)[0][0], x[1] + 3)
    sca2[h:,:,:] = False

    # Closing to remove holes
    ni =  round(8/5/pxz)  
    sca3 = np.zeros((sca2.shape[0]+20*ni, sca2.shape[1]+20*ni, sca2.shape[2]+20*ni), dtype=bool)
    sca3 [2*ni*5: -2*ni*5,2*ni*5: -2*ni*5,2*ni*5: -2*ni*5] = sca2
    sca3 = ndi.binary_dilation(sca3, structure=sph5, iterations=ni)
    sca3 = ndi.binary_erosion(sca3, structure=sph5, iterations=ni)
    sca2= sca3 [2*ni*5: -2*ni*5,2*ni*5: -2*ni*5,2*ni*5: -2*ni*5]
    cc, num_features = ndi.label(~sca2)
    sizes = np.array([np.sum(cc == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(sizes) + 1
    sca2= (cc != largest_region)
    sca2 = ndi.binary_erosion(sca2, structure=sph5, iterations=ni, border_value=1)
    sca2 = ndi.binary_dilation(sca2, structure=sph5, iterations=ni)
 
    # Threshold calculation
    tmp = mr_img * (sca & ~ins)
    tmp = tmp[tmp > 0]
    tmp.sort()
    factor = round(800 / pxz)
    trunc = (len(tmp)//factor)*factor
    tmp = tmp[:trunc+1]
    mm = np.mean(np.reshape(np.diff(tmp), (-1, factor)), axis=1)
    m = np.mean(np.reshape(tmp[:-1], (-1, factor)), axis=1)
    N = round(len(m)/50)
    mm = ndi.uniform_filter1d(ndi.maximum_filter1d(mm, N), N)
    u5=round(m.shape[0]*.15)
    im2 = np.where(
        (mm[u5-2:-2] <= mm[u5-1:-1]) &
        (mm[u5:] <= mm[u5-1:-1])
    )[0][0]
    thr = m[u5 + im2]
    # Apply thresholds and morphological operations
    sku = ((mr_img < thr) & sca) | ins
    cc, num_features = ndi.label(sku, structure=ndi.generate_binary_structure(3, 3))
    sizes = np.array([np.sum(cc == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(sizes) + 1
    sku = (cc == largest_region)
    sku = ndi.binary_dilation(sku, structure=sph5, iterations=round(12/5/pxz))
    sku = ndi.binary_erosion(sku, structure=sph5, iterations=round(12/5/pxz), border_value=1)
    cc, num_features = ndi.label(~sku)
    sizes = np.array([np.sum(cc == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(sizes) + 1
    sku = (cc != largest_region)
    sku = sku | ndi.binary_dilation(ins, sph4)    

    # ------------------------------------------------------------
    # Fill 2D holes slice-by-slice for ins, sku, sca2
    # ------------------------------------------------------------
    sq=ndi.generate_binary_structure(2,2)
    sq[1,1]=False
    for nz in range(sku.shape[2]):
        for arr in [ins, sku, sca2]:
            cc, num_features = ndi.label(~arr[:,:,nz],structure=sq)
            sizes = [(cc == i).sum() for i in range(1,num_features+1)]
            largest_region = np.argmax(sizes) + 1
            arr[:,:,nz] = cc != largest_region

    sku = sku & ndi.binary_erosion(sca2, sph5)
    
    # ------------------------------------------------------------
    # Surface extraction for brain (p1)
    # ------------------------------------------------------------

    sph1 = ndi.generate_binary_structure(3, 1)
    ins2 = ndi.binary_dilation(ins, sph1)
    xs, ys, zs = np.where(ins2 & (~ins))

    # Convert voxel → mm
    x_mm = pixdim[0]*xs + mr.header["qoffset_x"]
    y_mm = pixdim[1]*ys + mr.header["qoffset_y"]
    z_mm = pixdim[2]*zs + mr.header["qoffset_z"]

    pts = np.column_stack((x_mm, y_mm, z_mm))
    c3 = np.median(pts, axis=0)

    ps = pts - c3
    rs = np.linalg.norm(ps, axis=1)

    p, t1 = esfera(1, 5)
    pes = ps / rs[:,None]

    r = np.zeros(p.shape[0])
    for i in range(len(p)):
        r[i] = np.mean(rs[ pes @ p[i,:] > 0.9962 ])
    r[r < 50] = 50

    p1 = p * r[:,None] + c3

    # ------------------------------------------------------------
    # Surface extraction for skull (p2)
    # ------------------------------------------------------------
    sku2 = ndi.binary_dilation(sku, sph1)
    xs, ys, zs = np.where(sku2 & (~sku))
    x_mm = pixdim[0]*xs + mr.header["qoffset_x"]
    y_mm = pixdim[1]*ys + mr.header["qoffset_y"]
    z_mm = pixdim[2]*zs + mr.header["qoffset_z"]

    pts = np.column_stack((x_mm, y_mm, z_mm))
    ps = pts - c3
    rs = np.linalg.norm(ps, axis=1)

    p, t2 = esfera(1, 5)
    pes = ps / rs[:,None]

    r = np.zeros(p.shape[0])
    for i in range(len(p)):
        r[i] = np.mean(rs[ pes @ p[i,:] > 0.9962 ])
    r[r < 55] = 55

    p2 = p * r[:,None] + c3

    # ------------------------------------------------------------
    # Surface extraction for scalp (p3)
    # ------------------------------------------------------------    
    sca3 = np.zeros((sca2.shape[0]+2, sca2.shape[1]+2, sca2.shape[2]+2), dtype=bool)
    sca3 [1:-1, 1:-1, 1:-1] = sca2 
    sca2 = ndi.binary_dilation(sca3, sph1)
    xs, ys, zs = np.where(sca2 & (~sca3))
    xs -= 1; ys -= 1; zs -= 1

    x_mm = pixdim[0]*xs + mr.header["qoffset_x"]
    y_mm = pixdim[1]*ys + mr.header["qoffset_y"]
    z_mm = pixdim[2]*zs + mr.header["qoffset_z"]

    pts = np.column_stack((x_mm, y_mm, z_mm))
    ps = pts - c3
    rs = np.linalg.norm(ps, axis=1)

    p, t3 = esfera(1, 4, 1)
    pes = ps / rs[:,None]

    r = np.zeros(p.shape[0])
    for i in range(len(p)):
        r[i] = np.mean(rs[ pes @ p[i,:] > 0.9962 ])
    r[r < 60] = 60

    p3 = p * r[:,None] + c3

    return p1, p2, p3, t1, t2, t3, c3, pes, rs
