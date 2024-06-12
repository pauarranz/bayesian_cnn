import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours

import pylidc as pl
from pylidc.utils import consensus


def plot_scan(pid):
    # Query for all CT scans with desired traits.
    scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1,
                                     pl.Scan.pixel_spacing <= 0.6)
    print(scans.count())

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

    print(len(scan.annotations))

    nods = scan.cluster_annotations()

    print("%s has %d nodules." % (scan, len(nods)))

    scan.visualize(annotation_groups=nods)


def plot_annotation():
    ann = pl.query(pl.Annotation).first()
    vol = ann.scan.to_volume()

    padding = [(30, 10), (10, 25), (0, 0)]

    mask = ann.boolean_mask(pad=padding)
    bbox = ann.bbox(pad=padding)

    fig, ax = plt.subplots(1, 2, figsize=(5, 3))

    ax[0].imshow(vol[bbox][:, :, 2], cmap=plt.cm.gray)
    ax[0].axis('off')

    ax[1].imshow(mask[:, :, 2], cmap=plt.cm.gray)
    ax[1].axis('off')

    plt.tight_layout()
    # plt.savefig("../images/mask_bbox.png", bbox_inches="tight")
    plt.show()


def plot_annotation_consenus(pid):
    # Query for a scan, and convert it to an array volume.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = scan.to_volume()

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    anns = nods[0]

    # Perform a consensus consolidation and 50% agreement level.
    # We pad the slices to add context for viewing.
    cmask, cbbox, masks = consensus(anns, clevel=0.5,
                                    pad=[(20, 20), (20, 20), (0, 0)])

    # Get the central slice of the computed bounding box.
    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

    # Set up the plot.
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=0.5)

    # Plot the annotation contours for the kth slice.
    colors = ['r', 'g', 'b', 'y']
    for j in range(len(masks)):
        for c in find_contours(masks[j][:, :, k].astype(float), 0.5):
            label = "Annotation %d" % (j + 1)
            plt.plot(c[:, 1], c[:, 0], colors[j], label=label)

    # Plot the 50% consensus contour for the kth slice.
    for c in find_contours(cmask[:, :, k].astype(float), 0.5):
        plt.plot(c[:, 1], c[:, 0], '--k', label='50% Consensus')

    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    # plt.savefig("../images/consensus.png", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    patient_id = 'LIDC-IDRI-0004'
    plot_scan(pid=patient_id)
    plot_annotation_consenus(pid=patient_id)
