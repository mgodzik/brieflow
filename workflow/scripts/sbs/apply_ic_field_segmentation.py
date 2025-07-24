from tifffile import imread, imwrite

from lib.shared.illumination_correction import apply_ic_field

# load aligned segmentation image
aligned_seg = imread(snakemake.input[0])

# load illumination correction field
ic_field = imread(snakemake.input[1])

corrected = apply_ic_field(aligned_seg, correction=ic_field)

imwrite(snakemake.output[0], corrected)
