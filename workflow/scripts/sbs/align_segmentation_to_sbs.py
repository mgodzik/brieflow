from tifffile import imread, imwrite

from lib.sbs.align_cycles import align_seg_to_sbs

# load segmentation stack
seg_stack = imread(snakemake.input[0])

# load aligned SBS stack and use DAPI from first cycle as reference
aligned_sbs = imread(snakemake.input[1])
ref_dapi = aligned_sbs[0, 0]

aligned_seg = align_seg_to_sbs(
    seg_stack,
    ref_dapi,
    dapi_index=snakemake.params.dapi_index,
    upsample_factor=snakemake.params.upsample_factor,
    window=snakemake.params.window,
)

imwrite(snakemake.output[0], aligned_seg)
