import pandas as pd
from tifffile import imread

from lib.sbs.call_reads import call_reads

# load bases data
bases_data = pd.read_csv(snakemake.input[0], sep="\t")

# load peaks data
peaks_data = imread(snakemake.input[1])

# call reads
# GPX4 pilot: activate the gated raw-G/T recall + cycle-1 {C,T} fix.
# Values come from the rule params (config-overridable; validated defaults there).
reads_data = call_reads(
    bases_data=bases_data,
    peaks_data=peaks_data,
    method=snakemake.params.call_reads_method,
    gt_raw_threshold=snakemake.params.gt_raw_threshold,
    cycle1_ct_from_raw=snakemake.params.cycle1_ct_from_raw,
)

# save reads data
reads_data.to_csv(snakemake.output[0], index=False, sep="\t")
