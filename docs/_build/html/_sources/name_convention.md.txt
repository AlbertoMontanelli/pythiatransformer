<!-- docs/name_convention.md -->
# Naming conventions

All input, output, and plot filenames follow a default naming
convention unique for all ``pythiatransformer`` package.
Each file name includes a unique ``_{suffix}`` tag identifying the
specific dataset/plot/filename depending on the number of events.
This suffix is the only user-controlled identifier, and it must remain
consistent across all stages of the workflow (data generation,
preprocessing, training, inference, and plotting).
File names themselves should **not be changed manually**, as they are
managed automatically by the pipeline and ensure consistent data
linkage.