from MRCFile import MRCFile

# Setup file object
f = MRCFile('zika_153.mrc')

# Actually load volume slices from disk
f.load_all_slices()
print(f.slices[:,:,0].shape)
print()
