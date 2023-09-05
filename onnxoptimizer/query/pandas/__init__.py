import gorilla
import onnxoptimizer.query.pandas.patch as pdp

patches = gorilla.find_patches([pdp])

for patch in patches:
    gorilla.apply(patch)

