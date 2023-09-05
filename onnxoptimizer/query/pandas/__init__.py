import gorilla
import onnxoptimizer.query.pandas as pd

patches = gorilla.find_patches([pd])

for patch in patches:
    gorilla.apply(patch)

