# Learnings
## Transform bounding box dimensions
For OWLv2 model, the author provided the transformation of image but not the bounding boxes where as the model predicts the bounding boxes on target image scale. This was noticed after few epochs, the loss is reducing but the bbox predictions are moving away from expected location when visualized.

## Image guided querying to include only the item interested in
OWLv2 model provides capability to query an image when an embedding taken from another image. We tried 

## Loading metadata from multiple files vs one big file