import pixellib
from pixellib.semantic import semantic_segmentation
import time

ins = semantic_segmentation()
ins.load_pascalvoc_model("pascal.h5")
start1 = time.time()
ins.segmentAsPascalvoc("cycle1.jpg")
end1 = time.time()
print((end1-start1), "seconds")
start2 = time.time()
ins.segmentAsPascalvoc("cycle2.jpg" )
end2 = time.time()
print((end2-start2), "seconds")
start3 = time.time()
ins.segmentAsPascalvoc("former.jpg")
end3 = time.time()
print((end3-start3), "seconds")
start4 = time.time()
ins.segmentAsPascalvoc("ade_test1.jpg")
end4 = time.time()
print((end4-start4), "seconds")
start5 = time.time()
ins.segmentAsPascalvoc("fore6.jpg")
end5 = time.time()
print((end5-start5), "seconds")


#print("Average time taken {:.2f} seconds") 
#ins.segmentImage(".jpg", show_bboxes=True)

#ins.segmentAsAde20k("images/download1.jpg", overlay = True, output_image_name = "images/aa1.jpg")
"""ins.segmentAsAde20k("photos/two1.jpg", overlay=True, output_image_name = "photos/a1.jpg")
ins.segmentAsAde20k("photos/two2.jpg", overlay=True, output_image_name = "photos/a2.jpg")
ins.segmentAsAde20k("photos/two3.jpg", overlay=True, output_image_name = "photos/a3.jpg") 
ins.segmentAsAde20k("photos/family1.jpg", overlay=True, output_image_name = "photos/a6.jpg")
ins.segmentAsAde20k("photos/family2.jpg", overlay=True, output_image_name = "photos/a7.jpg")
ins.segmentAsAde20k("photos/family3.jpg", overlay=True, output_image_name = "photos/a8.jpg") """

