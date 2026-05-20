from ultralytics import YOLO

model = YOLO('yolo26n.pt') 

results = model('giraffe.jpg')

results[0].show()

results[0].save(filename='giraff_output.jpeg')