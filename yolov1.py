import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 416
IM_HEIGHT = 416

# zaladowanie sieci
LABELS_FILE='C:/Users/CarlaPZ/Desktop/Test_sieci_GPU/mscoco_labels.names'
CONFIG_FILE='C:/Users/CarlaPZ/Desktop/Test_sieci_GPU/yolov4-tiny.cfg'
WEIGHTS_FILE='C:/Users/CarlaPZ\Desktop/Test_sieci_GPU/yolov4-tiny.weights'

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

CONFIDENCE_THRESHOLD=0.3
LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
k=0
nazwa_pliku = 'z_yolo.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
wideo_wyj = cv2.VideoWriter(nazwa_pliku,fourcc, 10.0, (IM_WIDTH,IM_HEIGHT))
wideo_wyj2 = cv2.VideoWriter("bez_yolo.avi",fourcc, 10.0, (IM_WIDTH,IM_HEIGHT))


def process_img2(image):
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    wideo_wyj2.write(i3)
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0
   

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    image = np.array(i3)
    global k
    k=k+1
    
    (H, W) = [IM_HEIGHT, IM_WIDTH]    
    # określenie watstw wyjściowych, których potrzebujemy
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    # inicjalizacja list obwiedni wykrytych obiektow, pewności i wskaźników klas(ID) 
    boxes = []
    confidences = []
    classIDs = []

    # pętla iterująca po każdej warstwie wyjściowej
    for output in layerOutputs:
        
        for detection in output:
            # ekstrakcja klas i pewności 
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filtracaj wykrytych obiektów
            if confidence > CONFIDENCE_THRESHOLD:
                # tworzenie obwiedni(yolo zwraca x i y środka obiektu)
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))


                # aktualizacja listy obwieni pewności i wskaźników klas
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # zastoswoanie NMS(non-maximum suppresion) w celu ofiltrowania niepotrzebnych obwiedni
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)

    # zabezpieczenie przed zerową ilością obietów wykrytych
    if len(idxs) > 0:
    
        for i in idxs.flatten():
            # pobranie i narysowanie obwieni
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
           # print(LABELS[classIDs[i]])
           
    
    wideo_wyj.write(image)
    cv2.imshow("", image)
    cv2.waitKey(1)
    return image/255.0


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)
    
    spawn_point = carla.Transform(carla.Location(x=56.961758, y=-4.278575, z=0.275307), carla.Rotation(pitch=0.000000, yaw=-179.144165, roll=0.000000))
    print(spawn_point)
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)  
    print(vehicle.get_location())
    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # pozyskanie kamery 
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # ustawnienie parametrów kamery
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # ustawienie pozycji kamery(na przodzie samochodu)
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # umiejscowienie naszej kamey w symulacji
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # dodanie sensora do listy aktorów
    actor_list.append(sensor)

    # pobranie obrazu z kamery i przekazanie go do funkcji dokonującej wykrywania
    sensor.listen(lambda data: process_img(data))
    
    # sensor.listen(lambda data: process_img2(data))
    # po 15 sekunach symulacja się zakończy
  
    time.sleep(45)

#    while(True):
        
#       if cv2.waitKey(1) & 0xFF == ord('q'):
#            break


finally:
    print('destroying actors')
    print("Function process_img has been used ",k, " times.")
    for actor in actor_list:
        actor.destroy()
    print('done.')