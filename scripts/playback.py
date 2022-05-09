from argparse import ArgumentParser
import cv2
import numpy as np

import pyk4a
from pyk4a import Config, PyK4A

from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")

#formats images - copied from web
def format_yolov5(source):

    # put the image in square big enough
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source
    
    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)
    return result

def unwrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape


    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    return class_ids, confidences, boxes    



def play(playback: PyK4APlayback):
    net = cv2.dnn.readNet('../best.onnx')

    while True:
        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                cv2.imshow("Color", convert_to_bgra_if_required(playback.configuration["color_format"], capture.color))
                

            if capture.depth is not None:
                cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
    cv2.destroyAllWindows()




def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    parser.add_argument("FILE", type=str, help="Path to MKV file written by k4arecorder")

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    play(playback)

    playback.close()


if __name__ == "__main__":
    main()