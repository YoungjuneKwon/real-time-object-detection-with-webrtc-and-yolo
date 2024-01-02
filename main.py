""" WebRTC webcam demo """
import argparse
import asyncio
import json
import logging
import os
import cv2

from aiohttp import web
import numpy as np
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

ROOT = os.path.dirname(__file__)
MODEL = "yolov7-tiny_480x640.onnx"

# pylint: disable=too-many-instance-attributes
class YOLOVideoStreamTrack(VideoStreamTrack):
    """
    A video track thats returns camera track with annotated detected objects.
    """
    def __init__(self, track, conf_thres=0.7, iou_thres=0.5):
        super().__init__()  # don't forget this!
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        self.track = track

        # Initialize MODEL
        self.net = cv2.dnn.readNet(MODEL)
        input_shape = os.path.splitext(os.path.basename(MODEL))[0].split('_')[-1].split('x')
        self.input_height = int(input_shape[0])
        self.input_width = int(input_shape[1])

        with open('coco.names', 'r', encoding='utf-8') as f:
            self.class_names = list(map(lambda x: x.strip(), f.readlines()))
        self.colors = np.random.default_rng(3).uniform(0, 255, size=(len(self.class_names), 3))

        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.input_frame = None
        self.flag_proceed = False
        self.output_frame = None
        asyncio.create_task(self.store_input_frame())
        asyncio.create_task(self.process_output_frame())

    async def store_input_frame(self):
        """ Store input frame """
        while True:
            frame = await self.track.recv()
            self.input_frame = frame.to_ndarray(format="bgr24")
            self.flag_proceed = False

    async def process_output_frame(self):
        """ Process output frame """
        while self.input_frame is None:
            await asyncio.sleep(0.2)
        while True:
            while self.flag_proceed:
                await asyncio.sleep(0.2)
                continue
            pts, time_base = await self.next_timestamp()
            frame = self.input_frame
            boxes, scores, class_ids = self.detect(frame)
            frame = self.draw_detections(frame, boxes, scores, class_ids)
            frame = VideoFrame.from_ndarray(frame, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
            self.output_frame = frame
            self.flag_proceed = True

    async def recv(self):
        """ Receive frame """
        while self.output_frame is None:
            await asyncio.sleep(0.2)
        return self.output_frame

    def prepare_input(self, image):
        """ Prepare input image """
        # pylint: disable=attribute-defined-outside-init
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        return input_img

    def detect(self, frame):
        """ Detect objects """
        input_img = self.prepare_input(frame)
        blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0)
        # Perform inference on the image
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outputs = self.net.forward(self.output_names)

        boxes, scores, class_ids = self.process_output(outputs)
        return boxes, scores, class_ids

    def process_output(self, output):
        """ Process output """
        predictions = np.squeeze(output[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                              self.conf_threshold, self.iou_threshold)
        if len(indices) > 0:
            indices = indices.flatten()

        return boxes[indices], scores[indices], class_ids[indices]

    def rescale_boxes(self, boxes):
        """ Rescale boxes """
        input_shape = np.array([self.input_width, self.input_height,
                                self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def extract_boxes(self, predictions):
        """ Extract boxes """
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xywh format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        return boxes_

    def draw_detections(self, frame, boxes, scores, class_ids):
        """ Draw detections """
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box.astype(int)
            color = self.colors[class_id]

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'
            cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        return frame

async def index(_):
    """ Serve the client-side application """
    with open(os.path.join(ROOT, "index.html"), "r", encoding='utf-8') as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)

async def javascript(_):
    """ Client JavaScript """
    with open(os.path.join(ROOT, "client.js"), "r", encoding='utf-8') as f:
        content = f.read()
    return web.Response(content_type="application/javascript", text=content)

async def my_ice_servers(_):
    """ My ICE servers """
    with open(os.path.join(ROOT, "myIceServers.json"), "r", encoding='utf-8') as f:
        content = f.read()
    return web.Response(content_type="application/json", text=content)

async def on_offer(request):
    """ WebRTC offer """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pcs.add(pc := RTCPeerConnection())

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(YOLOVideoStreamTrack(track))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

pcs = set()

async def on_shutdown(_):
    """ Close peer connections """
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", on_offer)
    app.router.add_get("/myIceServers", my_ice_servers)
    web.run_app(app, host=args.host, port=args.port)
